"""
ETHRAEON Claude Opus 4.7 Agent for ARC-AGI-3
Uses extended thinking + vision to reason about game state.
"""

import base64
import logging
import os
from io import BytesIO
from typing import Any

import anthropic
import numpy as np
from arcengine import FrameData, GameAction, GameState
from PIL import Image

from ..agent import Agent

logger = logging.getLogger(__name__)

MODEL = "claude-opus-4-7"
MAX_ACTIONS = 400
THINKING_BUDGET = 8000  # tokens for extended thinking per action
MSG_HISTORY_LIMIT = 12  # keep last N turns in context


COLOR_PALETTE = {
    0: (20, 20, 20),       # Black (background)
    2: (220, 50, 50),      # Red
    4: (50, 200, 50),      # Green
    5: (140, 140, 140),    # Gray
    6: (50, 50, 220),      # Blue
    7: (230, 220, 50),     # Yellow
    8: (220, 140, 30),     # Orange
    9: (140, 30, 140),     # Purple
    10: (240, 240, 240),   # White
    11: (160, 160, 160),   # Light Gray
    12: (255, 100, 100),   # Player highlight
}
SCALE = 12  # pixels per grid cell


def render_grid(frame_3d: list) -> str:
    """Render a 3D ARC grid to a base64 PNG for Claude vision."""
    grid = np.array(frame_3d[0], dtype=np.uint8)
    h, w = grid.shape
    img = Image.new("RGB", (w * SCALE, h * SCALE), (20, 20, 20))
    pixels = img.load()
    for row in range(h):
        for col in range(w):
            color = COLOR_PALETTE.get(int(grid[row, col]), (200, 200, 200))
            for dy in range(SCALE):
                for dx in range(SCALE):
                    pixels[col * SCALE + dx, row * SCALE + dy] = color
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


SYSTEM_PROMPT = """\
You are an expert ARC-AGI-3 game agent. You observe a grid-based game and must choose the best action.

## GAME RULES
- You control a character on a 2D grid.
- Goal: transform the KEY to match the EXIT DOOR shape/color, then navigate to the exit.
- Win condition: complete all required levels.
- You have 3 lives, each with 25 energy. Losing all energy = lose a life.

## GRID OBJECTS (color codes)
- Black (0): empty space / floor
- Red (2): player or key element
- Green (4): walls or structures
- Gray (5/11): border / inactive elements
- Blue (6): refiller (restores energy) or key element
- Yellow (7): rotator (transforms key rotation)
- Orange (8): used/depleted energy indicator
- Purple (9): special objects
- White (10): exit door or key match indicator
- Your character moves 4 cells per action in cardinal directions.

## ACTIONS
- ACTION1: move UP
- ACTION2: move DOWN
- ACTION3: move LEFT
- ACTION4: move RIGHT
- ACTION5: interact / use object
- ACTION6: rotate key
- ACTION7: special action
- RESET: reset the current level (costs a life)

## STRATEGY
1. Study the game image carefully.
2. Identify: your character position, key shape, exit door, rotators, refillers.
3. Reason step by step about what action best advances toward winning.
4. Prefer actions that progress toward the goal; avoid RESET unless stuck.

## RESPONSE FORMAT
Respond with ONLY a JSON object:
{"action": "ACTION1", "reasoning": "brief explanation"}

Valid actions: ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, ACTION6, ACTION7, RESET
"""


ACTION_MAP = {
    "ACTION1": GameAction.ACTION1,
    "ACTION2": GameAction.ACTION2,
    "ACTION3": GameAction.ACTION3,
    "ACTION4": GameAction.ACTION4,
    "ACTION5": GameAction.ACTION5,
    "ACTION6": GameAction.ACTION6,
    "ACTION7": GameAction.ACTION7,
    "RESET": GameAction.RESET,
}


class ClaudeAgent(Agent):
    """Claude Opus 4.7 agent with extended thinking and vision for ARC-AGI-3."""

    MAX_ACTIONS = MAX_ACTIONS

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.message_history: list[dict] = []
        self.hypothesis = "No hypothesis yet — exploring."
        self.levels_at_start = 0

    def choose_action(
        self,
        frames: list[FrameData],
        current_frame: FrameData | None,
    ) -> GameAction:
        frame = current_frame or frames[-1]

        # Render the current game state as an image
        try:
            img_b64 = render_grid(frame.frame)
        except Exception as e:
            logger.warning(f"Render failed: {e} — using ACTION1 fallback")
            return GameAction.ACTION1

        # Build the user message with vision + context
        available = [f"ACTION{a}" for a in (frame.available_actions or [1, 2, 3, 4])]
        user_content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_b64,
                },
            },
            {
                "type": "text",
                "text": (
                    f"Action #{self.action_counter} | "
                    f"Levels completed: {frame.levels_completed}/{frame.win_levels} | "
                    f"State: {frame.state} | "
                    f"Available actions: {available}\n"
                    f"Current hypothesis: {self.hypothesis}\n"
                    f"Choose your next action."
                ),
            },
        ]

        # Trim history
        if len(self.message_history) > MSG_HISTORY_LIMIT * 2:
            self.message_history = self.message_history[-(MSG_HISTORY_LIMIT * 2):]

        messages = self.message_history + [{"role": "user", "content": user_content}]

        try:
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=THINKING_BUDGET + 512,
                thinking={
                    "type": "enabled",
                    "budget_tokens": THINKING_BUDGET,
                },
                system=SYSTEM_PROMPT,
                messages=messages,
            )

            # Extract text response
            text = ""
            for block in response.content:
                if block.type == "text":
                    text = block.text.strip()
                    break

            # Parse action
            import json
            import re
            action_name = "ACTION1"
            try:
                # Try direct JSON parse
                match = re.search(r'\{[^}]+\}', text, re.DOTALL)
                if match:
                    parsed = json.loads(match.group())
                    action_name = parsed.get("action", "ACTION1").upper()
                    self.hypothesis = parsed.get("reasoning", self.hypothesis)[:500]
            except Exception:
                # Fallback: find ACTION\d in text
                m = re.search(r'ACTION[1-7]|RESET', text)
                if m:
                    action_name = m.group()

            # Validate against available actions
            if action_name not in available and action_name != "RESET":
                action_name = available[0] if available else "ACTION1"

            chosen = ACTION_MAP.get(action_name, GameAction.ACTION1)

            # Update history with assistant response (text only for context efficiency)
            self.message_history.append({"role": "user", "content": user_content})
            self.message_history.append({
                "role": "assistant",
                "content": text[:800],  # truncate for history efficiency
            })

            logger.info(f"Claude chose {action_name}: {self.hypothesis[:80]}")
            return chosen

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e} — fallback ACTION1")
            return GameAction.ACTION1
        except Exception as e:
            logger.error(f"Agent error: {e} — fallback ACTION1")
            return GameAction.ACTION1

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

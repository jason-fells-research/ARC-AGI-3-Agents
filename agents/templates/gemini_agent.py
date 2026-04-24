"""
ETHRAEON Multi-LLM Agent for ARC-AGI-3
Primary: Perplexity sonar-pro (vision)
Fallback: Gemini 2.5 Flash (key rotation)
Final: game-mechanic heuristic

Game mechanics (64×64 grid):
- Rows 0-4:  UI (health row 1, energy row 2, lives top-right)
- Rows 5-54: Main playfield
- Rows 55-64: Key display (cols 2-11)
- Player: 3×3 [[0,0,0],[4,4,4],[4,4,4]], also marked with color 12
- Exit door: 4×4 with color 11 (gray) border
- Rotator: 3×3 pattern with center cells (r+1,c+1)=7 and (r+1,c+2)=7
- Refiller: 2×2 all color 6
- Wall: color 10 (white) — impassable
- Floor: color 8 (orange) — walkable
- Each action moves player 4 grid cells
"""

import base64
import json
import logging
import re
import time
from io import BytesIO
from typing import Any

import numpy as np
import requests
from arcengine import FrameData, GameAction, GameState
from PIL import Image

from ..agent import Agent

logger = logging.getLogger(__name__)

MAX_ACTIONS = 400
MSG_HISTORY_LIMIT = 4
LLM_CALL_INTERVAL = 4.0  # seconds between LLM calls

# --- API credentials ---
PPLX_KEY = "PPLX_KEY_REMOVED"
PPLX_MODEL = "sonar-pro"
PPLX_URL = "https://api.perplexity.ai/chat/completions"

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_BASE = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)
GEMINI_KEYS = [
    "GEMINI_KEY_1_REMOVED",
    "GEMINI_KEY_2_REMOVED",
    "GEMINI_KEY_3_REMOVED",
]

# Rendering colors
COLOR_PALETTE = {
    0: (0, 0, 0),        # Black: empty / player top
    2: (255, 0, 0),      # Red: key element
    4: (0, 200, 0),      # Green: wall or player body
    5: (128, 128, 128),  # Gray: border
    6: (0, 100, 255),    # Blue: refiller or energy cell
    7: (255, 220, 0),    # Yellow: rotator center
    8: (200, 100, 0),    # Orange: walkable floor
    9: (120, 0, 120),    # Purple: rotator border
    10: (240, 240, 240), # White: wall (impassable)
    11: (160, 160, 160), # Light gray: EXIT DOOR border
    12: (255, 80, 80),   # Bright red: player marker (top of player)
}
SCALE = 10

SYSTEM_PROMPT = """\
You are an ARC-AGI-3 game agent. 64×64 grid, top-down view.

## GRID LAYOUT
- Rows 0-4: UI (health=row1, energy=row2, lives=top-right)
- Rows 5-54: Playfield (player, exit, rotators, refillers, walls)
- Rows 55-64: Key display (bottom-left, cols 2-11)

## KEY OBJECTS
- PLAYER: 3×3 block. Top row is black (0), bottom 2 rows bright green (4). Marked with bright red (12) at top.
- EXIT DOOR: 4×4 square with LIGHT GRAY (11) border. Contains the target key pattern inside.
- ROTATOR: ~3×3 block with YELLOW (7) center cells. Walk into it to rotate your key.
- REFILLER: 2×2 BLUE (6) block. Restores your 25 energy units.
- WALL: WHITE (10) blocks — cannot walk through.
- FLOOR: ORANGE (8) — walkable.

## WIN CONDITION
1. Key (bottom-left display) must match exit door's center pattern.
2. Walk into exit door when key matches.

## STRATEGY
1. Look at bottom-left (key) and exit door center. Do patterns match?
2. If NO: navigate to ROTATOR (yellow/purple block). Walk into it to rotate key.
3. Repeat rotating until key matches exit.
4. If YES: navigate directly to EXIT DOOR.
5. Watch energy (blue cells in row 2). Use REFILLER if <10 energy remaining.

## ACTIONS
- ACTION1: up   ACTION2: down   ACTION3: left   ACTION4: right
- RESET: restart (costs a life — AVOID)

## OUTPUT
JSON only: {"action": "ACTION2", "reasoning": "rotator is below player, navigating down to reach it"}
"""


def render_grid(frame_3d: list) -> str:
    grid = np.array(frame_3d[0], dtype=np.uint8)
    h, w = grid.shape
    img = Image.new("RGB", (w * SCALE, h * SCALE), (10, 10, 10))
    pixels = img.load()
    for row in range(h):
        for col in range(w):
            color = COLOR_PALETTE.get(int(grid[row, col]), (200, 0, 200))
            for dy in range(SCALE):
                for dx in range(SCALE):
                    pixels[col * SCALE + dx, row * SCALE + dy] = color
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Game state analysis ──────────────────────────────────────────────────────

def find_player(grid: np.ndarray) -> tuple[int, int] | None:
    """Find player center. Color 12 marks the top of the player."""
    locs = list(zip(*np.where(grid == 12)))
    if locs:
        r, c = locs[0]
        return (r + 1, c + 1)  # center (one row below marker, centered)
    # Fallback: scan for green 3x3 player block
    rows, cols = grid.shape
    for r in range(5, min(rows - 2, 55)):
        for c in range(cols - 2):
            if ((grid[r, c:c+3] == 0).all() and
                    (grid[r+1, c:c+3] == 4).all() and
                    (grid[r+2, c:c+3] == 4).all()):
                return (r + 1, c + 1)
    return None


def find_rotator(grid: np.ndarray) -> tuple[int, int] | None:
    """Find rotator by its specific pattern: yellow 7 at (r+1,c+1) and (r+1,c+2)."""
    rows, cols = grid.shape
    for r in range(5, min(rows - 2, 55)):
        for c in range(cols - 2):
            if (grid[r+1, c+1] == 7 and grid[r+1, c+2] == 7 and
                    grid[r+2, c+1] == 7 and grid[r, c+1] == 9):
                return (r + 1, c + 2)  # center of rotator
            # Also try simpler check: any yellow cell with purple neighbor
            if grid[r, c] == 7:
                neighbors = []
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbors.append(grid[nr, nc])
                if 9 in neighbors:
                    return (r, c)
    return None


def find_exit_door(grid: np.ndarray) -> tuple[int, int] | None:
    """Find exit door: 4×4 area with ≥6 border cells of color 11."""
    rows, cols = grid.shape
    best = None
    for r in range(5, min(rows - 3, 55)):
        for c in range(cols - 3):
            patch = grid[r:r+4, c:c+4]
            border = np.concatenate([
                patch[0, :], patch[3, :], patch[1:3, 0], patch[1:3, 3]
            ])
            if (border == 11).sum() >= 8:
                return (r + 2, c + 2)
    return best


def find_refiller(grid: np.ndarray) -> tuple[int, int] | None:
    """Find any 2×2 block of color 6 in the playfield (not energy row)."""
    rows, cols = grid.shape
    for r in range(5, min(rows - 1, 55)):
        for c in range(cols - 1):
            if (grid[r, c] == 6 and grid[r, c+1] == 6 and
                    grid[r+1, c] == 6 and grid[r+1, c+1] == 6):
                return (r + 1, c + 1)
    return None


def energy_remaining(grid: np.ndarray) -> int:
    """Count remaining energy cells (color 6) in energy row (row 2)."""
    try:
        energy_row = grid[2, :54]
        return int((energy_row == 6).sum())
    except Exception:
        return 25


def navigate_toward(player: tuple, target: tuple) -> str:
    """Pick action moving player toward target."""
    pr, pc = player
    tr, tc = target
    dr = tr - pr
    dc = tc - pc
    if abs(dr) >= abs(dc):
        return "ACTION2" if dr > 0 else "ACTION1"
    return "ACTION4" if dc > 0 else "ACTION3"


def smart_heuristic(
    frame: FrameData,
    counter: int,
    key_matches: bool,
    last_llm_action: str,
    stuck_counter: int,
) -> str:
    """
    Between LLM calls:
    1. If stuck (same position for >3 turns): try perpendicular to navigate around wall
    2. If energy critically low: go to refiller
    3. Otherwise: persist last LLM direction
    """
    # Wall avoidance: when stuck, try alternate directions
    if stuck_counter >= 4:
        # Perpendicular directions to try going around the wall
        alternates = {
            "ACTION1": ["ACTION4", "ACTION3", "ACTION2"],  # up stuck → try right/left/down
            "ACTION2": ["ACTION4", "ACTION3", "ACTION1"],  # down stuck → try right/left/up
            "ACTION3": ["ACTION1", "ACTION2", "ACTION4"],  # left stuck → try up/down/right
            "ACTION4": ["ACTION1", "ACTION2", "ACTION3"],  # right stuck → try up/down/left
        }
        choices = alternates.get(last_llm_action, ["ACTION4", "ACTION2", "ACTION3", "ACTION1"])
        # Cycle through alternates based on stuck counter
        alt_idx = (stuck_counter // 4) % len(choices)
        return choices[alt_idx]

    try:
        grid = np.array(frame.frame[0], dtype=np.uint8)
        player = find_player(grid)

        # Critical energy check
        if player is not None:
            energy = energy_remaining(grid)
            if energy < 5:
                refiller = find_refiller(grid)
                if refiller:
                    return navigate_toward(player, refiller)

    except Exception:
        pass

    # Default: persist the last LLM action
    if last_llm_action and last_llm_action.startswith("ACTION"):
        return last_llm_action

    expl = ["ACTION1", "ACTION4", "ACTION2", "ACTION3"]
    return expl[counter % len(expl)]


# ── LLM call helpers ─────────────────────────────────────────────────────────

def parse_action(text: str) -> tuple[str, str]:
    try:
        match = re.search(r"\{[^}]+\}", text, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            action = str(parsed.get("action", "ACTION1")).upper().strip()
            reason = str(parsed.get("reasoning", ""))[:400]
            return action, reason
    except Exception:
        pass
    m = re.search(r"ACTION[1-4]|RESET", text, re.IGNORECASE)
    return (m.group().upper() if m else "ACTION1"), text[:100]


# ── Agent ─────────────────────────────────────────────────────────────────────

class GeminiAgent(Agent):
    """Multi-LLM ARC-AGI-3 agent with game-mechanic heuristic."""

    MAX_ACTIONS = MAX_ACTIONS

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.hypothesis = "Starting."
        self.key_matches = False
        self._last_llm_action = "ACTION1"
        self._last_player_pos: tuple | None = None
        self._stuck_counter = 0
        self._key_index = 0
        self._last_llm_call: float = 0.0
        self._pplx_history: list[dict] = []

    def _call_pplx(self, img_b64: str, turn_text: str) -> str:
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in self._pplx_history[-(MSG_HISTORY_LIMIT * 2):]:
            messages.append(msg)
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": turn_text},
            ],
        })
        resp = requests.post(
            PPLX_URL,
            headers={"Authorization": f"Bearer {PPLX_KEY}"},
            json={"model": PPLX_MODEL, "messages": messages, "max_tokens": 350},
            timeout=45,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        self._pplx_history.append({"role": "user", "content": turn_text})
        self._pplx_history.append({"role": "assistant", "content": text[:300]})
        if len(self._pplx_history) > MSG_HISTORY_LIMIT * 2:
            self._pplx_history = self._pplx_history[-(MSG_HISTORY_LIMIT * 2):]
        return text

    def _call_gemini(self, img_b64: str, turn_text: str) -> str:
        for _ in range(len(GEMINI_KEYS)):
            key = GEMINI_KEYS[self._key_index % len(GEMINI_KEYS)]
            self._key_index += 1
            payload = {
                "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [
                    {"inline_data": {"mime_type": "image/png", "data": img_b64}},
                    {"text": turn_text},
                ]}],
                "generationConfig": {"temperature": 0.5, "maxOutputTokens": 300},
            }
            try:
                resp = requests.post(
                    GEMINI_API_BASE,
                    params={"key": key},
                    json=payload,
                    timeout=30,
                )
                if resp.status_code in (429, 503):
                    continue
                resp.raise_for_status()
                return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            except Exception:
                continue
        raise RuntimeError("All Gemini keys failed")

    def choose_action(
        self,
        frames: list[FrameData],
        current_frame: FrameData | None,
    ) -> GameAction:
        frame = current_frame or frames[-1]
        available = [f"ACTION{a}" for a in (frame.available_actions or [1, 2, 3, 4])]

        now = time.monotonic()
        should_call_llm = (now - self._last_llm_call) >= LLM_CALL_INTERVAL

        if should_call_llm:
            try:
                img_b64 = render_grid(frame.frame)
            except Exception as e:
                logger.warning(f"Render failed: {e}")
                img_b64 = None

            if img_b64:
                # Include key-match status in prompt
                grid = np.array(frame.frame[0], dtype=np.uint8)
                energy = energy_remaining(grid)
                player = find_player(grid)
                rotator = find_rotator(grid)
                exit_door = find_exit_door(grid)

                stuck_note = ""
                if self._stuck_counter >= 6:
                    stuck_note = (
                        f"\n⚠️ STUCK ALERT: Player has NOT moved for {self._stuck_counter} turns! "
                        f"Current direction ({self._last_llm_action}) is blocked by a wall. "
                        f"You MUST choose a DIFFERENT direction to navigate AROUND the wall. "
                        f"Do NOT choose {self._last_llm_action} again."
                    )

                turn_text = (
                    f"Turn {self.action_counter} | "
                    f"Levels: {frame.levels_completed}/{frame.win_levels} | "
                    f"Energy: {energy}/25 | "
                    f"Available: {available}\n"
                    f"Grid analysis — Player: {player} | Rotator: {rotator} | Exit door: {exit_door}\n"
                    f"Key matches exit: {self.key_matches} | "
                    f"Stuck turns: {self._stuck_counter}"
                    f"{stuck_note}\n"
                    f"Last reasoning: {self.hypothesis[:100]}\n"
                    f"Study the image. Is the key (bottom-left) matching the exit door center? "
                    f"If key matches, navigate to exit door. If stuck, navigate AROUND the wall. "
                    f"Choose the optimal action."
                )

                text = None
                source = None

                try:
                    text = self._call_pplx(img_b64, turn_text)
                    source = "PPLX"
                except Exception as e:
                    logger.debug(f"PPLX failed: {e}")

                if text is None:
                    try:
                        text = self._call_gemini(img_b64, turn_text)
                        source = "Gemini"
                    except Exception as e:
                        logger.debug(f"Gemini failed: {e}")

                if text:
                    self._last_llm_call = time.monotonic()
                    action_name, reasoning = parse_action(text)
                    if reasoning:
                        self.hypothesis = reasoning
                        # Try to detect key match from reasoning
                        lower = reasoning.lower()
                        if "key match" in lower or "matches the exit" in lower or "key matches" in lower:
                            self.key_matches = True
                        elif "does not match" in lower or "no match" in lower or "doesn't match" in lower:
                            self.key_matches = False

                    if action_name not in available and action_name != "RESET":
                        action_name = available[0] if available else "ACTION1"

                    self._last_llm_action = action_name
                    logger.info(f"[{source}] T{self.action_counter} → {action_name}: {self.hypothesis[:60]}")
                    return GameAction.__members__.get(action_name, GameAction.ACTION1)

        # Track player position to detect being stuck
        try:
            grid = np.array(frame.frame[0], dtype=np.uint8)
            current_pos = find_player(grid)
            if current_pos is not None:
                if current_pos == self._last_player_pos:
                    self._stuck_counter += 1
                else:
                    self._stuck_counter = 0
                self._last_player_pos = current_pos
        except Exception:
            pass

        # Smart heuristic fallback
        action_name = smart_heuristic(
            frame, self.action_counter, self.key_matches,
            self._last_llm_action, self._stuck_counter
        )
        if action_name not in available and action_name != "RESET":
            action_name = available[0] if available else "ACTION1"
        return GameAction.__members__.get(action_name, GameAction.ACTION1)

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

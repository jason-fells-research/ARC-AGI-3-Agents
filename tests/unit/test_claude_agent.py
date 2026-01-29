import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from typing import Any

from arcengine import FrameData, GameAction, GameState
from agents.templates.claude_agents import ClaudeCodeAgent


@pytest.mark.unit
class TestClaudeCodeAgent:
    
    @pytest.fixture
    def mock_agent(self) -> ClaudeCodeAgent:
        with patch('agents.templates.claude_agents.create_arc_tools_server'):
            agent = ClaudeCodeAgent(
                card_id="test_card",
                game_id="test_game",
                agent_name="test_agent",
                ROOT_URL="http://localhost:8001",
                record=False,
                arc_env=Mock(),
            )
            agent.frames = [
                FrameData(
                    score=0,
                    state=GameState.NOT_PLAYED,
                    frame=[[[0 for _ in range(64)] for __ in range(64)]],
                    available_actions=[0, 1, 2, 3, 4, 5]
                )
            ]
            return agent
    
    @pytest.mark.parametrize(
        "tool_name,expected_action",
        [
            ("mcp__arc-game-tools__reset_game", GameAction.RESET),
            ("mcp__arc-game-tools__action1_move_up", GameAction.ACTION1),
            ("mcp__arc-game-tools__action2_move_down", GameAction.ACTION2),
            ("mcp__arc-game-tools__action3_move_left", GameAction.ACTION3),
            ("mcp__arc-game-tools__action4_move_right", GameAction.ACTION4),
            ("mcp__arc-game-tools__action5_interact", GameAction.ACTION5),
            ("mcp__arc-game-tools__action6_click", GameAction.ACTION6),
            ("mcp__arc-game-tools__action7_undo", GameAction.ACTION7),
        ],
    )
    def test_parse_action_from_tool(self, mock_agent, tool_name, expected_action):
        tool_input: dict[str, Any] = {}
        action = mock_agent.parse_action_from_tool(tool_name, tool_input)
        
        assert action is not None
        assert action == expected_action
        assert action.action_data.game_id == "test_game"
    
    def test_parse_action_from_tool_with_coordinates(self, mock_agent):
        tool_name = "mcp__arc-game-tools__action6_click"
        tool_input = {"x": 32, "y": 45}
        
        action = mock_agent.parse_action_from_tool(tool_name, tool_input)
        
        assert action is not None
        assert action == GameAction.ACTION6
        assert action.action_data.x == 32
        assert action.action_data.y == 45
        assert action.action_data.game_id == "test_game"
    
    def test_parse_action_with_reasoning(self, mock_agent):
        mock_agent.latest_reasoning = "This is my reasoning for taking this action."
        tool_name = "mcp__arc-game-tools__action1_move_up"
        tool_input: dict[str, Any] = {}
        
        action = mock_agent.parse_action_from_tool(tool_name, tool_input)
        
        assert action is not None
        assert "reasoning" in action.action_data.__dict__
        assert action.action_data.__dict__["reasoning"]["text"] == mock_agent.latest_reasoning
    
    def test_parse_action_with_long_reasoning(self, mock_agent):
        mock_agent.latest_reasoning = "x" * 20000
        tool_name = "mcp__arc-game-tools__action1_move_up"
        tool_input: dict[str, Any] = {}
        
        action = mock_agent.parse_action_from_tool(tool_name, tool_input)
        
        assert action is not None
        assert "reasoning" in action.action_data.__dict__
        assert len(action.action_data.__dict__["reasoning"]["text"]) == 16000
    
    def test_parse_action_unknown_tool(self, mock_agent):
        tool_name = "unknown_tool"
        tool_input: dict[str, Any] = {}
        
        action = mock_agent.parse_action_from_tool(tool_name, tool_input)
        
        assert action is None
    
    def test_track_tokens(self, mock_agent):
        mock_usage = Mock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 50
        
        initial_count = mock_agent.token_counter
        mock_agent.track_tokens(mock_usage)
        
        assert mock_agent.token_counter == initial_count + 150
    
    def test_is_done_when_win(self, mock_agent):
        mock_agent.frames = [
            FrameData(
                score=100,
                state=GameState.WIN,
                frame=[[[0 for _ in range(64)] for __ in range(64)]]
            )
        ]
        
        result = mock_agent.is_done(mock_agent.frames, mock_agent.frames[-1])
        
        assert result is True
    
    def test_is_not_done_when_not_finished(self, mock_agent):
        mock_agent.frames = [
            FrameData(
                score=0,
                state=GameState.NOT_FINISHED,
                frame=[[[0 for _ in range(64)] for __ in range(64)]]
            )
        ]
        
        result = mock_agent.is_done(mock_agent.frames, mock_agent.frames[-1])
        
        assert result is False
    
    def test_build_game_prompt(self, mock_agent):
        mock_agent.frames = [
            FrameData(
                score=42,
                state=GameState.NOT_FINISHED,
                levels_completed=2,
                frame=[[[i % 16 for i in range(64)] for __ in range(64)]],
                available_actions=[0, 1, 2, 6]
            )
        ]
        
        prompt = mock_agent.build_game_prompt(mock_agent.frames[-1])
        
        assert "test_game" in prompt
        assert "NOT_FINISHED" in prompt
        assert "Levels Completed: 2" in prompt
        assert "RESET, ACTION1, ACTION2, ACTION6" in prompt
    
    def test_agent_name_format(self, mock_agent):
        name = mock_agent.name
        
        assert "test_game" in name
        assert "claude-sonnet-4-5-20250929" in name
    
    def test_initialization_values(self, mock_agent):
        assert mock_agent.token_counter == 0
        assert mock_agent.step_counter == 0
        assert mock_agent.latest_reasoning == ""
        assert mock_agent.game_id == "test_game"
        assert mock_agent.MODEL == "claude-sonnet-4-5-20250929"
        assert mock_agent.MAX_ACTIONS == 80


@pytest.mark.unit
class TestClaudeToolsIntegration:
    
    @pytest.fixture
    def mock_agent(self) -> ClaudeCodeAgent:
        with patch('agents.templates.claude_agents.create_arc_tools_server'):
            agent = ClaudeCodeAgent(
                card_id="test_card",
                game_id="test_game",
                agent_name="test_agent",
                ROOT_URL="http://localhost:8001",
                record=False,
                arc_env=Mock(),
            )
            agent.frames = [
                FrameData(
                    score=0,
                    state=GameState.NOT_PLAYED,
                    frame=[[[0 for _ in range(64)] for __ in range(64)]],
                    available_actions=[0, 1, 2, 3, 4, 5, 6, 7]
                )
            ]
            return agent
    
    @pytest.mark.asyncio
    async def test_tools_server_creation(self, mock_agent):
        from agents.claude_tools import create_arc_tools_server
        
        server = create_arc_tools_server(mock_agent)
        
        assert server is not None
        assert isinstance(server, dict)
        assert server.get('name') == 'arc-game-tools'
    
    @pytest.mark.asyncio
    async def test_action6_coordinate_validation(self):
        from agents.claude_tools import create_arc_tools_server
        
        with patch('agents.templates.claude_agents.create_arc_tools_server'):
            agent = ClaudeCodeAgent(
                card_id="test_card",
                game_id="test_game",
                agent_name="test_agent",
                ROOT_URL="http://localhost:8001",
                record=False,
                arc_env=Mock(),
            )
            agent.frames = [FrameData(score=0, frame=[[[0 for _ in range(64)] for __ in range(64)]])]
        
        server = create_arc_tools_server(agent)
        
        assert server is not None

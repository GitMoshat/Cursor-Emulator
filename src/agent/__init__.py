# AI Agent System for GBC Emulator
from .interface import AgentInterface, GameState, AgentConfig
from .manager import AgentManager
from .ollama_agent import OllamaAgent
from .stub_agent import StubAgent, RandomAgent, ScriptedAgent
from .guided_agent import GuidedOllamaAgent
from .memory_agent import MemoryAgent
from .memory_manager import MemoryManager
from .game_stages import StageManager, GameStage, POKEMON_STAGES

__all__ = [
    'AgentInterface',
    'GameState',
    'AgentConfig',
    'AgentManager',
    'OllamaAgent',
    'GuidedOllamaAgent',
    'MemoryAgent',
    'MemoryManager',
    'StubAgent',
    'RandomAgent',
    'ScriptedAgent',
    'StageManager',
    'GameStage',
    'POKEMON_STAGES',
]


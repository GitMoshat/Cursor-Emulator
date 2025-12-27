"""
Agent Manager for GBC Emulator
Handles agent lifecycle, state extraction, and action execution.
"""

import threading
import queue
import time
from typing import Optional, Dict, Any, List, Type
import numpy as np

from .interface import (
    AgentInterface, AgentAction, GameState, Button, 
    SpriteInfo, AgentConfig
)


class AgentManager:
    """
    Manages AI agents and their interaction with the emulator.
    
    Features:
    - Agent lifecycle management (start/stop/switch)
    - Game state extraction from emulator
    - Action execution (button presses)
    - Thread-safe operation
    - Action history and statistics
    """
    
    # Button name to emulator button index mapping
    BUTTON_MAP = {
        Button.A: 'a',
        Button.B: 'b',
        Button.START: 'start',
        Button.SELECT: 'select',
        Button.UP: 'up',
        Button.DOWN: 'down',
        Button.LEFT: 'left',
        Button.RIGHT: 'right',
    }
    
    def __init__(self, emulator):
        self.emulator = emulator
        self.config = AgentConfig()
        
        # Current agent
        self.agent: Optional[AgentInterface] = None
        self.agent_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Action queue and state
        self.action_queue: queue.Queue = queue.Queue()
        self.current_action: Optional[AgentAction] = None
        self.action_frames_remaining = 0
        self.buttons_held: set = set()
        
        # Turbo mode support
        self.turbo_mode = False
        self.turbo_multiplier = 4  # How much faster turbo runs
        
        # Frame history for context
        self.frame_history: List[np.ndarray] = []
        self.max_frame_history = 4
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'actions_taken': 0,
            'errors': 0,
            'avg_decision_time_ms': 0,
        }
        self._decision_times: List[float] = []
        
        # Registered agent types
        self._agent_types: Dict[str, Type[AgentInterface]] = {}
        self._register_default_agents()
    
    def _register_default_agents(self):
        """Register built-in agent types."""
        from .ollama_agent import OllamaAgent, OllamaAgentSimple
        from .stub_agent import StubAgent, RandomAgent, ScriptedAgent
        from .guided_agent import GuidedOllamaAgent
        from .memory_agent import MemoryAgent
        from .smart_agent import SmartAgent
        from .action_toolkit import ToolkitAgent
        
        self._agent_types['toolkit'] = ToolkitAgent  # Default - uses action toolkit
        self._agent_types['smart'] = SmartAgent  # Smart heuristics + LLM
        self._agent_types['memory'] = MemoryAgent  # Rule-based
        self._agent_types['guided'] = GuidedOllamaAgent  # Vision-based
        self._agent_types['ollama'] = OllamaAgent
        self._agent_types['ollama_simple'] = OllamaAgentSimple
        self._agent_types['stub'] = StubAgent
        self._agent_types['random'] = RandomAgent
        self._agent_types['scripted'] = ScriptedAgent
    
    def register_agent_type(self, name: str, agent_class: Type[AgentInterface]):
        """Register a new agent type."""
        self._agent_types[name] = agent_class
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agent types."""
        return list(self._agent_types.keys())
    
    def create_agent(self, agent_type: str, **kwargs) -> Optional[AgentInterface]:
        """Create an agent instance by type name."""
        if agent_type not in self._agent_types:
            print(f"Unknown agent type: {agent_type}")
            return None
        
        agent_class = self._agent_types[agent_type]
        return agent_class(**kwargs)
    
    def set_agent(self, agent: AgentInterface) -> bool:
        """Set the active agent."""
        # Stop current agent if running
        if self.running:
            self.stop()
        
        self.agent = agent
        return True
    
    def start(self, **init_kwargs) -> bool:
        """Start the agent."""
        if not self.agent:
            print("No agent set")
            return False
        
        if self.running:
            print("Agent already running")
            return False
        
        # Always pass emulator reference to agent
        init_kwargs['emulator'] = self.emulator
        
        # Initialize agent
        if not self.agent.initialize(**init_kwargs):
            print(f"Failed to initialize agent: {self.agent.name}")
            return False
        
        self.agent.enabled = True
        self.running = True
        
        print(f"Agent '{self.agent.name}' started")
        return True
    
    def stop(self):
        """Stop the agent."""
        self.running = False
        
        if self.agent:
            self.agent.enabled = False
            self.agent.shutdown()
        
        # Release all held buttons
        self._release_all_buttons()
        
        print("Agent stopped")
    
    def process_frame(self, frame: np.ndarray, turbo: bool = False) -> Optional[AgentAction]:
        """
        Process a frame and get agent action.
        Called by the emulator/GUI each frame.
        
        Args:
            frame: Current frame buffer
            turbo: Whether turbo mode is active (adjusts timing)
        """
        if not self.running or not self.agent:
            return None
        
        self.turbo_mode = turbo
        self.stats['frames_processed'] += 1
        
        # In turbo mode, action holds are shorter (game runs faster)
        hold_decrement = self.turbo_multiplier if turbo else 1
        
        # Handle current action hold - fast path, no processing needed
        if self.action_frames_remaining > 0:
            self.action_frames_remaining -= hold_decrement
            if self.action_frames_remaining <= 0:
                self._release_action_buttons()
                self.action_frames_remaining = 0
            return None
        
        # Skip frame history in turbo mode to save memory bandwidth
        if not turbo and len(self.frame_history) < self.max_frame_history:
            self.frame_history.append(frame)  # Don't copy, just reference
            if len(self.frame_history) > self.max_frame_history:
                self.frame_history.pop(0)
        
        # Create minimal state - ToolkitAgent uses its own MemoryManager
        # Only extract what's actually needed by the agent interface
        state = self._extract_minimal_state(frame)
        
        # Get agent decision
        start_time = time.time()
        action = self.agent.process_frame(state)
        decision_time = (time.time() - start_time) * 1000
        
        # Only track timing occasionally to reduce overhead
        if self.stats['frames_processed'] % 10 == 0:
            self._decision_times.append(decision_time)
            if len(self._decision_times) > 50:
                self._decision_times.pop(0)
            self.stats['avg_decision_time_ms'] = sum(self._decision_times) / len(self._decision_times)
        
        if action and action.buttons_to_press:
            self.stats['actions_taken'] += 1
            self._execute_action(action)
            self.current_action = action
            effective_hold = action.hold_frames * (self.turbo_multiplier if turbo else 1)
            self.action_frames_remaining = effective_hold - hold_decrement
        
        return action
    
    def _extract_minimal_state(self, frame: np.ndarray) -> GameState:
        """Extract minimal state - agents with MemoryManager don't need full extraction."""
        memory = self.emulator.memory
        # Just return frame reference and minimal info - no heavy copies
        return GameState(
            frame=frame,
            frame_number=self.stats['frames_processed'],
            wram=np.zeros(1, dtype=np.uint8),  # Placeholder - agent uses MemoryManager
            hram=np.zeros(1, dtype=np.uint8),
            oam=np.zeros(1, dtype=np.uint8),
            sprites=[],
            buttons_pressed=[],
        )
    
    def _extract_game_state(self, frame: np.ndarray) -> GameState:
        """Extract complete game state from emulator."""
        memory = self.emulator.memory
        ppu = self.emulator.ppu
        
        # Get WRAM (flatten the banked array)
        wram = memory.wram.flatten().copy()
        
        # Get HRAM
        hram = memory.hram.copy()
        
        # Get OAM
        oam = memory.oam.copy()
        
        # Parse sprites
        sprites = []
        for i in range(40):
            y = int(oam[i * 4]) - 16
            x = int(oam[i * 4 + 1]) - 8
            tile = int(oam[i * 4 + 2])
            flags = int(oam[i * 4 + 3])
            
            sprites.append(SpriteInfo(
                index=i,
                x=x,
                y=y,
                tile=tile,
                flags=flags,
                palette=flags & 0x07 if memory.cgb_mode else (flags >> 4) & 0x01,
                flip_x=(flags & 0x20) != 0,
                flip_y=(flags & 0x40) != 0,
                priority=(flags & 0x80) != 0,
            ))
        
        # Get current buttons
        buttons_pressed = []
        joypad = memory.joypad_state
        if not (joypad & 0x01): buttons_pressed.append(Button.A)
        if not (joypad & 0x02): buttons_pressed.append(Button.B)
        if not (joypad & 0x04): buttons_pressed.append(Button.SELECT)
        if not (joypad & 0x08): buttons_pressed.append(Button.START)
        if not (joypad & 0x10): buttons_pressed.append(Button.RIGHT)
        if not (joypad & 0x20): buttons_pressed.append(Button.LEFT)
        if not (joypad & 0x40): buttons_pressed.append(Button.UP)
        if not (joypad & 0x80): buttons_pressed.append(Button.DOWN)
        
        return GameState(
            frame=frame.copy(),
            frame_number=self.emulator.total_frames,
            wram=wram,
            hram=hram,
            oam=oam,
            sprites=sprites,
            scroll_x=ppu.scx,
            scroll_y=ppu.scy,
            window_x=ppu.wx,
            window_y=ppu.wy,
            buttons_pressed=buttons_pressed,
            previous_frames=self.frame_history.copy(),
        )
    
    def _execute_action(self, action: AgentAction):
        """Execute an agent action (press buttons)."""
        # Release buttons that should be released
        for btn in action.buttons_to_release:
            self._release_button(btn)
        
        # Press buttons
        for btn in action.buttons_to_press:
            self._press_button(btn)
    
    def _press_button(self, button: Button):
        """Press a button on the emulator."""
        if button in self.BUTTON_MAP:
            self.emulator.press_button(self.BUTTON_MAP[button])
            self.buttons_held.add(button)
    
    def _release_button(self, button: Button):
        """Release a button on the emulator."""
        if button in self.BUTTON_MAP:
            self.emulator.release_button(self.BUTTON_MAP[button])
            self.buttons_held.discard(button)
    
    def _release_action_buttons(self):
        """Release buttons from current action."""
        if self.current_action:
            for btn in self.current_action.buttons_to_press:
                self._release_button(btn)
    
    def _release_all_buttons(self):
        """Release all held buttons."""
        for btn in list(self.buttons_held):
            self._release_button(btn)
    
    def get_status(self) -> Dict[str, Any]:
        """Get manager and agent status."""
        status = {
            'running': self.running,
            'agent_name': self.agent.name if self.agent else None,
            'buttons_held': [b.name for b in self.buttons_held],
            'stats': self.stats.copy(),
        }
        
        if self.agent:
            status['agent_status'] = self.agent.get_status()
        
        if self.current_action:
            status['current_action'] = {
                'buttons': [b.name for b in self.current_action.buttons_to_press],
                'reasoning': self.current_action.reasoning,
                'frames_remaining': self.action_frames_remaining,
            }
        
        return status
    
    def reset(self):
        """Reset agent state (e.g., on game reset)."""
        self._release_all_buttons()
        self.frame_history.clear()
        self.current_action = None
        self.action_frames_remaining = 0
        
        if self.agent:
            self.agent.reset()
        
        self.stats = {
            'frames_processed': 0,
            'actions_taken': 0,
            'errors': 0,
            'avg_decision_time_ms': 0,
        }
        self._decision_times.clear()


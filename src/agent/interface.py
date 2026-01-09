"""
AI Agent Interface for GBC Emulator
Defines the data structures and base class for all AI agents.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from abc import ABC, abstractmethod
from enum import Enum, auto
import base64
from io import BytesIO


class Button(Enum):
    """GBC Controller buttons."""
    A = auto()
    B = auto()
    START = auto()
    SELECT = auto()
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()


@dataclass
class SpriteInfo:
    """Information about a sprite from OAM."""
    index: int
    x: int
    y: int
    tile: int
    flags: int
    palette: int
    flip_x: bool
    flip_y: bool
    priority: bool
    
    @property
    def visible(self) -> bool:
        """Check if sprite is on screen."""
        return -8 < self.x < 168 and -16 < self.y < 160


@dataclass
class GameState:
    """
    Complete snapshot of the game state for AI decision making.
    
    This provides everything an AI might need to understand what's
    happening in the game and make informed decisions.
    """
    
    # Visual data
    frame: np.ndarray  # Current frame (144x160x3 RGB)
    frame_number: int  # Frame counter
    
    # Memory snapshots
    wram: np.ndarray  # Work RAM (game variables, player state, etc.)
    hram: np.ndarray  # High RAM
    oam: np.ndarray   # Sprite data
    
    # Parsed sprite information
    sprites: List[SpriteInfo] = field(default_factory=list)
    
    # Screen state
    scroll_x: int = 0
    scroll_y: int = 0
    window_x: int = 0
    window_y: int = 0
    
    # Current input state
    buttons_pressed: List[Button] = field(default_factory=list)
    
    # Timing
    emulator_speed: float = 1.0  # Current speed multiplier
    
    # Optional: Previous frames for motion analysis
    previous_frames: List[np.ndarray] = field(default_factory=list)
    
    # Optional: Game-specific extracted data
    custom_data: Dict[str, Any] = field(default_factory=dict)
    
    def get_frame_base64(self, format: str = 'PNG') -> str:
        """Get frame as base64 encoded image for LLM vision APIs."""
        from PIL import Image
        img = Image.fromarray(self.frame)
        buffer = BytesIO()
        img.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def get_frame_description(self) -> str:
        """Get a text description of visible sprites and screen state."""
        lines = [
            f"Frame {self.frame_number}",
            f"Scroll: ({self.scroll_x}, {self.scroll_y})",
            f"Window: ({self.window_x}, {self.window_y})",
            f"Visible sprites: {sum(1 for s in self.sprites if s.visible)}",
        ]
        
        for sprite in self.sprites:
            if sprite.visible:
                lines.append(f"  Sprite {sprite.index}: pos=({sprite.x}, {sprite.y}) tile={sprite.tile}")
        
        if self.buttons_pressed:
            lines.append(f"Buttons: {', '.join(b.name for b in self.buttons_pressed)}")
        
        return '\n'.join(lines)
    
    def read_memory(self, address: int, length: int = 1) -> bytes:
        """Read from WRAM at a specific address (0xC000-0xDFFF maps to wram)."""
        if 0xC000 <= address < 0xE000:
            offset = address - 0xC000
            if length == 1:
                return bytes([self.wram.flat[offset] if offset < len(self.wram.flat) else 0])
            return bytes(self.wram.flat[offset:offset+length])
        return bytes(length)
    
    def read_byte(self, address: int) -> int:
        """Read a single byte from WRAM."""
        data = self.read_memory(address, 1)
        return data[0] if data else 0
    
    def read_word(self, address: int) -> int:
        """Read a 16-bit word from WRAM (little-endian)."""
        data = self.read_memory(address, 2)
        if len(data) >= 2:
            return data[0] | (data[1] << 8)
        return 0


@dataclass
class AgentAction:
    """An action the agent wants to take."""
    buttons_to_press: List[Button] = field(default_factory=list)
    buttons_to_release: List[Button] = field(default_factory=list)
    
    # Optional: Hold duration in frames
    hold_frames: int = 1
    
    # Optional: Agent's reasoning for this action
    reasoning: str = ""
    
    # Optional: Confidence score (0-1)
    confidence: float = 1.0


class AgentInterface(ABC):
    """
    Base class for all AI agents.
    
    Subclass this to create agents that can play GBC games.
    """
    
    def __init__(self, name: str = "BaseAgent"):
        self.name = name
        self.enabled = False
        self.action_history: List[AgentAction] = []
        self.frame_skip = 0  # How many frames to skip between decisions
        self._frame_counter = 0
        
        # Callbacks
        self.on_action: Optional[Callable[[AgentAction], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
    
    @abstractmethod
    def initialize(self, **kwargs) -> bool:
        """
        Initialize the agent (connect to LLM, load model, etc.)
        Returns True if successful.
        """
        pass
    
    @abstractmethod
    def decide(self, state: GameState) -> AgentAction:
        """
        Given the current game state, decide what action to take.
        This is the main AI logic.
        """
        pass
    
    def process_frame(self, state: GameState) -> Optional[AgentAction]:
        """
        Process a frame and potentially return an action.
        Handles frame skipping automatically.
        """
        if not self.enabled:
            return None
        
        self._frame_counter += 1
        if self._frame_counter <= self.frame_skip:
            return None
        
        self._frame_counter = 0
        
        try:
            action = self.decide(state)
            self.action_history.append(action)
            
            # Keep history bounded
            if len(self.action_history) > 1000:
                self.action_history = self.action_history[-500:]
            
            if self.on_action:
                self.on_action(action)
            
            return action
            
        except Exception as e:
            if self.on_error:
                self.on_error(e)
            return None
    
    def reset(self):
        """Reset agent state (e.g., when game restarts)."""
        self.action_history.clear()
        self._frame_counter = 0
    
    def shutdown(self):
        """Clean up resources."""
        self.enabled = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status for debugging."""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'frame_skip': self.frame_skip,
            'actions_taken': len(self.action_history),
        }


class AgentConfig:
    """Configuration for agent behavior."""
    
    def __init__(self):
        # Decision frequency
        self.frame_skip = 4  # Decide every N frames (4 = ~15 decisions/sec)
        
        # History settings
        self.keep_previous_frames = 4  # Number of previous frames to provide
        
        # Ollama settings
        self.ollama_host = "http://localhost:11434"
        self.ollama_model = "llava"  # Vision model
        self.ollama_timeout = 30
        
        # Action settings
        self.max_hold_frames = 30  # Max frames to hold a button
        self.allow_simultaneous_dpad = False  # Allow UP+DOWN, LEFT+RIGHT
        
        # Debug
        self.log_actions = True
        self.log_reasoning = True




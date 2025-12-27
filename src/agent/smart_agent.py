"""
Smart AI Agent for GBC Emulator
Uses memory state + optional LLM to figure out what to do dynamically.
No hardcoded stages - learns from the game state itself.
"""

import json
import requests
import time
import random
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from .interface import AgentInterface, AgentAction, GameState as AgentGameState, Button, AgentConfig
from .memory_manager import MemoryManager, GameState


@dataclass
class GameContext:
    """Dynamic context about the current game situation."""
    situation: str  # "title", "dialog", "overworld", "battle", "menu", "unknown"
    can_move: bool
    needs_input: bool  # Waiting for player input (dialog, menu)
    suggested_actions: List[str]
    description: str


class SmartAgent(AgentInterface):
    """
    Smart AI Agent that figures out what to do from game state.
    
    No hardcoded stages - analyzes memory and screen to understand:
    - What screen/mode we're on
    - What actions are available
    - What the goal should be
    
    Uses LLM when available for complex decisions, falls back to
    intelligent heuristics when offline.
    """
    
    # LLM system prompt - teaches the AI how to play
    SYSTEM_PROMPT = """You are an AI playing a Pokemon-style Game Boy game. You can see the current game state from memory.

Your goal: Progress through the game naturally. Start the game, talk to NPCs, get your first Pokemon, explore.

Based on the game state I provide, decide what button to press.
Available: A, B, UP, DOWN, LEFT, RIGHT, START, SELECT

Rules:
- If at title/intro screen: Press START or A to begin
- If dialog is showing: Press A to advance
- If in a menu: Navigate with D-pad, select with A, back with B
- If in overworld: Explore, talk to people (A), enter buildings
- If in battle: Fight (A to select moves)

Respond with ONLY a JSON object:
{"button": "BUTTON_NAME", "hold": 1-20, "reason": "why"}

Be concise. One button at a time."""

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(name="SmartAgent")
        self.config = config or AgentConfig()
        self.frame_skip = 8  # Check every 8 frames
        
        # Memory manager
        self.memory_manager: Optional[MemoryManager] = None
        
        # LLM settings
        self.ollama_host = self.config.ollama_host
        self.ollama_model = "llama3.2"  # Fast text model
        self.llm_connected = False
        self.use_llm = True  # Try LLM first, fallback to heuristics
        
        # State tracking
        self.last_situation = ""
        self.same_situation_count = 0
        self.action_history: List[str] = []
        self.position_history: List[tuple] = []
        
        # Thinking log
        self.thinking_history: List[str] = []
        self.current_thinking = ""
        
        # Current held action
        self.current_hold: Optional[AgentAction] = None
        self.hold_remaining = 0
    
    def initialize(self, **kwargs) -> bool:
        """Initialize with emulator reference."""
        emulator = kwargs.get('emulator')
        if emulator:
            self.memory_manager = MemoryManager(emulator)
            self.memory_manager.detect_game()
            self._log(f"Game: {self.memory_manager.game_type.name}")
        
        # Try LLM connection
        try:
            resp = requests.get(f"{self.ollama_host}/api/tags", timeout=3)
            self.llm_connected = resp.status_code == 200
            if self.llm_connected:
                self._log(f"LLM connected: {self.ollama_model}")
            else:
                self._log("LLM not available, using heuristics")
        except:
            self.llm_connected = False
            self._log("LLM offline, using heuristics")
        
        self._log("SmartAgent ready - will analyze game state dynamically")
        return True
    
    def _log(self, msg: str):
        """Log thinking."""
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        self.thinking_history.append(entry)
        self.current_thinking = msg
        if len(self.thinking_history) > 100:
            self.thinking_history.pop(0)
        print(f"[SmartAgent] {msg}")
    
    def _analyze_situation(self, state: GameState) -> GameContext:
        """Analyze current game state to understand situation."""
        
        # Default context
        ctx = GameContext(
            situation="unknown",
            can_move=False,
            needs_input=True,
            suggested_actions=[],
            description=""
        )
        
        # Check if game seems to have started
        game_active = state.game_started or state.party_count > 0 or state.player_position.x > 0
        
        # Battle check
        if state.battle.in_battle:
            ctx.situation = "battle"
            ctx.can_move = False
            ctx.needs_input = True
            enemy = state.battle.enemy_name or f"Enemy Lv{state.battle.enemy_level}"
            ctx.description = f"In battle vs {enemy}"
            ctx.suggested_actions = ["A"]  # Usually want to attack/advance
            return ctx
        
        # Menu check
        if state.menu.in_menu:
            ctx.situation = "menu"
            ctx.can_move = False
            ctx.needs_input = True
            ctx.description = f"In menu, cursor at {state.menu.cursor_position}"
            ctx.suggested_actions = ["A", "B", "UP", "DOWN"]
            return ctx
        
        # Dialog/text check
        if state.menu.text_active:
            ctx.situation = "dialog"
            ctx.can_move = False
            ctx.needs_input = True
            ctx.description = "Text/dialog is showing"
            ctx.suggested_actions = ["A", "B"]
            return ctx
        
        # Title/intro screen check (no party, no real position)
        if not game_active:
            ctx.situation = "title"
            ctx.can_move = False
            ctx.needs_input = True
            ctx.description = "Title or intro screen"
            ctx.suggested_actions = ["START", "A"]
            return ctx
        
        # Overworld
        ctx.situation = "overworld"
        ctx.can_move = True
        ctx.needs_input = True
        pos = state.player_position
        ctx.description = f"Overworld at ({pos.x},{pos.y}) {pos.map_name}, facing {pos.facing}"
        ctx.suggested_actions = ["UP", "DOWN", "LEFT", "RIGHT", "A", "START"]
        
        return ctx
    
    def _get_heuristic_action(self, state: GameState, ctx: GameContext) -> AgentAction:
        """Get action using smart heuristics (no LLM)."""
        
        buttons = []
        hold = 5
        reason = ""
        
        if ctx.situation == "title":
            # At title - press START or A
            if self.same_situation_count % 2 == 0:
                buttons = [Button.START]
                reason = "Title screen - pressing START"
            else:
                buttons = [Button.A]
                reason = "Title screen - pressing A"
            hold = 10
        
        elif ctx.situation == "dialog":
            # Dialog showing - advance it
            buttons = [Button.A]
            hold = 8
            reason = "Advancing dialog"
        
        elif ctx.situation == "menu":
            # In menu - try to navigate/select
            if self.same_situation_count > 5:
                buttons = [Button.B]
                reason = "Stuck in menu, pressing B to exit"
            else:
                buttons = [Button.A]
                reason = "Menu - selecting option"
            hold = 5
        
        elif ctx.situation == "battle":
            # In battle - fight!
            menu_state = state.battle.menu_state
            if menu_state == 0:
                buttons = [Button.A]
                reason = "Battle - selecting FIGHT"
            elif menu_state == 1:
                buttons = [Button.A]
                reason = "Battle - selecting move"
            else:
                buttons = [Button.A]
                reason = "Battle - confirming"
            hold = 5
        
        elif ctx.situation == "overworld":
            # Can move - explore!
            pos = state.player_position
            
            # Track position for stuck detection
            current_pos = (pos.x, pos.y)
            self.position_history.append(current_pos)
            if len(self.position_history) > 30:
                self.position_history.pop(0)
            
            # Check if stuck
            if len(self.position_history) >= 10:
                recent = self.position_history[-10:]
                if all(p == recent[0] for p in recent):
                    # Stuck! Try interacting or different direction
                    if self.same_situation_count % 3 == 0:
                        buttons = [Button.A]
                        reason = "Stuck - trying to interact"
                    else:
                        # Random direction
                        directions = [Button.UP, Button.DOWN, Button.LEFT, Button.RIGHT]
                        buttons = [random.choice(directions)]
                        reason = f"Stuck - trying {buttons[0].name}"
                    hold = 15
                    return AgentAction(buttons_to_press=buttons, hold_frames=hold, reasoning=reason)
            
            # Smart exploration based on facing
            facing = pos.facing
            
            # Tend to keep moving forward, occasionally interact
            if self.same_situation_count % 5 == 0:
                # Try to interact with whatever we're facing
                buttons = [Button.A]
                reason = f"Exploring - checking what's ahead"
                hold = 5
            else:
                # Move in current direction or explore
                if facing == "up":
                    buttons = [Button.UP]
                elif facing == "down":
                    buttons = [Button.DOWN]
                elif facing == "left":
                    buttons = [Button.LEFT]
                else:
                    buttons = [Button.RIGHT]
                reason = f"Exploring - moving {facing}"
                hold = 12
            
            # Occasionally change direction for exploration
            if random.random() < 0.2:
                directions = [Button.UP, Button.DOWN, Button.LEFT, Button.RIGHT]
                buttons = [random.choice(directions)]
                reason = f"Exploring - changing direction to {buttons[0].name}"
        
        else:
            # Unknown - try A
            buttons = [Button.A]
            reason = "Unknown state - pressing A"
            hold = 10
        
        return AgentAction(
            buttons_to_press=buttons,
            hold_frames=hold,
            reasoning=reason
        )
    
    def _get_llm_action(self, state: GameState, ctx: GameContext) -> Optional[AgentAction]:
        """Get action from LLM based on game state."""
        if not self.llm_connected:
            return None
        
        try:
            # Build state description
            state_desc = f"""
Game State:
- Situation: {ctx.situation} - {ctx.description}
- Player: {state.player_name or 'Unknown'} at ({state.player_position.x}, {state.player_position.y})
- Map: {state.player_position.map_name}
- Facing: {state.player_position.facing}
- Party: {state.party_count} Pokemon
- In Battle: {state.battle.in_battle}
- Has Starter: {state.has_starter}
- Money: ${state.money}
- Badges: {state.badges}

Recent actions: {', '.join(self.action_history[-5:])}

What button should I press?"""

            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": state_desc,
                    "system": self.SYSTEM_PROMPT,
                    "stream": False,
                    "options": {"num_predict": 80, "temperature": 0.7}
                },
                timeout=10
            )
            
            if response.status_code == 200:
                text = response.json().get('response', '')
                return self._parse_llm_response(text)
        
        except Exception as e:
            self._log(f"LLM error: {e}")
        
        return None
    
    def _parse_llm_response(self, text: str) -> Optional[AgentAction]:
        """Parse LLM JSON response."""
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                
                btn_name = data.get('button', 'A').upper()
                try:
                    button = Button[btn_name]
                except KeyError:
                    button = Button.A
                
                hold = min(max(data.get('hold', 5), 1), 20)
                reason = data.get('reason', 'LLM decision')
                
                return AgentAction(
                    buttons_to_press=[button],
                    hold_frames=hold,
                    reasoning=f"LLM: {reason}"
                )
        except:
            pass
        return None
    
    def decide(self, agent_state: AgentGameState) -> AgentAction:
        """Main decision function."""
        
        # Continue holding current action
        if self.hold_remaining > 0:
            self.hold_remaining -= 1
            return self.current_hold or AgentAction()
        
        # Get game state from memory
        if not self.memory_manager:
            return AgentAction(buttons_to_press=[Button.A], reasoning="No memory")
        
        state = self.memory_manager.get_state()
        
        # Analyze current situation
        ctx = self._analyze_situation(state)
        
        # Track if situation changed
        if ctx.situation == self.last_situation:
            self.same_situation_count += 1
        else:
            self._log(f"Situation: {ctx.situation} - {ctx.description}")
            self.same_situation_count = 0
            self.last_situation = ctx.situation
        
        # Get action - try LLM first if enabled
        action = None
        if self.use_llm and self.llm_connected and self.same_situation_count % 3 == 0:
            action = self._get_llm_action(state, ctx)
        
        # Fallback to heuristics
        if not action:
            action = self._get_heuristic_action(state, ctx)
        
        # Log action
        if action.reasoning:
            self._log(action.reasoning)
        
        # Track action
        btn_names = [b.name for b in action.buttons_to_press]
        self.action_history.append(','.join(btn_names))
        if len(self.action_history) > 50:
            self.action_history.pop(0)
        
        # Set hold
        self.current_hold = action
        self.hold_remaining = action.hold_frames - 1
        
        return action
    
    def get_thinking_output(self) -> List[str]:
        return self.thinking_history.copy()
    
    def get_current_stage_info(self) -> Dict[str, Any]:
        return {
            'stage': self.last_situation.upper(),
            'name': self.last_situation.title(),
            'goal': f"Analyzing game state ({self.same_situation_count} frames)",
        }
    
    def get_memory_state(self) -> Optional[GameState]:
        if self.memory_manager:
            return self.memory_manager.get_state()
        return None
    
    def manual_advance_stage(self):
        """Reset situation tracking."""
        self.same_situation_count = 0
        self.position_history.clear()
        self._log("Situation tracking reset")
    
    def get_status(self) -> Dict[str, Any]:
        return {
            **super().get_status(),
            'situation': self.last_situation,
            'llm_connected': self.llm_connected,
            'use_llm': self.use_llm,
        }


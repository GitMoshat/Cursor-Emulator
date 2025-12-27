"""
Memory-Based AI Agent for GBC Emulator
Uses direct memory access instead of vision - much faster and more reliable.
"""

import json
import requests
import time
import random
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

from .interface import AgentInterface, AgentAction, GameState as AgentGameState, Button, AgentConfig
from .memory_manager import MemoryManager, GameState, GameType
from .game_stages import StageManager, GameStage, POKEMON_STAGES


@dataclass
class ActionRule:
    """A rule-based action."""
    condition: str
    buttons: List[str]
    hold_frames: int
    priority: int
    description: str


class MemoryAgent(AgentInterface):
    """
    AI Agent that uses direct memory reading instead of vision.
    
    Advantages:
    - Much faster (no image encoding/LLM calls)
    - More accurate (reads exact game state)
    - Works offline (no Ollama needed for basic operation)
    - Can use LLM for complex decisions with text state
    
    Modes:
    - RULES: Pure rule-based (fastest)
    - HYBRID: Rules + LLM for complex situations
    - LLM: Always consult LLM (slowest but smartest)
    """
    
    # Rule-based actions for common situations
    DEFAULT_RULES = [
        # Battle rules (highest priority)
        ActionRule("battle.in_battle AND battle.menu_state == 0", ["A"], 5, 100, "Select FIGHT in battle"),
        ActionRule("battle.in_battle AND battle.menu_state == 1", ["A"], 5, 99, "Use first move"),
        ActionRule("battle.in_battle", ["A"], 3, 90, "Battle: press A"),
        
        # Dialog/text rules
        ActionRule("menu.text_active", ["A"], 5, 80, "Advance dialog"),
        ActionRule("menu.in_menu AND menu.menu_type == 'confirm'", ["A"], 3, 75, "Confirm selection"),
        
        # Title screen
        ActionRule("NOT game_started", ["START"], 10, 70, "Press START at title"),
        
        # Professor/starter selection
        ActionRule("NOT has_starter AND player_position.map_id == 37", ["A"], 5, 65, "In Oak's Lab - interact"),
        ActionRule("NOT has_starter", ["UP", "A"], 5, 60, "Look for starter Pokemon"),
        
        # Menu navigation
        ActionRule("menu.in_menu", ["B"], 5, 50, "Exit menu"),
        
        # Default exploration
        ActionRule("has_starter", ["UP"], 15, 10, "Move forward"),
    ]
    
    def __init__(self, config: Optional[AgentConfig] = None, mode: str = "HYBRID"):
        super().__init__(name="MemoryAgent")
        self.config = config or AgentConfig()
        self.frame_skip = 5  # Faster decisions since no vision processing
        
        # Mode: RULES, HYBRID, LLM
        self.mode = mode
        
        # Memory manager (set when initialized with emulator)
        self.memory_manager: Optional[MemoryManager] = None
        
        # Stage manager
        self.stage_manager = StageManager(POKEMON_STAGES)
        
        # Rules
        self.rules = self.DEFAULT_RULES.copy()
        
        # LLM settings (for HYBRID/LLM modes)
        self.ollama_host = self.config.ollama_host
        self.ollama_model = "llama3.2"  # Text model, not vision
        self.llm_connected = False
        
        # Action state
        self.current_hold: Optional[AgentAction] = None
        self.hold_frames_remaining = 0
        
        # Stuck detection
        self.stuck_counter = 0
        self.last_position = (0, 0)
        
        # Thinking log
        self.thinking_history: List[str] = []
        self.max_thinking_history = 100
        self.current_thinking = ""
    
    def initialize(self, **kwargs) -> bool:
        """Initialize agent with emulator reference."""
        emulator = kwargs.get('emulator')
        if emulator:
            self.memory_manager = MemoryManager(emulator)
            self.memory_manager.detect_game()
            self._log(f"Game detected: {self.memory_manager.game_type.name}")
        
        # Try to connect to Ollama for HYBRID/LLM modes
        if self.mode in ["HYBRID", "LLM"]:
            try:
                response = requests.get(f"{self.ollama_host}/api/tags", timeout=3)
                self.llm_connected = response.status_code == 200
                if self.llm_connected:
                    self._log(f"LLM connected ({self.ollama_model})")
                else:
                    self._log("LLM not available, using rules only")
            except:
                self.llm_connected = False
                self._log("LLM not available, using rules only")
        
        # Load stage memory
        try:
            self.stage_manager.load_memory()
        except:
            pass
        
        self._log(f"Mode: {self.mode}")
        self._log("Ready to play!")
        
        return True
    
    def _log(self, message: str):
        """Log a thought."""
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.thinking_history.append(entry)
        self.current_thinking = message
        
        if len(self.thinking_history) > self.max_thinking_history:
            self.thinking_history.pop(0)
        
        print(f"[MemoryAgent] {message}")
    
    def decide(self, agent_state: AgentGameState) -> AgentAction:
        """Make a decision based on memory state."""
        
        # Continue holding if needed
        if self.hold_frames_remaining > 0:
            self.hold_frames_remaining -= 1
            return self.current_hold or AgentAction()
        
        # Get game state from memory
        if not self.memory_manager:
            return AgentAction(buttons_to_press=[Button.A], reasoning="No memory manager")
        
        game_state = self.memory_manager.get_state()
        
        # Update stage manager
        self.stage_manager.update(game_state.frame_count)
        
        # Check if stuck
        current_pos = (game_state.player_position.x, game_state.player_position.y)
        if current_pos == self.last_position and not game_state.battle.in_battle:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self.last_position = current_pos
        
        # Get action based on mode
        if self.mode == "RULES" or not self.llm_connected:
            action = self._get_rule_action(game_state)
        elif self.mode == "LLM":
            action = self._get_llm_action(game_state)
        else:  # HYBRID
            # Use rules for simple cases, LLM for complex
            if self._is_complex_situation(game_state):
                action = self._get_llm_action(game_state)
            else:
                action = self._get_rule_action(game_state)
        
        # Handle stuck situation
        if self.stuck_counter > 30:
            self._log("Stuck detected! Trying random movement")
            action = self._get_unstuck_action()
            self.stuck_counter = 0
        
        # Set hold state
        self.current_hold = action
        self.hold_frames_remaining = action.hold_frames - 1
        
        return action
    
    def _get_rule_action(self, state: GameState) -> AgentAction:
        """Get action from rules."""
        
        # Sort rules by priority
        sorted_rules = sorted(self.rules, key=lambda r: -r.priority)
        
        for rule in sorted_rules:
            if self._evaluate_condition(rule.condition, state):
                buttons = [Button[b] for b in rule.buttons]
                self._log(f"Rule: {rule.description}")
                return AgentAction(
                    buttons_to_press=buttons,
                    hold_frames=rule.hold_frames,
                    reasoning=rule.description
                )
        
        # Default: press A
        self._log("No rule matched, pressing A")
        return AgentAction(
            buttons_to_press=[Button.A],
            hold_frames=5,
            reasoning="Default: press A"
        )
    
    def _evaluate_condition(self, condition: str, state: GameState) -> bool:
        """Evaluate a rule condition against game state."""
        try:
            # Build evaluation context
            ctx = {
                'battle': state.battle,
                'menu': state.menu,
                'player_position': state.player_position,
                'game_started': state.game_started,
                'has_starter': state.has_starter,
                'has_pokedex': state.has_pokedex,
                'party_count': state.party_count,
                'badges': state.badges,
                'NOT': lambda x: not x,
                'AND': lambda a, b: a and b,
                'OR': lambda a, b: a or b,
            }
            
            # Replace operators for eval
            expr = condition
            expr = expr.replace(' AND ', ' and ')
            expr = expr.replace(' OR ', ' or ')
            expr = expr.replace('NOT ', 'not ')
            
            return eval(expr, {"__builtins__": {}}, ctx)
        except Exception as e:
            return False
    
    def _is_complex_situation(self, state: GameState) -> bool:
        """Determine if situation needs LLM reasoning."""
        # Complex situations:
        # - Low HP in battle
        # - Multiple menu options
        # - Name entry
        # - Unknown map
        
        if state.battle.in_battle:
            if state.party and state.party[0].current_hp < state.party[0].max_hp * 0.3:
                return True  # Low HP, might need to use item/switch
        
        if state.menu.in_menu and state.menu.options_count > 2:
            return True  # Multiple choices
        
        return False
    
    def _get_llm_action(self, state: GameState) -> AgentAction:
        """Get action from LLM based on text state."""
        try:
            # Build prompt from game state
            state_text = state.to_prompt()
            stage_info = self.stage_manager.get_prompt_context()
            
            prompt = f"""You are playing Pokemon. Here is the current game state:

{state_text}

{stage_info}

Based on this state, what button should be pressed?
Available: A, B, UP, DOWN, LEFT, RIGHT, START, SELECT

Respond with JSON:
{{"button": "BUTTON_NAME", "hold": 1-20, "reason": "brief explanation"}}"""

            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 100, "temperature": 0.5}
                },
                timeout=15
            )
            
            if response.status_code == 200:
                text = response.json().get('response', '')
                return self._parse_llm_response(text)
            
        except Exception as e:
            self._log(f"LLM error: {e}")
        
        # Fallback to rules
        return self._get_rule_action(state)
    
    def _parse_llm_response(self, text: str) -> AgentAction:
        """Parse LLM response."""
        try:
            # Extract JSON
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                
                button_name = data.get('button', 'A').upper()
                try:
                    button = Button[button_name]
                except KeyError:
                    button = Button.A
                
                hold = min(max(data.get('hold', 5), 1), 20)
                reason = data.get('reason', 'LLM decision')
                
                self._log(f"LLM: {reason}")
                
                return AgentAction(
                    buttons_to_press=[button],
                    hold_frames=hold,
                    reasoning=reason
                )
        except:
            pass
        
        return AgentAction(buttons_to_press=[Button.A], hold_frames=5, reasoning="LLM parse failed")
    
    def _get_unstuck_action(self) -> AgentAction:
        """Get action to try to get unstuck."""
        # Try random direction
        directions = [Button.UP, Button.DOWN, Button.LEFT, Button.RIGHT]
        button = random.choice(directions)
        
        return AgentAction(
            buttons_to_press=[button],
            hold_frames=20,
            reasoning=f"Unstuck: moving {button.name}"
        )
    
    def add_rule(self, condition: str, buttons: List[str], hold: int = 5, 
                 priority: int = 50, description: str = "Custom rule"):
        """Add a custom rule."""
        rule = ActionRule(condition, buttons, hold, priority, description)
        self.rules.append(rule)
        self._log(f"Added rule: {description}")
    
    def get_thinking_output(self) -> List[str]:
        """Get thinking history for display."""
        return self.thinking_history.copy()
    
    def get_current_stage_info(self) -> Dict[str, Any]:
        """Get current stage info."""
        config = self.stage_manager.get_current_config()
        return {
            'stage': self.stage_manager.current_stage.name,
            'name': config.name,
            'goal': config.goal,
            'frames_in_stage': self.stage_manager.frames_in_stage,
        }
    
    def get_memory_state(self) -> Optional[GameState]:
        """Get current memory state for external use."""
        if self.memory_manager:
            return self.memory_manager.get_state()
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        status = super().get_status()
        status.update({
            'mode': self.mode,
            'llm_connected': self.llm_connected,
            'stuck_counter': self.stuck_counter,
            'rules_count': len(self.rules),
        })
        
        if self.memory_manager:
            state = self.memory_manager.get_state()
            status['game_state'] = {
                'position': f"({state.player_position.x}, {state.player_position.y})",
                'map': state.player_position.map_name,
                'in_battle': state.battle.in_battle,
                'party_count': state.party_count,
            }
        
        return status
    
    def manual_advance_stage(self):
        """Manually advance stage."""
        self.stage_manager.advance_stage()
    
    def shutdown(self):
        """Save and cleanup."""
        super().shutdown()
        try:
            self.stage_manager.save_memory()
        except:
            pass


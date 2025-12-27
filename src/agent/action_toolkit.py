"""
Action Toolkit for AI Agent
Provides discoverable actions the AI can request - similar to MCP/function calling.
The AI queries available actions and their descriptions, then requests them by name.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import random

from .interface import Button


@dataclass
class ActionResult:
    """Result of executing an action."""
    success: bool
    buttons: List[Button]
    hold_frames: int
    message: str
    next_suggestion: Optional[str] = None  # Suggested follow-up action


@dataclass
class ActionDefinition:
    """Definition of an available action."""
    name: str
    description: str
    when_to_use: str
    parameters: Dict[str, str] = field(default_factory=dict)  # param_name -> description
    examples: List[str] = field(default_factory=list)
    
    def to_prompt(self) -> str:
        """Format for LLM prompt."""
        lines = [
            f"**{self.name}**",
            f"  Description: {self.description}",
            f"  When to use: {self.when_to_use}",
        ]
        if self.parameters:
            lines.append("  Parameters:")
            for p, desc in self.parameters.items():
                lines.append(f"    - {p}: {desc}")
        if self.examples:
            lines.append(f"  Example: {self.examples[0]}")
        return '\n'.join(lines)


class ActionToolkit:
    """
    Toolkit of actions the AI can discover and request.
    
    The AI doesn't need to know HOW to do things - it just requests
    actions by name, and the toolkit handles the button presses.
    
    Example flow:
    1. AI queries: "What actions are available?"
    2. Toolkit responds with action list and descriptions
    3. AI requests: {"action": "advance_dialog"}
    4. Toolkit returns the buttons to press
    """
    
    def __init__(self, memory_manager=None):
        self.memory_manager = memory_manager
        self.actions: Dict[str, ActionDefinition] = {}
        self.handlers: Dict[str, Callable] = {}
        
        # Register default actions
        self._register_default_actions()
    
    def _register_default_actions(self):
        """Register the built-in action set."""
        
        # === Navigation Actions ===
        self.register_action(
            ActionDefinition(
                name="move_up",
                description="Move the player character upward",
                when_to_use="When you need to move up on the map, in menus, or navigate north",
                examples=["Move up to enter a building", "Navigate up in a menu"]
            ),
            self._handle_move_up
        )
        
        self.register_action(
            ActionDefinition(
                name="move_down",
                description="Move the player character downward",
                when_to_use="When you need to move down on the map or navigate south",
                examples=["Move down to exit a building"]
            ),
            self._handle_move_down
        )
        
        self.register_action(
            ActionDefinition(
                name="move_left",
                description="Move the player character left",
                when_to_use="When you need to move left/west on the map",
            ),
            self._handle_move_left
        )
        
        self.register_action(
            ActionDefinition(
                name="move_right",
                description="Move the player character right",
                when_to_use="When you need to move right/east on the map",
            ),
            self._handle_move_right
        )
        
        self.register_action(
            ActionDefinition(
                name="explore",
                description="Move in a random direction to explore the area",
                when_to_use="When you're not sure where to go and want to explore",
            ),
            self._handle_explore
        )
        
        # === Interaction Actions ===
        self.register_action(
            ActionDefinition(
                name="interact",
                description="Press A to interact with whatever is in front of you",
                when_to_use="To talk to NPCs, examine objects, pick up items, confirm selections",
                examples=["Talk to Professor", "Pick up item", "Read sign"]
            ),
            self._handle_interact
        )
        
        self.register_action(
            ActionDefinition(
                name="cancel",
                description="Press B to cancel, go back, or decline",
                when_to_use="To exit menus, cancel selections, or decline prompts",
                examples=["Exit menu", "Say no to prompt", "Go back"]
            ),
            self._handle_cancel
        )
        
        self.register_action(
            ActionDefinition(
                name="advance_dialog",
                description="Press A to advance text/dialog that is currently showing",
                when_to_use="When text or dialog boxes are on screen and need to be dismissed",
                examples=["Continue conversation", "Dismiss message box"]
            ),
            self._handle_advance_dialog
        )
        
        # === Menu Actions ===
        self.register_action(
            ActionDefinition(
                name="open_menu",
                description="Press START to open the game menu",
                when_to_use="When you want to access the main game menu (Pokemon, Items, Save, etc)",
            ),
            self._handle_open_menu
        )
        
        self.register_action(
            ActionDefinition(
                name="start_game",
                description="Press START to begin the game from title screen",
                when_to_use="When at the title screen and you want to start playing",
            ),
            self._handle_start_game
        )
        
        self.register_action(
            ActionDefinition(
                name="select_option",
                description="Press A to select the currently highlighted menu option",
                when_to_use="When in a menu and you want to choose the highlighted option",
            ),
            self._handle_select_option
        )
        
        self.register_action(
            ActionDefinition(
                name="menu_up",
                description="Move cursor up in a menu",
                when_to_use="When in a menu and you want to highlight the option above",
            ),
            self._handle_menu_up
        )
        
        self.register_action(
            ActionDefinition(
                name="menu_down",
                description="Move cursor down in a menu",
                when_to_use="When in a menu and you want to highlight the option below",
            ),
            self._handle_menu_down
        )
        
        # === Battle Actions ===
        self.register_action(
            ActionDefinition(
                name="battle_fight",
                description="Select FIGHT option in battle to attack",
                when_to_use="When in battle and you want to attack the opponent",
            ),
            self._handle_battle_fight
        )
        
        self.register_action(
            ActionDefinition(
                name="battle_run",
                description="Try to run away from battle",
                when_to_use="When in battle and you want to escape (may fail against trainers)",
            ),
            self._handle_battle_run
        )
        
        # === Compound Actions ===
        self.register_action(
            ActionDefinition(
                name="wait",
                description="Do nothing for a moment",
                when_to_use="When you need to wait for animations, loading, or transitions",
            ),
            self._handle_wait
        )
        
        self.register_action(
            ActionDefinition(
                name="mash_a",
                description="Repeatedly press A to quickly advance through dialog/cutscenes",
                when_to_use="When there's a lot of dialog to get through quickly",
            ),
            self._handle_mash_a
        )
    
    def register_action(self, definition: ActionDefinition, handler: Callable):
        """Register a new action."""
        self.actions[definition.name] = definition
        self.handlers[definition.name] = handler
    
    def get_available_actions(self) -> List[ActionDefinition]:
        """Get list of all available actions."""
        return list(self.actions.values())
    
    def get_action_prompt(self) -> str:
        """Get formatted list of actions for LLM prompt."""
        lines = ["=== AVAILABLE ACTIONS ===", ""]
        for action in self.actions.values():
            lines.append(action.to_prompt())
            lines.append("")
        lines.append("To use an action, respond with: {\"action\": \"action_name\"}")
        return '\n'.join(lines)
    
    def get_contextual_actions(self, situation: str) -> List[str]:
        """Get suggested actions for a given situation."""
        suggestions = {
            "title": ["start_game", "interact"],
            "dialog": ["advance_dialog", "mash_a", "cancel"],
            "menu": ["select_option", "menu_up", "menu_down", "cancel"],
            "battle": ["battle_fight", "select_option", "battle_run"],
            "overworld": ["interact", "explore", "move_up", "move_down", "move_left", "move_right", "open_menu"],
            "unknown": ["interact", "start_game", "advance_dialog"],
        }
        return suggestions.get(situation, suggestions["unknown"])
    
    def execute_action(self, action_name: str, **params) -> ActionResult:
        """Execute an action by name."""
        if action_name not in self.handlers:
            return ActionResult(
                success=False,
                buttons=[],
                hold_frames=0,
                message=f"Unknown action: {action_name}"
            )
        
        try:
            return self.handlers[action_name](**params)
        except Exception as e:
            return ActionResult(
                success=False,
                buttons=[],
                hold_frames=0,
                message=f"Action failed: {e}"
            )
    
    # === Action Handlers ===
    
    def _handle_move_up(self, **kwargs) -> ActionResult:
        return ActionResult(True, [Button.UP], 12, "Moving up", "interact")
    
    def _handle_move_down(self, **kwargs) -> ActionResult:
        return ActionResult(True, [Button.DOWN], 12, "Moving down", "interact")
    
    def _handle_move_left(self, **kwargs) -> ActionResult:
        return ActionResult(True, [Button.LEFT], 12, "Moving left", "interact")
    
    def _handle_move_right(self, **kwargs) -> ActionResult:
        return ActionResult(True, [Button.RIGHT], 12, "Moving right", "interact")
    
    def _handle_explore(self, **kwargs) -> ActionResult:
        direction = random.choice([Button.UP, Button.DOWN, Button.LEFT, Button.RIGHT])
        return ActionResult(True, [direction], 15, f"Exploring {direction.name}", "interact")
    
    def _handle_interact(self, **kwargs) -> ActionResult:
        return ActionResult(True, [Button.A], 5, "Interacting", "advance_dialog")
    
    def _handle_cancel(self, **kwargs) -> ActionResult:
        return ActionResult(True, [Button.B], 5, "Canceling/going back")
    
    def _handle_advance_dialog(self, **kwargs) -> ActionResult:
        return ActionResult(True, [Button.A], 8, "Advancing dialog", "advance_dialog")
    
    def _handle_open_menu(self, **kwargs) -> ActionResult:
        return ActionResult(True, [Button.START], 5, "Opening menu")
    
    def _handle_start_game(self, **kwargs) -> ActionResult:
        return ActionResult(True, [Button.START], 10, "Starting game", "advance_dialog")
    
    def _handle_select_option(self, **kwargs) -> ActionResult:
        return ActionResult(True, [Button.A], 5, "Selecting option")
    
    def _handle_menu_up(self, **kwargs) -> ActionResult:
        return ActionResult(True, [Button.UP], 3, "Menu cursor up")
    
    def _handle_menu_down(self, **kwargs) -> ActionResult:
        return ActionResult(True, [Button.DOWN], 3, "Menu cursor down")
    
    def _handle_battle_fight(self, **kwargs) -> ActionResult:
        return ActionResult(True, [Button.A], 5, "Selecting FIGHT", "select_option")
    
    def _handle_battle_run(self, **kwargs) -> ActionResult:
        # Run is usually bottom-right: DOWN, RIGHT, then A
        return ActionResult(True, [Button.DOWN], 3, "Selecting RUN")
    
    def _handle_wait(self, **kwargs) -> ActionResult:
        return ActionResult(True, [], 30, "Waiting")
    
    def _handle_mash_a(self, **kwargs) -> ActionResult:
        return ActionResult(True, [Button.A], 3, "Mashing A", "mash_a")


class ToolkitAgent:
    """
    AI Agent that uses the ActionToolkit to discover and request actions.
    
    Instead of directly deciding buttons, it:
    1. Gets the list of available actions
    2. Analyzes the situation
    3. Requests an action by name
    4. The toolkit converts that to button presses
    """
    
    def __init__(self, config=None):
        from .interface import AgentConfig
        self.name = "ToolkitAgent"
        self.enabled = False
        self.frame_skip = 6
        self._frame_counter = 0
        self.config = config or AgentConfig()
        
        self.toolkit: Optional[ActionToolkit] = None
        self.memory_manager = None
        
        # LLM
        self.ollama_host = self.config.ollama_host
        self.ollama_model = "llama3.2"
        self.llm_connected = False
        
        # State
        self.thinking_history: List[str] = []
        self.last_situation = ""
        self.pending_action: Optional[str] = None
        self.hold_remaining = 0
        self.current_result: Optional[ActionResult] = None
        self.action_history: List = []
        
        import time
        self.time = time
    
    def initialize(self, **kwargs) -> bool:
        from .memory_manager import MemoryManager
        
        emulator = kwargs.get('emulator')
        if emulator:
            self.memory_manager = MemoryManager(emulator)
            self.memory_manager.detect_game()
            self.toolkit = ActionToolkit(self.memory_manager)
            self._log(f"Toolkit loaded with {len(self.toolkit.actions)} actions")
        
        # Try LLM
        try:
            import requests
            resp = requests.get(f"{self.ollama_host}/api/tags", timeout=3)
            self.llm_connected = resp.status_code == 200
            self._log(f"LLM: {'connected' if self.llm_connected else 'offline'}")
        except:
            self.llm_connected = False
        
        self._log("ToolkitAgent ready")
        return True
    
    def _log(self, msg: str):
        ts = self.time.strftime("%H:%M:%S")
        self.thinking_history.append(f"[{ts}] {msg}")
        if len(self.thinking_history) > 100:
            self.thinking_history.pop(0)
        print(f"[ToolkitAgent] {msg}")
    
    def _analyze_situation(self, state) -> str:
        """Determine current situation."""
        if state.battle.in_battle:
            return "battle"
        if state.menu.text_active:
            return "dialog"
        if state.menu.in_menu:
            return "menu"
        if not state.game_started and state.party_count == 0:
            return "title"
        return "overworld"
    
    def _get_llm_action(self, state, situation: str) -> Optional[str]:
        """Ask LLM which action to take."""
        if not self.llm_connected:
            return None
        
        import requests
        
        # Build prompt with available actions
        available = self.toolkit.get_contextual_actions(situation)
        actions_desc = "\n".join(f"- {a}: {self.toolkit.actions[a].description}" for a in available)
        
        prompt = f"""Current situation: {situation}
Player at: ({state.player_position.x}, {state.player_position.y}) - {state.player_position.map_name}
Party: {state.party_count} Pokemon
Has starter: {state.has_starter}

Available actions:
{actions_desc}

Which action should I take? Respond with just the action name."""

        try:
            resp = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "system": "You are playing a Pokemon game. Choose the best action from the list. Respond with ONLY the action name, nothing else.",
                    "stream": False,
                    "options": {"num_predict": 20, "temperature": 0.5}
                },
                timeout=8
            )
            
            if resp.status_code == 200:
                action_name = resp.json().get('response', '').strip().lower()
                # Clean up response
                action_name = action_name.replace('"', '').replace("'", "").split()[0] if action_name else ""
                if action_name in self.toolkit.actions:
                    return action_name
        except:
            pass
        
        return None
    
    def _get_heuristic_action(self, situation: str) -> str:
        """Get action using simple heuristics."""
        if situation == "title":
            return "start_game"
        elif situation == "dialog":
            return "advance_dialog"
        elif situation == "menu":
            return "select_option"
        elif situation == "battle":
            return "battle_fight"
        else:  # overworld
            import random
            if random.random() < 0.3:
                return "interact"
            return "explore"
    
    def decide(self, agent_state) -> 'AgentAction':
        from .interface import AgentAction
        
        # Continue current action
        if self.hold_remaining > 0:
            self.hold_remaining -= 1
            if self.current_result:
                return AgentAction(
                    buttons_to_press=self.current_result.buttons,
                    hold_frames=1,
                    reasoning=self.current_result.message
                )
            return AgentAction()
        
        if not self.memory_manager or not self.toolkit:
            return AgentAction(buttons_to_press=[Button.A], reasoning="No toolkit")
        
        # Get state and situation
        state = self.memory_manager.get_state()
        situation = self._analyze_situation(state)
        
        if situation != self.last_situation:
            self._log(f"Situation: {situation}")
            self.last_situation = situation
        
        # Get action - try LLM, fallback to heuristic
        action_name = None
        if self.llm_connected:
            action_name = self._get_llm_action(state, situation)
        
        if not action_name:
            action_name = self._get_heuristic_action(situation)
        
        # Execute action through toolkit
        result = self.toolkit.execute_action(action_name)
        self._log(f"Action: {action_name} -> {result.message}")
        
        self.current_result = result
        self.hold_remaining = result.hold_frames - 1
        
        return AgentAction(
            buttons_to_press=result.buttons,
            hold_frames=result.hold_frames,
            reasoning=f"{action_name}: {result.message}"
        )
    
    def get_thinking_output(self) -> List[str]:
        return self.thinking_history.copy()
    
    def get_current_stage_info(self) -> Dict[str, Any]:
        return {
            'stage': self.last_situation.upper(),
            'name': f"Toolkit - {self.last_situation.title()}",
            'goal': "Using action toolkit to play",
        }
    
    def get_memory_state(self):
        if self.memory_manager:
            return self.memory_manager.get_state()
        return None
    
    def manual_advance_stage(self):
        self._log("Manual reset")
    
    def process_frame(self, state) -> Optional['AgentAction']:
        """Process a frame - called by manager."""
        if not self.enabled:
            return None
        
        self._frame_counter += 1
        if self._frame_counter <= self.frame_skip:
            return None
        self._frame_counter = 0
        
        return self.decide(state)
    
    def reset(self):
        """Reset agent state."""
        self.thinking_history.clear()
        self._frame_counter = 0
    
    def shutdown(self):
        """Clean up."""
        self.enabled = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get status."""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'situation': self.last_situation,
            'llm_connected': self.llm_connected,
        }


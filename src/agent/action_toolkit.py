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
        
        # === Name Entry Actions ===
        self.register_action(
            ActionDefinition(
                name="enter_name",
                description="Enter a random Pokemon-style name on the name entry screen",
                when_to_use="When on the YOUR NAME? screen with character grid",
            ),
            self._handle_enter_name
        )
        
        self.register_action(
            ActionDefinition(
                name="name_grid_right",
                description="Move cursor right on name entry grid",
                when_to_use="To navigate to the next character on name entry screen",
            ),
            self._handle_name_grid_right
        )
        
        self.register_action(
            ActionDefinition(
                name="name_grid_left",
                description="Move cursor left on name entry grid",
                when_to_use="To navigate to previous character on name entry screen",
            ),
            self._handle_name_grid_left
        )
        
        self.register_action(
            ActionDefinition(
                name="name_grid_down",
                description="Move cursor down on name entry grid",
                when_to_use="To move down a row on name entry screen",
            ),
            self._handle_name_grid_down
        )
        
        self.register_action(
            ActionDefinition(
                name="name_select_char",
                description="Select current character on name entry grid",
                when_to_use="To add the highlighted character to the name",
            ),
            self._handle_name_select_char
        )
        
        self.register_action(
            ActionDefinition(
                name="name_confirm",
                description="Confirm name entry (press END)",
                when_to_use="After entering characters, to confirm the name",
            ),
            self._handle_name_confirm
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
            "name_entry": ["enter_name", "name_select_char", "name_grid_right", "name_grid_down", "name_confirm"],
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
    
    # === Name Entry Handlers ===
    
    def _handle_enter_name(self, **kwargs) -> ActionResult:
        """Start entering a random name - handled by ToolkitAgent's name entry state machine."""
        # This signals the agent to begin the name entry sequence
        return ActionResult(True, [], 1, "Starting name entry", "name_grid_right")
    
    def _handle_name_grid_right(self, **kwargs) -> ActionResult:
        return ActionResult(True, [Button.RIGHT], 4, "Grid right")
    
    def _handle_name_grid_left(self, **kwargs) -> ActionResult:
        return ActionResult(True, [Button.LEFT], 4, "Grid left")
    
    def _handle_name_grid_down(self, **kwargs) -> ActionResult:
        return ActionResult(True, [Button.DOWN], 4, "Grid down")
    
    def _handle_name_select_char(self, **kwargs) -> ActionResult:
        return ActionResult(True, [Button.A], 6, "Select char", "name_grid_right")
    
    def _handle_name_confirm(self, **kwargs) -> ActionResult:
        return ActionResult(True, [Button.A], 8, "Confirm name")


class ToolkitAgent:
    """
    AI Agent that uses the ActionToolkit to discover and request actions.
    
    Instead of directly deciding buttons, it:
    1. Gets the current goal from the goal system
    2. Analyzes the situation
    3. Requests an action that helps achieve the goal
    4. The toolkit converts that to button presses
    5. Checks if goal is complete
    """
    
    def __init__(self, config=None):
        from .interface import AgentConfig
        from .goal_system import GoalSystem
        
        self.name = "ToolkitAgent"
        self.enabled = False
        self.frame_skip = 15  # Increased for better FPS - decide every 15 frames
        self._frame_counter = 0
        self.config = config or AgentConfig()
        
        self.toolkit: Optional[ActionToolkit] = None
        self.memory_manager = None
        self.goal_system = GoalSystem()  # Goal tracking
        
        # LLM
        self.ollama_host = self.config.ollama_host
        self.ollama_model = "llama3.2"
        self.llm_connected = False
        self.fast_mode = False  # Keep LLM enabled - do not disable
        
        # State
        self.thinking_history: List[str] = []
        self.last_situation = ""
        self.last_goal_id = ""
        self.pending_action: Optional[str] = None
        self.hold_remaining = 0
        self.current_result: Optional[ActionResult] = None
        self.action_history: List = []
        
        # Debug focus - what the AI is looking at / interacting with
        self.current_focus = {
            'type': 'none',           # 'dialog', 'menu_option', 'button', 'area', 'player'
            'action': '',             # What action is being taken
            'target': '',             # What is being targeted
            'rect': None,             # (x, y, w, h) in game pixels (160x144)
            'label': '',              # Label to show
        }
        
        # Name entry state machine
        self.name_entry_active = False
        self.name_to_enter = ""
        self.name_char_index = 0  # Which character we're entering
        self.name_entry_step = "idle"  # idle, navigating, selecting, confirming
        self.target_grid_x = 0
        self.target_grid_y = 0
        
        # Async LLM support - don't block game loop
        self.llm_thread = None
        self.llm_result = None
        self.llm_pending = False
        self.last_llm_call_time = 0
        self.llm_cooldown = 2.0  # Seconds between LLM calls
        
        # Performance tracking
        self.perf_stats = {
            'llm_calls': 0,
            'llm_avg_ms': 0,
            'heuristic_calls': 0,
            'state_reads': 0,
            'frames_processed': 0,
        }
        self._llm_times = []
        
        import time
        self.time = time
        import threading
        self.threading = threading
    
    def _generate_random_name(self) -> str:
        """Generate a random Pokemon-style name."""
        import random
        # Pokemon-style names - short, memorable, mix of styles
        POKEMON_NAMES = [
            "RED", "BLUE", "ASH", "GOLD", "LEAF",
            "ETHAN", "LYRA", "KRIS", "MAY", "DAWN",
            "LUCAS", "BARRY", "CYNTHIA", "LANCE", "OAK",
            "GARY", "BROCK", "MISTY", "ERIKA", "BLAINE",
            "ACE", "MAX", "REX", "LEO", "NINA",
            "ZACK", "LUNA", "JADE", "RUBY", "ALEX",
            "SAM", "KAI", "RAY", "SKY", "STAR",
        ]
        return random.choice(POKEMON_NAMES)
    
    def _get_char_grid_position(self, char: str) -> tuple:
        """Get grid (x, y) position for a character on the name entry screen."""
        # Pokemon Crystal character grid layout (uppercase)
        # Row 0: A B C D E F G H I J
        # Row 1: K L M N O P Q R S T
        # Row 2: U V W X Y Z (spaces)
        # Row 3: special characters
        # Row 4: lower DEL END
        
        GRID = [
            "ABCDEFGHIJ",
            "KLMNOPQRST",
            "UVWXYZ    ",
            "          ",  # Special chars - skip
        ]
        
        char = char.upper()
        for y, row in enumerate(GRID):
            if char in row:
                x = row.index(char)
                return (x, y)
        
        # Default to A if character not found
        return (0, 0)
    
    def _get_smart_menu_action(self, state) -> str:
        """Smart menu/dialog handling - adaptive, not hardcoded.
        
        When in a menu or seeing dialog:
        - If text is showing, usually press A to continue
        - If making a selection, press A to confirm
        - Use arrows to navigate if needed
        """
        # Track attempts to avoid infinite loops
        if not hasattr(self, '_menu_attempts'):
            self._menu_attempts = 0
        self._menu_attempts += 1
        
        # If text/dialog is showing, press A to advance/confirm
        if state.menu.text_active:
            return "advance_dialog"
        
        # If in a menu, try selecting current option
        # This handles time selection, gender selection, etc.
        if state.menu.in_menu:
            # Every few attempts, try A to select
            if self._menu_attempts % 3 == 0:
                return "select_option"
            # Sometimes try navigating
            elif self._menu_attempts % 5 == 0:
                return "menu_down"
            else:
                return "advance_dialog"
        
        return "advance_dialog"
    
    def _get_name_entry_action(self, state) -> str:
        """Handle name entry - be adaptive.
        
        The name entry screen has a character grid.
        Strategy: Try START for presets, then navigate to END and press A.
        """
        current_x = state.menu.cursor_x
        current_y = state.menu.cursor_y
        current_name = state.player_name
        
        # Track attempts
        if not hasattr(self, '_name_entry_attempts'):
            self._name_entry_attempts = 0
            self._log(f"[NAME ENTRY] Starting - cursor: ({current_x}, {current_y})")
        
        self._name_entry_attempts += 1
        
        # Check if name changed (success!)
        name_is_placeholder = current_name and all(c == '?' for c in current_name.strip())
        if not name_is_placeholder and len(current_name.strip()) > 0:
            self._log(f"[NAME ENTRY SUCCESS!] Name: '{current_name}'")
            self._name_entry_attempts = 0
            return "advance_dialog"
        
        # Log occasionally
        if self._name_entry_attempts % 15 == 0:
            self._log(f"[NAME ENTRY] Attempt {self._name_entry_attempts}, cursor=({current_x},{current_y})")
        
        # Simple adaptive strategy:
        # 1. First try START to get preset names (attempts 1-20)
        # 2. Then try navigating to END and pressing A (attempts 21+)
        
        if self._name_entry_attempts <= 20:
            return "start_game"  # Press START for presets
        else:
            # Cycle through: down, right, A
            cycle = self._name_entry_attempts % 4
            if cycle == 0:
                return "name_grid_down"
            elif cycle == 1:
                return "name_grid_right"
            else:
                return "select_option"  # Press A
    
    def initialize(self, **kwargs) -> bool:
        from .memory_manager import MemoryManager
        
        emulator = kwargs.get('emulator')
        if emulator:
            self.memory_manager = MemoryManager(emulator)
            self.memory_manager.detect_game()
            self.toolkit = ActionToolkit(self.memory_manager)
            self._log(f"Toolkit: {len(self.toolkit.actions)} actions available")
        
        # Try LLM
        try:
            import requests
            resp = requests.get(f"{self.ollama_host}/api/tags", timeout=3)
            self.llm_connected = resp.status_code == 200
            self._log(f"LLM: {'connected' if self.llm_connected else 'offline (using heuristics)'}")
        except:
            self.llm_connected = False
            self._log("LLM: offline (using heuristics)")
        
        # Log initial goal
        goal = self.goal_system.get_current_goal()
        if goal:
            self._log(f"GOAL: {goal.name}")
            self._log(f"  -> {goal.description}")
        
        self._log("ToolkitAgent ready!")
        return True
    
    def _log(self, msg: str):
        ts = self.time.strftime("%H:%M:%S")
        self.thinking_history.append(f"[{ts}] {msg}")
        if len(self.thinking_history) > 100:
            self.thinking_history.pop(0)
        print(f"[ToolkitAgent] {msg}")
    
    def _analyze_situation(self, state) -> str:
        """Determine current situation - be ADAPTIVE, not hardcoded."""
        screen_type = state.menu.screen_type
        
        # Log screen type changes for debugging
        if not hasattr(self, '_last_screen_type') or self._last_screen_type != screen_type:
            cursor = state.menu
            self._log(f"[SCREEN CHANGE] {self._last_screen_type if hasattr(self, '_last_screen_type') else 'none'} -> {screen_type}")
            self._log(f"[SCREEN INFO] cursor=({cursor.cursor_x},{cursor.cursor_y}), menu={cursor.in_menu}, text={cursor.text_active}")
            self._last_screen_type = screen_type
        
        # Let the LLM handle complex situations - don't over-categorize
        # Just provide basic context and let intelligence decide
        
        if state.battle.in_battle:
            return "battle"
        
        # Name entry is special because it has a character grid
        if screen_type == "name_entry":
            return "name_entry"
        
        # Dialog/text showing - could be a question, could be info
        if state.menu.text_active:
            return "dialog"
        
        # Menu open - could be many things (time select, options, etc.)
        if state.menu.in_menu:
            return "menu"  # Let LLM figure out what kind of menu
        
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
    
    def _get_heuristic_action(self, situation: str, state=None) -> str:
        """Get action using simple heuristics."""
        import random
        
        if situation == "title":
            return "start_game"
        elif situation == "dialog":
            return "advance_dialog"
        elif situation == "name_entry":
            # Use name entry state machine
            if state:
                return self._get_name_entry_action(state)
            return "select_option"  # Just press A
        elif situation == "menu":
            return "select_option"
        elif situation == "battle":
            return "battle_fight"
        elif situation == "overworld":
            # Actually explore! Mostly move, occasionally interact
            roll = random.random()
            if roll < 0.15:
                return "interact"  # 15% interact
            elif roll < 0.35:
                return "move_up"   # 20% up
            elif roll < 0.55:
                return "move_down" # 20% down
            elif roll < 0.75:
                return "move_left" # 20% left
            else:
                return "move_right" # 25% right
        else:
            # Unknown - try to advance
            return "advance_dialog"
    
    def _get_goal_heuristic_action(self, situation: str, goal, state=None) -> str:
        """Get action based on situation - ADAPTIVE, let LLM handle complex cases."""
        import random
        
        # Name entry has special handling (character grid)
        if situation == "name_entry":
            if state:
                return self._get_name_entry_action(state)
            return "select_option"
        
        # Dialog and menu - use SMART adaptive handling
        # This handles time selection, gender selection, confirmations, etc.
        if situation in ["dialog", "menu"]:
            if state:
                return self._get_smart_menu_action(state)
            return "advance_dialog"  # Default to pressing A
        
        # Handle battle
        if situation == "battle":
            return "battle_fight"
        
        # Handle title
        if situation == "title":
            return "start_game"
        
        # Handle overworld - the main exploration
        if situation == "overworld":
            if not goal:
                return self._get_heuristic_action(situation, state)
            
            goal_id = goal.id
            
            if goal_id == "leave_house":
                # Mostly move down to find exit
                roll = random.random()
                if roll < 0.5:
                    return "move_down"
                elif roll < 0.7:
                    return "move_left"
                elif roll < 0.9:
                    return "move_right"
                else:
                    return "interact"
            
            elif goal_id == "find_professor":
                # Explore all directions, interact with NPCs
                roll = random.random()
                if roll < 0.2:
                    return "interact"
                elif roll < 0.4:
                    return "move_up"
                elif roll < 0.6:
                    return "move_down"
                elif roll < 0.8:
                    return "move_left"
                else:
                    return "move_right"
            
            elif goal_id == "get_starter":
                # Interact with everything
                if random.random() < 0.4:
                    return "interact"
                return "explore"
            
            else:
                # Generic exploration
                return self._get_heuristic_action(situation, state)
        
        # Default
        return self._get_heuristic_action(situation, state)
    
    def _build_game_context(self, state, situation: str) -> str:
        """Build detailed game context for LLM - what's ACTUALLY on screen."""
        lines = []
        
        # Screen type with explanation
        screen_explanations = {
            "title": "Title screen - press START or A to begin",
            "name_entry": "Name entry screen - character grid to type a name, or select preset",
            "gender_select": "Gender/option selection - choose BOY or GIRL",
            "dialog": "Dialog/text box showing - need to press A to continue",
            "menu": "Menu is open - navigate with arrows, select with A",
            "battle": "In Pokemon battle - choose FIGHT, POKEMON, ITEM, or RUN",
            "overworld": "Walking around the game world - can move and interact",
            "intro": "Introduction sequence - usually need to press A or make choices",
            "time_select": "Time selection screen - choose what time it is",
        }
        
        # Detect time selection specifically
        if "time" in str(state.player_name).lower() or situation == "name_entry":
            # Check if this might be time selection based on context
            if state.menu.cursor_y > 5 or state.menu.cursor_x > 5:
                lines.append("SCREEN: Time/option selection - picking from a list")
            else:
                lines.append(f"SCREEN: {screen_explanations.get(situation, situation)}")
        else:
            lines.append(f"SCREEN: {screen_explanations.get(situation, situation)}")
        
        # Menu state details
        if state.menu.in_menu:
            lines.append(f"MENU: Open at cursor position ({state.menu.cursor_x}, {state.menu.cursor_y})")
            if state.menu.options_count > 0:
                lines.append(f"OPTIONS: {state.menu.options_count} choices available")
        
        # Text/dialog active
        if state.menu.text_active:
            lines.append("TEXT: Dialog box is showing - press A to continue or make selection")
        
        # Player name status
        if state.player_name:
            is_placeholder = all(c == '?' for c in state.player_name.strip())
            if is_placeholder:
                lines.append("NAME: Not yet entered (showing placeholder)")
            else:
                lines.append(f"NAME: '{state.player_name}'")
        
        # Location
        lines.append(f"LOCATION: ({state.player_position.x}, {state.player_position.y}) - {state.player_position.map_name}")
        
        # Party status
        if state.party_count > 0:
            lines.append(f"PARTY: {state.party_count} Pokemon")
        else:
            lines.append("PARTY: No Pokemon yet")
        
        return "\n".join(lines)
    
    def _start_async_llm_call(self, state, situation: str, goal):
        """Start async LLM call in background thread."""
        if self.llm_pending or not self.llm_connected:
            return
        
        # Check cooldown
        current_time = self.time.time()
        if current_time - self.last_llm_call_time < self.llm_cooldown:
            return
        
        self.last_llm_call_time = current_time
        self.llm_pending = True
        self.llm_result = None
        
        # Build DETAILED context about what's on screen
        game_context = self._build_game_context(state, situation)
        
        # Get available actions with descriptions
        available = self.toolkit.get_contextual_actions(situation)
        actions_desc = "\n".join(f"- {a}: {self.toolkit.actions[a].description}" for a in available if a in self.toolkit.actions)
        
        goal_context = ""
        if goal:
            goal_context = f"CURRENT GOAL: {goal.name}\nObjective: {goal.description}\n"
        
        prompt = f"""You are an AI playing Pokemon Crystal. Look at the current game state and choose the BEST action.

{goal_context}
=== CURRENT GAME STATE ===
{game_context}

=== AVAILABLE ACTIONS ===
{actions_desc}

IMPORTANT: 
- If you see a question or selection (like "What time is it?"), use select_option (A) to confirm your choice
- If you see a text box, use advance_dialog (A) to continue
- If entering a name, navigate the grid and select characters
- Think about what the screen is asking you to do

What action should I take? Reply with ONLY the action name, nothing else."""

        def llm_worker():
            import requests
            start_time = self.time.time()
            try:
                resp = requests.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "system": "You are a Pokemon player AI. Reply with ONLY an action name.",
                        "stream": False,
                        "options": {"num_predict": 30, "temperature": 0.5}
                    },
                    timeout=8
                )
                
                elapsed_ms = (self.time.time() - start_time) * 1000
                self._llm_times.append(elapsed_ms)
                if len(self._llm_times) > 20:
                    self._llm_times.pop(0)
                self.perf_stats['llm_avg_ms'] = sum(self._llm_times) / len(self._llm_times)
                self.perf_stats['llm_calls'] += 1
                
                if resp.status_code == 200:
                    raw_response = resp.json().get('response', '').strip()
                    action_name = raw_response.lower().replace('"', '').replace("'", "")
                    action_name = action_name.split()[0] if action_name else ""
                    
                    if action_name in self.toolkit.actions:
                        self.llm_result = action_name
                        self._log(f"[ASYNC LLM] -> {action_name} ({int(elapsed_ms)}ms)")
            except Exception as e:
                self._log(f"[ASYNC LLM] Error: {e}")
            finally:
                self.llm_pending = False
        
        self.llm_thread = self.threading.Thread(target=llm_worker, daemon=True)
        self.llm_thread.start()
    
    def _get_async_llm_result(self) -> Optional[str]:
        """Get result from async LLM call if available."""
        if self.llm_result:
            result = self.llm_result
            self.llm_result = None
            return result
        return None
    
    def _get_llm_action_with_goal(self, state, situation: str, goal) -> Optional[str]:
        """Get LLM action - now async, returns cached result or starts new call."""
        # Check for completed async result first
        result = self._get_async_llm_result()
        if result:
            return result
        
        # Start new async call if not pending (will be ready next decision)
        if not self.llm_pending:
            self._start_async_llm_call(state, situation, goal)
        
        # Return None - use heuristics this frame
        return None
    
    def decide(self, agent_state) -> 'AgentAction':
        from .interface import AgentAction
        
        self.perf_stats['frames_processed'] += 1
        
        if not self.memory_manager or not self.toolkit:
            return AgentAction(buttons_to_press=[Button.A], reasoning="No toolkit")
        
        # During hold frames - just return the held action, no processing
        if self.hold_remaining > 0:
            self.hold_remaining -= 1
            if self.current_result:
                return AgentAction(
                    buttons_to_press=self.current_result.buttons,
                    hold_frames=1,
                    reasoning=self.current_result.message
                )
            return AgentAction()
        
        # Only read state when making NEW decisions (not during holds)
        self.perf_stats['state_reads'] += 1
        state = self.memory_manager.get_state()
        situation = self._analyze_situation(state)
        
        # Update focus for debug overlay
        current_action = self.current_result.message.split(':')[0] if self.current_result and ':' in self.current_result.message else "waiting"
        self._update_focus(current_action, state, situation)
        
        # Log detailed perception of the state
        self._log_perception(state, situation)
        
        # Check goal progress
        goal = self.goal_system.get_current_goal()
        if goal:
            check = self.goal_system.check_goal_completion(state)
            
            # Log goal changes
            if goal.id != self.last_goal_id:
                self._log(f"=== GOAL: {goal.name} ===")
                self._log(f"  {goal.description}")
                self.last_goal_id = goal.id
            
            # Goal completed!
            if check.is_complete:
                self._log(f"[OK] GOAL COMPLETE: {check.reason}")
                next_goal = self.goal_system.advance_goal()
                if next_goal:
                    self._log(f"=== NEXT GOAL: {next_goal.name} ===")
                    self._log(f"  {next_goal.description}")
                else:
                    self._log("*** ALL GOALS COMPLETED! ***")
            elif check.progress > goal.progress:
                goal.progress = check.progress
                self._log(f"Progress: {int(check.progress * 100)}% - {check.reason}")
        
        if situation != self.last_situation:
            self._log(f"Situation changed: {self.last_situation} -> {situation}")
            self.last_situation = situation
        
        # Log thinking about what action to take
        self._log_thinking_process(state, situation, goal)
        
        # Write state to JSON file for external monitoring
        if not hasattr(self, '_state_write_counter'):
            self._state_write_counter = 0
        self._state_write_counter += 1
        if self._state_write_counter % 30 == 0:  # Every 30 frames (~0.5 seconds)
            self._write_state_json(state, situation, goal)
        
        # Get action - use goal hints + situation
        action_name = None
        
        # Try LLM with goal context (disable for max FPS)
        # LLM is async but still has overhead - skip in fast mode
        use_llm = self.llm_connected and not getattr(self, 'fast_mode', False)
        if use_llm:
            action_name = self._get_llm_action_with_goal(state, situation, goal)
            if action_name:
                self._log(f"LLM chose: {action_name}")
        
        # Fallback to goal-aware heuristics
        if not action_name:
            action_name = self._get_goal_heuristic_action(situation, goal, state)
            self._log(f"Heuristic chose: {action_name}")
        
        # Execute action through toolkit
        result = self.toolkit.execute_action(action_name)
        self._log(f">> {action_name}: {result.message}")
        
        # Update debug focus based on action and game state
        self._update_focus(action_name, state, situation)
        
        self.current_result = result
        self.hold_remaining = result.hold_frames - 1
        
        return AgentAction(
            buttons_to_press=result.buttons,
            hold_frames=result.hold_frames,
            reasoning=f"{action_name}: {result.message}"
        )
    
    def _update_focus(self, action_name: str, state, situation: str):
        """Update debug focus using actual memory-based positions."""
        menu = state.menu
        cursor = menu.cursor_position
        
        # Get pixel rectangle from memory manager's calculations
        sel_x = menu.selection_pixel_x
        sel_y = menu.selection_pixel_y
        sel_w = menu.selection_pixel_w
        sel_h = menu.selection_pixel_h
        
        # ALWAYS update focus based on current game state, not just action
        # Priority: dialog > menu > battle > other
        
        # Handle name entry screen first
        if situation == "name_entry" or menu.screen_type == "name_entry":
            # Show what character we're targeting
            target_char = ""
            if self.name_entry_active and self.name_char_index < len(self.name_to_enter):
                target_char = self.name_to_enter[self.name_char_index]
            elif self.name_entry_active:
                target_char = "END"
            
            label = f"AI: Name '{self.name_to_enter}'" if self.name_entry_active else "AI: Name entry"
            if target_char:
                label += f" -> {target_char}"
            
            self.current_focus = {
                'type': 'name_entry',
                'action': 'Enter name',
                'target': target_char,
                'rect': (sel_x, sel_y, sel_w, sel_h) if sel_x > 0 else (16, 56, 14, 14),
                'label': label,
            }
        
        elif menu.text_active:
            # Dialog box is active - show it regardless of action
            self.current_focus = {
                'type': 'dialog',
                'action': 'Press A',
                'target': 'Dialog',
                'rect': (sel_x, sel_y, sel_w, sel_h) if sel_x > 0 else (4, 96, 152, 46),  # Fallback if not calculated
                'label': f'AI: Reading dialog',
            }
        
        elif menu.in_menu or situation == "menu":
            # Menu selection - use memory-calculated position
            screen = menu.screen_type
            
            # Determine option name
            if screen == "gender_select":
                opt_name = "BOY" if cursor == 0 else "GIRL"
            elif screen == "option_menu" or menu.options_count == 2:
                opt_name = "YES" if cursor == 0 else "NO"
            else:
                opt_name = f"Option {cursor}"
            
            # Use calculated rect or fallback
            rect = (sel_x, sel_y, sel_w, sel_h) if sel_x > 0 else (88, 40 + cursor * 16, 48, 14)
            
            self.current_focus = {
                'type': 'menu_option',
                'action': 'Select',
                'target': opt_name,
                'rect': rect,
                'label': f'AI: {opt_name}',
            }
        
        elif action_name == "start_game" or situation == "title":
            self.current_focus = {
                'type': 'button',
                'action': 'Press START',
                'target': 'Start',
                'rect': (40, 110, 80, 20),
                'label': 'AI: Start game',
            }
        
        elif action_name in ["move_up", "move_down", "move_left", "move_right", "explore"]:
            direction = action_name.replace("move_", "").upper() if "move_" in action_name else "EXPLORE"
            self.current_focus = {
                'type': 'player',
                'action': f'Move',
                'target': direction,
                'rect': (72, 64, 16, 16),
                'label': f'AI: {direction}',
            }
        
        elif action_name == "interact":
            facing = state.player_position.facing
            offsets = {"up": (0, -16), "down": (0, 16), "left": (-16, 0), "right": (16, 0)}
            dx, dy = offsets.get(facing, (0, 0))
            self.current_focus = {
                'type': 'interact',
                'action': 'Press A',
                'target': facing.upper(),
                'rect': (72 + dx, 64 + dy, 16, 16),
                'label': f'AI: Interact {facing}',
            }
        
        elif situation == "battle":
            # Use calculated battle menu position
            menu_opts = ["FIGHT", "PKMN", "ITEM", "RUN"]
            menu_state = state.battle.menu_state
            opt_name = menu_opts[menu_state] if menu_state < 4 else "?"
            self.current_focus = {
                'type': 'battle_menu',
                'action': 'Select',
                'target': opt_name,
                'rect': (sel_x, sel_y, sel_w, sel_h),
                'label': f'AI: {opt_name}',
            }
        
        else:
            # Default focus - show what we're doing
            self.current_focus = {
                'type': 'none',
                'action': action_name or 'waiting',
                'target': '',
                'rect': (0, 0, 0, 0),  # Empty rect, won't draw
                'label': f'AI: {action_name or "waiting"}',
            }
    
    def get_focus(self) -> dict:
        """Get current debug focus for GUI overlay."""
        return self.current_focus
    
    def _log_perception(self, state, situation: str):
        """Minimal - raw LLM I/O is the main output."""
        # Only log situation changes
        perception_key = f"{situation}_{state.menu.screen_type}"
        if hasattr(self, '_last_perception') and self._last_perception == perception_key:
            return
        self._last_perception = perception_key
        self._log(f"[{situation.upper()}]")
    
    def _log_thinking_process(self, state, situation: str, goal):
        """Minimal logging - main output is raw LLM I/O."""
        # Just log the situation change, LLM prompt/response handles the rest
        pass
    
    def _write_state_json(self, state, situation: str, goal):
        """Write current state to JSON file for monitoring."""
        try:
            import json
            status = {
                "frame": self._state_write_counter,
                "situation": situation,
                "screen_type": state.menu.screen_type,
                "goal": {
                    "id": goal.id if goal else "none",
                    "name": goal.name if goal else "none",
                    "progress": goal.progress if goal else 0
                },
                "player": {
                    "name": state.player_name,
                    "position": (state.player_position.x, state.player_position.y),
                    "map": state.player_position.map_name,
                    "party_count": state.party_count
                },
                "menu": {
                    "in_menu": state.menu.in_menu,
                    "text_active": state.menu.text_active,
                    "cursor": (state.menu.cursor_x, state.menu.cursor_y)
                },
                "last_action": self.current_result.message if self.current_result else "none"
            }
            with open("ai_state.json", "w") as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            pass  # Silently fail if can't write
    
    def get_thinking_output(self) -> List[str]:
        return self.thinking_history.copy()
    
    def get_current_stage_info(self) -> Dict[str, Any]:
        goal = self.goal_system.get_current_goal()
        goal_status = self.goal_system.get_status()
        return {
            'stage': goal.id if goal else "COMPLETE",
            'name': goal.name if goal else "All Goals Done!",
            'goal': goal.description if goal else "Explore freely",
            'progress': f"{goal_status['completed']}/{goal_status['total']} goals",
        }
    
    def get_memory_state(self):
        if self.memory_manager:
            return self.memory_manager.get_state()
        return None
    
    def manual_advance_stage(self):
        """Skip to next goal."""
        self._log("Manual: Skipping goal")
        next_goal = self.goal_system.skip_goal()
        if next_goal:
            self._log(f"=== NEXT GOAL: {next_goal.name} ===")
        else:
            self._log("All goals complete!")
    
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


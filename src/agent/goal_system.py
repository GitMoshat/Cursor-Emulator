"""
Goal System for AI Agent
Provides objectives and tracks progress toward them.
Goals are NOT hardcoded behaviors - they're objectives the AI works toward.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from enum import Enum, auto


class GoalStatus(Enum):
    """Status of a goal."""
    PENDING = auto()      # Not started
    ACTIVE = auto()       # Currently working on
    COMPLETED = auto()    # Done!
    FAILED = auto()       # Couldn't complete
    SKIPPED = auto()      # Manually skipped


@dataclass
class Goal:
    """A single goal/objective."""
    id: str
    name: str
    description: str
    success_hints: List[str]  # What indicates success
    action_hints: List[str]   # Suggested actions to try
    status: GoalStatus = GoalStatus.PENDING
    progress: float = 0.0     # 0.0 to 1.0
    attempts: int = 0
    max_attempts: int = 1000  # Frames before considering stuck
    
    def to_prompt(self) -> str:
        """Format for AI prompt."""
        return f"""
CURRENT GOAL: {self.name}
Description: {self.description}
Progress: {int(self.progress * 100)}%
Hints: {', '.join(self.action_hints[:3])}
Success when: {', '.join(self.success_hints[:2])}
"""


@dataclass
class GoalCheckResult:
    """Result of checking if a goal is complete."""
    is_complete: bool
    progress: float  # 0.0 to 1.0
    reason: str


class GoalSystem:
    """
    Manages goals and tracks progress.
    
    The AI receives the current goal and works toward it.
    Goals are checked against game state to determine completion.
    """
    
    def __init__(self):
        self.goals: List[Goal] = []
        self.current_goal_idx: int = 0
        self.completed_goals: List[str] = []
        
        # Goal completion checkers
        self.checkers: Dict[str, Callable] = {}
        
        # Initialize default Pokemon game goals
        self._init_pokemon_goals()
    
    def _init_pokemon_goals(self):
        """Initialize goals for Pokemon-style games."""
        
        # Goal 1: Start the game
        self.add_goal(Goal(
            id="start_game",
            name="Start the Game",
            description="Get past the title screen and into the game",
            success_hints=[
                "See intro cutscene or professor",
                "Player name entry appears",
                "Game world loads"
            ],
            action_hints=[
                "start_game - Press START at title",
                "interact - Press A to confirm",
                "advance_dialog - Skip any intro text"
            ],
        ))
        
        # Goal 2: Get through intro
        self.add_goal(Goal(
            id="complete_intro",
            name="Complete Introduction", 
            description="Get through the professor's introduction and name entry",
            success_hints=[
                "Player appears in bedroom/house",
                "Can move freely",
                "No more forced dialog"
            ],
            action_hints=[
                "advance_dialog - Keep pressing A through dialog",
                "interact - Select name options",
                "select_option - Confirm choices"
            ],
        ))
        
        # Goal 3: Leave starting area
        self.add_goal(Goal(
            id="leave_house",
            name="Leave Starting Area",
            description="Exit the starting house/room and explore",
            success_hints=[
                "Player is outside",
                "Different map loaded",
                "Can see other buildings/NPCs"
            ],
            action_hints=[
                "move_down - Exit through door (usually down)",
                "explore - Find the exit",
                "interact - Check doors/stairs"
            ],
        ))
        
        # Goal 4: Find the professor
        self.add_goal(Goal(
            id="find_professor",
            name="Find the Professor",
            description="Locate and talk to the Pokemon professor",
            success_hints=[
                "Professor dialog starts",
                "See lab interior",
                "Professor offers Pokemon"
            ],
            action_hints=[
                "explore - Look around town",
                "interact - Talk to NPCs",
                "move_up - Enter buildings (lab)",
                "advance_dialog - Listen to professor"
            ],
        ))
        
        # Goal 5: Get starter Pokemon
        self.add_goal(Goal(
            id="get_starter",
            name="Get Starter Pokemon",
            description="Choose and receive your first Pokemon",
            success_hints=[
                "Pokemon added to party",
                "Party count > 0",
                "Has starter flag set"
            ],
            action_hints=[
                "interact - Examine Pokeballs",
                "select_option - Choose Pokemon",
                "advance_dialog - Confirm selection"
            ],
        ))
    
    def add_goal(self, goal: Goal):
        """Add a goal to the list."""
        self.goals.append(goal)
    
    def get_current_goal(self) -> Optional[Goal]:
        """Get the current active goal."""
        if self.current_goal_idx < len(self.goals):
            return self.goals[self.current_goal_idx]
        return None
    
    def get_goal_prompt(self) -> str:
        """Get current goal formatted for AI prompt."""
        goal = self.get_current_goal()
        if goal:
            return goal.to_prompt()
        return "All goals completed! Explore freely."
    
    def check_goal_completion(self, game_state) -> GoalCheckResult:
        """Check if current goal is complete based on game state."""
        goal = self.get_current_goal()
        if not goal:
            return GoalCheckResult(True, 1.0, "All goals done")
        
        goal.attempts += 1
        
        # Check based on goal ID and game state
        if goal.id == "start_game":
            # Complete when game_started or we see intro
            if game_state.game_started or game_state.player_position.x > 0:
                return GoalCheckResult(True, 1.0, "Game started!")
            # Progress based on attempts
            return GoalCheckResult(False, min(goal.attempts / 100, 0.9), "Trying to start...")
        
        elif goal.id == "complete_intro":
            # Complete when player can move (position changes or in overworld)
            if game_state.player_position.map_id > 0 and not game_state.menu.text_active:
                return GoalCheckResult(True, 1.0, "Intro complete!")
            if game_state.player_name:
                return GoalCheckResult(False, 0.5, f"Named: {game_state.player_name}")
            return GoalCheckResult(False, 0.2, "In intro...")
        
        elif goal.id == "leave_house":
            # Complete when outside (map changes from starting interior)
            pos = game_state.player_position
            # Check if we've moved to a different map
            if pos.map_id not in [0, 1, 2, 33, 34]:  # Starting house maps
                return GoalCheckResult(True, 1.0, f"Now at {pos.map_name}")
            return GoalCheckResult(False, 0.3, f"In {pos.map_name}")
        
        elif goal.id == "find_professor":
            # Complete when in lab or dialog with professor
            pos = game_state.player_position
            map_name = pos.map_name.lower()
            if 'lab' in map_name or 'professor' in map_name:
                return GoalCheckResult(True, 1.0, "Found professor!")
            if game_state.menu.text_active:
                return GoalCheckResult(False, 0.5, "In dialog...")
            return GoalCheckResult(False, 0.2, "Searching...")
        
        elif goal.id == "get_starter":
            # Complete when party has Pokemon
            if game_state.party_count > 0 or game_state.has_starter:
                return GoalCheckResult(True, 1.0, "Got starter Pokemon!")
            return GoalCheckResult(False, 0.3, "Selecting starter...")
        
        # Default
        return GoalCheckResult(False, 0.0, "Working on goal...")
    
    def advance_goal(self) -> Optional[Goal]:
        """Mark current goal complete and move to next."""
        goal = self.get_current_goal()
        if goal:
            goal.status = GoalStatus.COMPLETED
            goal.progress = 1.0
            self.completed_goals.append(goal.id)
            self.current_goal_idx += 1
        return self.get_current_goal()
    
    def skip_goal(self) -> Optional[Goal]:
        """Skip current goal and move to next."""
        goal = self.get_current_goal()
        if goal:
            goal.status = GoalStatus.SKIPPED
            self.current_goal_idx += 1
        return self.get_current_goal()
    
    def reset(self):
        """Reset all goals."""
        self.current_goal_idx = 0
        self.completed_goals.clear()
        for goal in self.goals:
            goal.status = GoalStatus.PENDING
            goal.progress = 0.0
            goal.attempts = 0
    
    def get_status(self) -> Dict[str, Any]:
        """Get goal system status."""
        goal = self.get_current_goal()
        return {
            'current_goal': goal.name if goal else "None",
            'goal_id': goal.id if goal else None,
            'progress': goal.progress if goal else 1.0,
            'completed': len(self.completed_goals),
            'total': len(self.goals),
        }


"""
Game Stage System for Pokemon-like games.
Guides AI through game progression with stage-specific goals and hints.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
import json
import os


class GameStage(Enum):
    """Stages of early game progression."""
    TITLE_SCREEN = auto()      # Press START at title
    INTRO_CUTSCENE = auto()    # Skip intro dialogue
    NAME_ENTRY = auto()        # Enter player name
    STARTER_SELECT = auto()    # Choose starter Pokemon
    RIVAL_BATTLE = auto()      # First rival battle (if any)
    FIRST_ROUTE = auto()       # Navigate to first town
    FIRST_TOWN = auto()        # Arrived at first town!
    COMPLETED = auto()         # Goal reached


@dataclass
class StageHint:
    """Hint for completing a stage."""
    description: str
    buttons: List[str]  # Suggested buttons
    priority: int = 0   # Higher = try first


@dataclass
class StageConfig:
    """Configuration for a game stage."""
    name: str
    goal: str
    hints: List[StageHint] = field(default_factory=list)
    success_indicators: List[str] = field(default_factory=list)  # Text/patterns that indicate success
    timeout_frames: int = 3600  # Max frames before moving on (60 sec at 60fps)
    
    def get_prompt_context(self) -> str:
        """Get context string for LLM prompt."""
        lines = [
            f"Current Stage: {self.name}",
            f"Goal: {self.goal}",
        ]
        if self.hints:
            lines.append("Hints:")
            for hint in sorted(self.hints, key=lambda h: -h.priority):
                lines.append(f"  - {hint.description}")
                if hint.buttons:
                    lines.append(f"    Try: {', '.join(hint.buttons)}")
        return '\n'.join(lines)


# Default stage configurations for Pokemon-like games
POKEMON_STAGES: Dict[GameStage, StageConfig] = {
    GameStage.TITLE_SCREEN: StageConfig(
        name="Title Screen",
        goal="Press START or A to begin the game",
        hints=[
            StageHint("Press START to begin", ["START"], priority=10),
            StageHint("Press A if START doesn't work", ["A"], priority=5),
        ],
        success_indicators=["NEW GAME", "CONTINUE", "new game"],
        timeout_frames=600,
    ),
    
    GameStage.INTRO_CUTSCENE: StageConfig(
        name="Introduction",
        goal="Skip through introduction dialogue by pressing A or B",
        hints=[
            StageHint("Press A to advance dialogue", ["A"], priority=10),
            StageHint("Press B to try skipping", ["B"], priority=5),
            StageHint("Keep pressing A repeatedly", ["A"], priority=8),
        ],
        success_indicators=["your name", "boy", "girl", "name"],
        timeout_frames=3600,
    ),
    
    GameStage.NAME_ENTRY: StageConfig(
        name="Name Entry",
        goal="Enter a name or select default name",
        hints=[
            StageHint("Move to select letters with D-pad", ["UP", "DOWN", "LEFT", "RIGHT"], priority=5),
            StageHint("Press A to select a letter", ["A"], priority=7),
            StageHint("Look for 'END' or default name option", ["A"], priority=10),
            StageHint("Press START to confirm name", ["START"], priority=8),
        ],
        success_indicators=["received", "room", "bed", "mom", "house"],
        timeout_frames=3600,
    ),
    
    GameStage.STARTER_SELECT: StageConfig(
        name="Starter Selection",
        goal="Choose your starter Pokemon from the professor",
        hints=[
            StageHint("Walk to the Pokeballs/Pokemon", ["UP", "DOWN", "LEFT", "RIGHT"], priority=5),
            StageHint("Press A to interact with Pokeball", ["A"], priority=10),
            StageHint("Confirm your choice with A", ["A"], priority=9),
            StageHint("You may need to talk to professor first", ["A"], priority=6),
        ],
        success_indicators=["received", "obtained", "got", "chose", "nickname"],
        timeout_frames=7200,
    ),
    
    GameStage.RIVAL_BATTLE: StageConfig(
        name="Rival Battle",
        goal="Win the first battle against your rival",
        hints=[
            StageHint("Select FIGHT to attack", ["A"], priority=10),
            StageHint("Choose your attack move", ["A"], priority=9),
            StageHint("Keep attacking until victory", ["A"], priority=8),
        ],
        success_indicators=["won", "victory", "defeated", "money"],
        timeout_frames=3600,
    ),
    
    GameStage.FIRST_ROUTE: StageConfig(
        name="First Route",
        goal="Navigate through the first route to reach the next town",
        hints=[
            StageHint("Head towards the exit/next area", ["UP", "DOWN", "LEFT", "RIGHT"], priority=5),
            StageHint("Follow the path, avoid grass if low HP", ["UP", "DOWN"], priority=6),
            StageHint("If in battle, fight or run (DOWN + A)", ["A", "DOWN"], priority=7),
            StageHint("Look for town entrance", ["UP"], priority=8),
        ],
        success_indicators=["town", "city", "center", "mart", "welcome"],
        timeout_frames=14400,  # 4 minutes
    ),
    
    GameStage.FIRST_TOWN: StageConfig(
        name="First Town Reached!",
        goal="You made it! Explore the town.",
        hints=[
            StageHint("Visit the Pokemon Center to heal", ["A"], priority=10),
            StageHint("Explore the town", ["UP", "DOWN", "LEFT", "RIGHT"], priority=5),
        ],
        success_indicators=[],
        timeout_frames=0,  # No timeout - goal reached
    ),
}


@dataclass
class ReinforcementMemory:
    """Memory of what worked and what didn't."""
    successful_actions: Dict[str, List[Dict]] = field(default_factory=dict)
    failed_actions: Dict[str, List[Dict]] = field(default_factory=dict)
    stage_completions: Dict[str, int] = field(default_factory=dict)  # stage -> frame count
    total_attempts: int = 0
    
    def record_success(self, stage: str, action: Dict, context: str = ""):
        """Record a successful action."""
        if stage not in self.successful_actions:
            self.successful_actions[stage] = []
        self.successful_actions[stage].append({
            'action': action,
            'context': context,
            'attempt': self.total_attempts,
        })
    
    def record_failure(self, stage: str, action: Dict, reason: str = ""):
        """Record a failed action."""
        if stage not in self.failed_actions:
            self.failed_actions[stage] = []
        self.failed_actions[stage].append({
            'action': action,
            'reason': reason,
            'attempt': self.total_attempts,
        })
    
    def record_stage_completion(self, stage: str, frames: int):
        """Record completing a stage."""
        self.stage_completions[stage] = frames
    
    def get_advice_for_stage(self, stage: str) -> str:
        """Get advice based on past experience."""
        lines = []
        
        if stage in self.successful_actions:
            successes = self.successful_actions[stage][-5:]  # Last 5
            if successes:
                lines.append("Previously successful actions:")
                for s in successes:
                    lines.append(f"  - {s['action']}")
        
        if stage in self.failed_actions:
            failures = self.failed_actions[stage][-3:]  # Last 3
            if failures:
                lines.append("Actions to avoid:")
                for f in failures:
                    lines.append(f"  - {f['action']}: {f['reason']}")
        
        return '\n'.join(lines) if lines else "No previous experience with this stage."
    
    def save(self, filepath: str):
        """Save memory to file."""
        data = {
            'successful_actions': self.successful_actions,
            'failed_actions': self.failed_actions,
            'stage_completions': self.stage_completions,
            'total_attempts': self.total_attempts,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load memory from file."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.successful_actions = data.get('successful_actions', {})
            self.failed_actions = data.get('failed_actions', {})
            self.stage_completions = data.get('stage_completions', {})
            self.total_attempts = data.get('total_attempts', 0)


class StageManager:
    """
    Manages game stage progression and provides context to AI agents.
    """
    
    def __init__(self, stages: Dict[GameStage, StageConfig] = None):
        self.stages = stages or POKEMON_STAGES
        self.current_stage = GameStage.TITLE_SCREEN
        self.stage_start_frame = 0
        self.frames_in_stage = 0
        
        # Reinforcement memory
        self.memory = ReinforcementMemory()
        self.memory_file = "agent_memory.json"
        
        # Thinking log
        self.thinking_log: List[str] = []
        self.max_thinking_log = 50
        
        # Callbacks
        self.on_stage_change: Optional[Callable[[GameStage, GameStage], None]] = None
        self.on_goal_reached: Optional[Callable[[], None]] = None
    
    def load_memory(self):
        """Load reinforcement memory from file."""
        self.memory.load(self.memory_file)
        self.memory.total_attempts += 1
        self.log_thinking(f"Loaded memory. Attempt #{self.memory.total_attempts}")
    
    def save_memory(self):
        """Save reinforcement memory to file."""
        self.memory.save(self.memory_file)
    
    def log_thinking(self, thought: str):
        """Log a thought/reasoning step."""
        self.thinking_log.append(thought)
        if len(self.thinking_log) > self.max_thinking_log:
            self.thinking_log.pop(0)
    
    def get_current_config(self) -> StageConfig:
        """Get config for current stage."""
        return self.stages.get(self.current_stage, StageConfig(
            name="Unknown", goal="Continue playing"
        ))
    
    def get_prompt_context(self) -> str:
        """Get full context for LLM prompt including stage and memory."""
        config = self.get_current_config()
        
        parts = [
            "=== GAME PROGRESSION ===",
            config.get_prompt_context(),
            "",
            f"Frames in this stage: {self.frames_in_stage}",
            "",
            "=== LEARNED EXPERIENCE ===",
            self.memory.get_advice_for_stage(self.current_stage.name),
        ]
        
        return '\n'.join(parts)
    
    def update(self, frame_num: int) -> bool:
        """
        Update stage tracking. Returns True if stage changed.
        """
        self.frames_in_stage = frame_num - self.stage_start_frame
        
        # Check for timeout (auto-advance)
        config = self.get_current_config()
        if config.timeout_frames > 0 and self.frames_in_stage > config.timeout_frames:
            self.log_thinking(f"Stage '{config.name}' timed out after {self.frames_in_stage} frames")
            return self.advance_stage()
        
        return False
    
    def advance_stage(self) -> bool:
        """Advance to the next stage."""
        old_stage = self.current_stage
        
        # Record completion
        self.memory.record_stage_completion(old_stage.name, self.frames_in_stage)
        
        # Get next stage
        stages = list(GameStage)
        current_idx = stages.index(self.current_stage)
        
        if current_idx < len(stages) - 1:
            self.current_stage = stages[current_idx + 1]
            self.stage_start_frame += self.frames_in_stage
            self.frames_in_stage = 0
            
            config = self.get_current_config()
            self.log_thinking(f"Advanced to stage: {config.name}")
            self.log_thinking(f"Goal: {config.goal}")
            
            if self.on_stage_change:
                self.on_stage_change(old_stage, self.current_stage)
            
            # Check if we reached the goal
            if self.current_stage == GameStage.FIRST_TOWN:
                self.log_thinking("*** GOAL REACHED! Made it to first town! ***")
                if self.on_goal_reached:
                    self.on_goal_reached()
            
            return True
        
        return False
    
    def check_stage_completion(self, screen_text: str = "") -> bool:
        """
        Check if current stage is completed based on indicators.
        This would ideally use OCR on the screen, but for now uses hints.
        """
        config = self.get_current_config()
        
        if not config.success_indicators:
            return False
        
        screen_lower = screen_text.lower()
        for indicator in config.success_indicators:
            if indicator.lower() in screen_lower:
                self.log_thinking(f"Detected success indicator: '{indicator}'")
                return True
        
        return False
    
    def record_action_result(self, action: Dict, success: bool, reason: str = ""):
        """Record the result of an action for reinforcement."""
        stage_name = self.current_stage.name
        
        if success:
            self.memory.record_success(stage_name, action)
            self.log_thinking(f"[OK] Action succeeded: {action}")
        else:
            self.memory.record_failure(stage_name, action, reason)
            self.log_thinking(f"[X] Action failed: {action} - {reason}")
    
    def get_suggested_action(self) -> Optional[StageHint]:
        """Get the top suggested action for current stage."""
        config = self.get_current_config()
        if config.hints:
            return max(config.hints, key=lambda h: h.priority)
        return None
    
    def reset(self):
        """Reset to beginning (but keep memory)."""
        self.current_stage = GameStage.TITLE_SCREEN
        self.stage_start_frame = 0
        self.frames_in_stage = 0
        self.log_thinking("Stage manager reset. Starting from title screen.")


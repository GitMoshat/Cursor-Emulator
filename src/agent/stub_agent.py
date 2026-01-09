"""
Stub AI Agents for GBC Emulator
Placeholder implementations for future AI providers.
"""

import random
from typing import Optional, Dict, Any
from .interface import AgentInterface, AgentAction, GameState, Button, AgentConfig


class StubAgent(AgentInterface):
    """
    Stub agent that does nothing.
    Use as a template for new agent implementations.
    """
    
    def __init__(self):
        super().__init__(name="StubAgent")
    
    def initialize(self, **kwargs) -> bool:
        """Stub always initializes successfully."""
        return True
    
    def decide(self, state: GameState) -> AgentAction:
        """Stub returns no action."""
        return AgentAction(reasoning="Stub agent - no action")


class RandomAgent(AgentInterface):
    """
    Agent that presses random buttons.
    Useful for testing and baseline comparison.
    """
    
    def __init__(self, press_probability: float = 0.3):
        super().__init__(name="RandomAgent")
        self.press_probability = press_probability
        self.current_action: Optional[AgentAction] = None
        self.frames_remaining = 0
    
    def initialize(self, **kwargs) -> bool:
        self.press_probability = kwargs.get('press_probability', self.press_probability)
        return True
    
    def decide(self, state: GameState) -> AgentAction:
        """Press random buttons with some probability."""
        # Continue current action if holding
        if self.frames_remaining > 0:
            self.frames_remaining -= 1
            return self.current_action or AgentAction()
        
        action = AgentAction()
        
        # Random chance to press each button
        all_buttons = [Button.A, Button.B, Button.UP, Button.DOWN, 
                       Button.LEFT, Button.RIGHT, Button.START, Button.SELECT]
        
        for btn in all_buttons:
            if random.random() < self.press_probability:
                action.buttons_to_press.append(btn)
        
        # Random hold duration
        if action.buttons_to_press:
            action.hold_frames = random.randint(1, 15)
            action.reasoning = f"Random: {', '.join(b.name for b in action.buttons_to_press)}"
            
            self.current_action = action
            self.frames_remaining = action.hold_frames - 1
        
        return action


class ScriptedAgent(AgentInterface):
    """
    Agent that follows a predefined script of button presses.
    Useful for automated testing and speedrun practice.
    """
    
    def __init__(self):
        super().__init__(name="ScriptedAgent")
        self.script: list = []
        self.script_index = 0
        self.frames_remaining = 0
        self.current_action: Optional[AgentAction] = None
        self.loop = False
    
    def initialize(self, **kwargs) -> bool:
        """
        Initialize with a script.
        
        Script format: List of (buttons, hold_frames) tuples
        Example: [(['A'], 1), (['RIGHT'], 30), (['A', 'B'], 5)]
        """
        self.script = kwargs.get('script', [])
        self.loop = kwargs.get('loop', False)
        self.script_index = 0
        return True
    
    def load_script(self, script: list, loop: bool = False):
        """Load a new script."""
        self.script = script
        self.loop = loop
        self.script_index = 0
        self.frames_remaining = 0
    
    def decide(self, state: GameState) -> AgentAction:
        """Execute next action from script."""
        # Continue current action if holding
        if self.frames_remaining > 0:
            self.frames_remaining -= 1
            return self.current_action or AgentAction()
        
        # Check if script is done
        if self.script_index >= len(self.script):
            if self.loop:
                self.script_index = 0
            else:
                return AgentAction(reasoning="Script complete")
        
        if self.script_index < len(self.script):
            entry = self.script[self.script_index]
            self.script_index += 1
            
            buttons = []
            for btn_name in entry[0]:
                try:
                    buttons.append(Button[btn_name.upper()])
                except KeyError:
                    pass
            
            hold = entry[1] if len(entry) > 1 else 1
            
            action = AgentAction(
                buttons_to_press=buttons,
                hold_frames=hold,
                reasoning=f"Script step {self.script_index}"
            )
            
            self.current_action = action
            self.frames_remaining = hold - 1
            
            return action
        
        return AgentAction()


class OpenAIAgent(AgentInterface):
    """
    Stub for OpenAI GPT-4 Vision agent.
    TODO: Implement when needed.
    """
    
    def __init__(self):
        super().__init__(name="OpenAIAgent")
        self.api_key: Optional[str] = None
    
    def initialize(self, **kwargs) -> bool:
        self.api_key = kwargs.get('api_key')
        if not self.api_key:
            print("OpenAIAgent: API key required")
            return False
        # TODO: Validate API key
        print("OpenAIAgent: Stub - not implemented")
        return False
    
    def decide(self, state: GameState) -> AgentAction:
        return AgentAction(reasoning="OpenAI agent not implemented")


class AnthropicAgent(AgentInterface):
    """
    Stub for Anthropic Claude Vision agent.
    TODO: Implement when needed.
    """
    
    def __init__(self):
        super().__init__(name="AnthropicAgent")
        self.api_key: Optional[str] = None
    
    def initialize(self, **kwargs) -> bool:
        self.api_key = kwargs.get('api_key')
        if not self.api_key:
            print("AnthropicAgent: API key required")
            return False
        print("AnthropicAgent: Stub - not implemented")
        return False
    
    def decide(self, state: GameState) -> AgentAction:
        return AgentAction(reasoning="Anthropic agent not implemented")


class ReinforcementLearningAgent(AgentInterface):
    """
    Stub for RL-based agent.
    TODO: Implement with stable-baselines3 or similar.
    """
    
    def __init__(self):
        super().__init__(name="RLAgent")
        self.model = None
    
    def initialize(self, **kwargs) -> bool:
        model_path = kwargs.get('model_path')
        if model_path:
            # TODO: Load trained model
            print(f"RLAgent: Would load model from {model_path}")
        print("RLAgent: Stub - not implemented")
        return False
    
    def decide(self, state: GameState) -> AgentAction:
        return AgentAction(reasoning="RL agent not implemented")




"""
Guided AI Agent with Stage-Based Progression
Uses game stages and reinforcement memory to play through early game.
"""

import json
import requests
import time
from typing import Optional, Dict, Any, List
import numpy as np

from .interface import AgentInterface, AgentAction, GameState, Button, AgentConfig
from .game_stages import StageManager, GameStage, POKEMON_STAGES


class GuidedOllamaAgent(AgentInterface):
    """
    Ollama-based agent guided by stage system with reinforcement learning.
    
    Features:
    - Stage-aware prompting
    - Reinforcement memory (learns from past attempts)
    - Detailed thinking/reasoning output
    - Fallback to stage hints when LLM fails
    """
    
    SYSTEM_PROMPT = """You are an AI playing a Pokemon-style Game Boy Color game. Your goal is to progress through the early game: start the game, choose a starter Pokemon, and reach the first town.

You can see the current game screen. Based on what you see and the current stage goal, decide what buttons to press.

Available buttons: A, B, START, SELECT, UP, DOWN, LEFT, RIGHT

IMPORTANT RULES:
1. Focus on the CURRENT GOAL - don't get distracted
2. If you see a dialog box, press A to advance it
3. If you're stuck, try different buttons
4. Remember what worked before

Respond with JSON only:
{
    "buttons": ["BUTTON"],
    "hold_frames": 1-30,
    "reasoning": "What I see and why I'm doing this",
    "confidence": 0.0-1.0
}"""

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(name="GuidedOllama")
        self.config = config or AgentConfig()
        self.frame_skip = 10  # Slower decisions for thoughtful play
        
        # Ollama settings
        self.host = self.config.ollama_host
        self.model = self.config.ollama_model
        self.timeout = self.config.ollama_timeout
        self.connected = False
        
        # Stage manager
        self.stage_manager = StageManager(POKEMON_STAGES)
        
        # Thinking output
        self.current_thinking = ""
        self.thinking_history: List[str] = []
        self.max_thinking_history = 100
        
        # State tracking
        self.last_response = ""
        self.last_error = ""
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        
        # Action tracking for reinforcement
        self.pending_action: Optional[Dict] = None
        self.action_start_frame = 0
    
    def initialize(self, **kwargs) -> bool:
        """Initialize agent and load memory."""
        self.host = kwargs.get('host', self.host)
        self.model = kwargs.get('model', self.model)
        
        # Try to connect to Ollama
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                self.connected = True
                self._log(f"Connected to Ollama at {self.host}")
                self._log(f"Using model: {self.model}")
            else:
                self._log(f"Ollama returned status {response.status_code}")
                self.connected = False
        except Exception as e:
            self._log(f"Failed to connect to Ollama: {e}")
            self.connected = False
        
        # Load reinforcement memory
        try:
            self.stage_manager.load_memory()
            self._log(f"Loaded memory from previous {self.stage_manager.memory.total_attempts - 1} attempts")
        except Exception as e:
            self._log(f"No previous memory found: {e}")
        
        # Log initial stage
        config = self.stage_manager.get_current_config()
        self._log(f"Starting at: {config.name}")
        self._log(f"Goal: {config.goal}")
        
        return True  # Always return True - we can fall back to hints
    
    def _log(self, message: str):
        """Log a message to thinking history."""
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.thinking_history.append(entry)
        self.current_thinking = message
        
        if len(self.thinking_history) > self.max_thinking_history:
            self.thinking_history.pop(0)
        
        # Also log to stage manager
        self.stage_manager.log_thinking(message)
        
        # Print for debugging
        print(f"[GuidedAgent] {message}")
    
    def decide(self, state: GameState) -> AgentAction:
        """Make a decision based on game state and current stage."""
        
        # Update stage manager
        stage_changed = self.stage_manager.update(state.frame_number)
        if stage_changed:
            self.consecutive_failures = 0
        
        # Get stage context
        stage_config = self.stage_manager.get_current_config()
        stage_context = self.stage_manager.get_prompt_context()
        
        # Try to get LLM decision
        action = None
        if self.connected:
            action = self._get_llm_decision(state, stage_context)
        
        # Fall back to stage hints if LLM fails
        if not action or not action.buttons_to_press:
            self.consecutive_failures += 1
            
            if self.consecutive_failures >= self.max_consecutive_failures:
                self._log("Too many failures, using hint-based action")
                action = self._get_hint_action(stage_config)
                self.consecutive_failures = 0
            else:
                # Try basic A press for dialog
                action = AgentAction(
                    buttons_to_press=[Button.A],
                    hold_frames=3,
                    reasoning="Fallback: pressing A for dialog"
                )
        else:
            self.consecutive_failures = 0
        
        # Track action for reinforcement
        self.pending_action = {
            'buttons': [b.name for b in action.buttons_to_press],
            'reasoning': action.reasoning,
            'frame': state.frame_number,
        }
        self.action_start_frame = state.frame_number
        
        return action
    
    def _get_llm_decision(self, state: GameState, stage_context: str) -> Optional[AgentAction]:
        """Get decision from Ollama LLM."""
        try:
            # Build prompt
            prompt = f"""{stage_context}

Frame: {state.frame_number}
Sprites visible: {sum(1 for s in state.sprites if s.visible)}
Scroll: ({state.scroll_x}, {state.scroll_y})

What do you see on screen and what button should I press?
Respond with JSON only."""

            # Get frame as base64
            frame_b64 = state.get_frame_base64('PNG')
            
            # Make request
            payload = {
                "model": self.model,
                "prompt": prompt,
                "system": self.SYSTEM_PROMPT,
                "images": [frame_b64],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 300,
                }
            }
            
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                self.last_response = result.get('response', '')
                return self._parse_response(self.last_response)
            else:
                self.last_error = f"Status {response.status_code}"
                return None
                
        except requests.exceptions.Timeout:
            self.last_error = "Request timed out"
            self._log("LLM request timed out")
            return None
        except Exception as e:
            self.last_error = str(e)
            self._log(f"LLM error: {e}")
            return None
    
    def _parse_response(self, response: str) -> Optional[AgentAction]:
        """Parse LLM response to action."""
        try:
            # Extract JSON
            response = response.strip()
            
            # Handle markdown code blocks
            if '```json' in response:
                start = response.find('```json') + 7
                end = response.find('```', start)
                response = response[start:end].strip()
            elif '```' in response:
                start = response.find('```') + 3
                end = response.find('```', start)
                response = response[start:end].strip()
            
            # Find JSON object
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                
                # Parse buttons
                buttons = []
                for btn_name in data.get('buttons', []):
                    try:
                        buttons.append(Button[btn_name.upper()])
                    except KeyError:
                        pass
                
                reasoning = data.get('reasoning', '')
                hold_frames = min(max(data.get('hold_frames', 5), 1), 30)
                confidence = data.get('confidence', 0.5)
                
                if reasoning:
                    self._log(f"AI thinks: {reasoning}")
                
                return AgentAction(
                    buttons_to_press=buttons,
                    hold_frames=hold_frames,
                    reasoning=reasoning,
                    confidence=confidence
                )
            
            return None
            
        except json.JSONDecodeError as e:
            self._log(f"Failed to parse JSON: {e}")
            return None
    
    def _get_hint_action(self, stage_config) -> AgentAction:
        """Get action from stage hints when LLM fails."""
        hint = self.stage_manager.get_suggested_action()
        
        if hint:
            buttons = []
            for btn_name in hint.buttons:
                try:
                    buttons.append(Button[btn_name.upper()])
                except KeyError:
                    pass
            
            self._log(f"Using hint: {hint.description}")
            
            return AgentAction(
                buttons_to_press=buttons,
                hold_frames=10,
                reasoning=f"Hint: {hint.description}"
            )
        
        # Ultimate fallback
        return AgentAction(
            buttons_to_press=[Button.A],
            hold_frames=5,
            reasoning="Default: press A"
        )
    
    def manual_advance_stage(self):
        """Manually advance to next stage (for testing)."""
        self.stage_manager.advance_stage()
    
    def get_thinking_output(self) -> List[str]:
        """Get the thinking history for display."""
        return self.thinking_history.copy()
    
    def get_current_stage_info(self) -> Dict[str, Any]:
        """Get current stage information."""
        config = self.stage_manager.get_current_config()
        return {
            'stage': self.stage_manager.current_stage.name,
            'name': config.name,
            'goal': config.goal,
            'frames_in_stage': self.stage_manager.frames_in_stage,
            'timeout': config.timeout_frames,
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed agent status."""
        status = super().get_status()
        status.update({
            'connected': self.connected,
            'model': self.model,
            'current_stage': self.stage_manager.current_stage.name,
            'stage_goal': self.stage_manager.get_current_config().goal,
            'frames_in_stage': self.stage_manager.frames_in_stage,
            'consecutive_failures': self.consecutive_failures,
            'memory_attempts': self.stage_manager.memory.total_attempts,
            'current_thinking': self.current_thinking,
        })
        return status
    
    def shutdown(self):
        """Save memory and cleanup."""
        super().shutdown()
        try:
            self.stage_manager.save_memory()
            self._log("Memory saved")
        except Exception as e:
            self._log(f"Failed to save memory: {e}")


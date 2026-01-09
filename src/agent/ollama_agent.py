"""
Ollama-based AI Agent for GBC Emulator
Uses local LLM with vision capabilities to play games.
"""

import json
import requests
from typing import Optional, Dict, Any, List
import numpy as np

from .interface import AgentInterface, AgentAction, GameState, Button, AgentConfig


class OllamaAgent(AgentInterface):
    """
    AI Agent powered by Ollama local LLM.
    
    Uses vision models (like LLaVA) to understand the game screen
    and make decisions about what buttons to press.
    """
    
    # System prompt for the game-playing AI
    SYSTEM_PROMPT = """You are an AI playing a Game Boy Color game. You can see the current game screen and must decide what buttons to press.

Available buttons: A, B, START, SELECT, UP, DOWN, LEFT, RIGHT

Your goal is to play the game effectively:
- Navigate menus and dialogs
- Move the character around
- Interact with objects and NPCs
- Battle enemies (if applicable)
- Progress through the game

Respond with a JSON object containing:
{
    "buttons": ["BUTTON_NAME", ...],  // Buttons to press this frame
    "hold_frames": 1-30,              // How many frames to hold (1 for tap, more for movement)
    "reasoning": "Brief explanation"   // Why you chose this action
}

Examples:
- To walk right: {"buttons": ["RIGHT"], "hold_frames": 10, "reasoning": "Moving right to explore"}
- To select menu: {"buttons": ["A"], "hold_frames": 1, "reasoning": "Confirming menu selection"}
- To open menu: {"buttons": ["START"], "hold_frames": 1, "reasoning": "Opening game menu"}
- To run right: {"buttons": ["RIGHT", "B"], "hold_frames": 15, "reasoning": "Running right quickly"}

Analyze the screen carefully and make smart decisions!"""

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(name="OllamaAgent")
        self.config = config or AgentConfig()
        self.frame_skip = self.config.frame_skip
        
        self.host = self.config.ollama_host
        self.model = self.config.ollama_model
        self.timeout = self.config.ollama_timeout
        
        self.connected = False
        self.last_response: Optional[str] = None
        self.last_error: Optional[str] = None
        
        # Context for multi-turn conversation
        self.conversation_history: List[Dict] = []
        self.max_history = 5  # Keep last N exchanges
        
        # Game-specific context (can be set by user)
        self.game_context = ""
    
    def initialize(self, **kwargs) -> bool:
        """Connect to Ollama and verify it's running."""
        self.host = kwargs.get('host', self.host)
        self.model = kwargs.get('model', self.model)
        self.game_context = kwargs.get('game_context', '')
        
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '').split(':')[0] for m in models]
                
                if self.model.split(':')[0] in model_names or any(self.model in n for n in model_names):
                    self.connected = True
                    print(f"OllamaAgent connected to {self.host} using model {self.model}")
                    return True
                else:
                    self.last_error = f"Model '{self.model}' not found. Available: {model_names}"
                    print(f"OllamaAgent: {self.last_error}")
                    return False
            else:
                self.last_error = f"Ollama returned status {response.status_code}"
                return False
                
        except requests.exceptions.ConnectionError:
            self.last_error = f"Cannot connect to Ollama at {self.host}. Is it running?"
            print(f"OllamaAgent: {self.last_error}")
            return False
        except Exception as e:
            self.last_error = str(e)
            return False
    
    def set_game_context(self, context: str):
        """Set game-specific context to help the AI understand the game."""
        self.game_context = context
    
    def _build_prompt(self, state: GameState) -> str:
        """Build the prompt for the current game state."""
        parts = []
        
        # Add game context if provided
        if self.game_context:
            parts.append(f"Game: {self.game_context}")
        
        # Add current state info
        parts.append(f"Frame: {state.frame_number}")
        parts.append(f"Scroll position: ({state.scroll_x}, {state.scroll_y})")
        
        # Add visible sprites info
        visible_sprites = [s for s in state.sprites if s.visible]
        if visible_sprites:
            parts.append(f"Visible sprites: {len(visible_sprites)}")
            for s in visible_sprites[:10]:  # Limit to 10
                parts.append(f"  - Sprite at ({s.x}, {s.y})")
        
        # Add current buttons
        if state.buttons_pressed:
            parts.append(f"Currently pressed: {', '.join(b.name for b in state.buttons_pressed)}")
        
        # Add recent action history
        if self.action_history:
            recent = self.action_history[-3:]
            parts.append("Recent actions:")
            for action in recent:
                btns = ', '.join(b.name for b in action.buttons_to_press) or 'none'
                parts.append(f"  - {btns}: {action.reasoning[:50]}")
        
        parts.append("\nWhat buttons should I press next? Respond with JSON only.")
        
        return '\n'.join(parts)
    
    def _call_ollama(self, state: GameState) -> Optional[Dict]:
        """Make API call to Ollama with the game screenshot."""
        if not self.connected:
            return None
        
        try:
            # Get frame as base64
            frame_b64 = state.get_frame_base64('PNG')
            
            # Build prompt
            prompt = self._build_prompt(state)
            
            # Prepare request
            payload = {
                "model": self.model,
                "prompt": prompt,
                "system": self.SYSTEM_PROMPT,
                "images": [frame_b64],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 200,
                }
            }
            
            # Make request
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
                self.last_error = f"Ollama returned {response.status_code}"
                return None
                
        except requests.exceptions.Timeout:
            self.last_error = "Ollama request timed out"
            return None
        except Exception as e:
            self.last_error = str(e)
            return None
    
    def _parse_response(self, response: str) -> Optional[Dict]:
        """Parse the JSON response from the LLM."""
        try:
            # Try to extract JSON from the response
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
                return json.loads(json_str)
            
            return None
            
        except json.JSONDecodeError:
            self.last_error = f"Failed to parse JSON: {response[:100]}"
            return None
    
    def decide(self, state: GameState) -> AgentAction:
        """Use Ollama to decide what action to take."""
        action = AgentAction()
        
        # Call Ollama
        result = self._call_ollama(state)
        
        if result:
            # Parse buttons
            buttons = result.get('buttons', [])
            for btn_name in buttons:
                try:
                    btn = Button[btn_name.upper()]
                    action.buttons_to_press.append(btn)
                except (KeyError, AttributeError):
                    pass
            
            # Parse hold frames
            hold = result.get('hold_frames', 1)
            action.hold_frames = max(1, min(hold, self.config.max_hold_frames))
            
            # Parse reasoning
            action.reasoning = result.get('reasoning', '')
            
            if self.config.log_reasoning and action.reasoning:
                print(f"[OllamaAgent] {action.reasoning}")
        else:
            # Default action on failure - do nothing
            action.reasoning = f"No response: {self.last_error}"
        
        return action
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed agent status."""
        status = super().get_status()
        status.update({
            'connected': self.connected,
            'host': self.host,
            'model': self.model,
            'last_error': self.last_error,
            'last_response_preview': self.last_response[:100] if self.last_response else None,
        })
        return status
    
    def shutdown(self):
        """Clean up."""
        super().shutdown()
        self.connected = False
        self.conversation_history.clear()


class OllamaAgentSimple(AgentInterface):
    """
    Simplified Ollama agent that uses text-only reasoning.
    Faster but less accurate than vision-based agent.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(name="OllamaSimple")
        self.config = config or AgentConfig()
        self.host = self.config.ollama_host
        self.model = "llama3.2"  # Text-only model
        self.connected = False
    
    def initialize(self, **kwargs) -> bool:
        self.host = kwargs.get('host', self.host)
        self.model = kwargs.get('model', self.model)
        
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            self.connected = response.status_code == 200
            return self.connected
        except:
            return False
    
    def decide(self, state: GameState) -> AgentAction:
        """Make decision based on game state description only."""
        action = AgentAction()
        
        # Build text description
        desc = state.get_frame_description()
        
        prompt = f"""Game state:
{desc}

Based on this, what button should be pressed? Choose from: A, B, UP, DOWN, LEFT, RIGHT, START, SELECT
Respond with just the button name."""

        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 20}
                },
                timeout=10
            )
            
            if response.status_code == 200:
                text = response.json().get('response', '').upper().strip()
                for btn in Button:
                    if btn.name in text:
                        action.buttons_to_press.append(btn)
                        break
                        
        except Exception as e:
            action.reasoning = str(e)
        
        return action




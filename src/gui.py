"""
Game Boy Color Emulator GUI
Full-featured interface with game display, tilemap viewer, debug tools, and AI agent controls.
"""

import pygame
import numpy as np
from typing import Optional
import sys

# Try to import agent system
try:
    from .agent import AgentManager, AgentConfig
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False


class EmulatorGUI:
    """
    Pygame-based GUI for the GBC emulator.
    
    Features:
    - Main game display (scaled)
    - Tilemap viewer
    - Tile viewer
    - Debug information panel
    - CPU/PPU state display
    """
    
    # Colors
    BG_COLOR = (18, 20, 28)
    PANEL_BG = (28, 32, 42)
    TEXT_COLOR = (200, 210, 220)
    HIGHLIGHT_COLOR = (80, 140, 200)
    BORDER_COLOR = (50, 55, 65)
    
    def __init__(self, emulator, scale: int = 3):
        self.emulator = emulator
        self.scale = scale
        
        # Window dimensions
        self.game_width = 160 * scale
        self.game_height = 144 * scale
        
        # Debug panel dimensions
        self.debug_panel_width = 420
        self.tilemap_size = 256  # Tilemap is 256x256
        self.tiles_width = 128
        self.tiles_height = 192
        
        # Total window size - wider for AI panel on right
        self.ai_panel_width = 350  # AI thinking panel width
        self.window_width = self.game_width + self.ai_panel_width + 40
        self.window_height = max(self.game_height + 100, 650)
        
        # Initialize Pygame
        pygame.init()
        ppu_ver = getattr(emulator, 'PPU_VERSION', 'Unknown')
        pygame.display.set_caption(f"GBC Emulator [{ppu_ver} PPU] - {emulator.rom_title or 'No ROM'}")
        
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.clock = pygame.time.Clock()
        
        # Fonts
        pygame.font.init()
        try:
            self.font = pygame.font.SysFont('Consolas', 14)
            self.font_small = pygame.font.SysFont('Consolas', 12)
            self.font_title = pygame.font.SysFont('Consolas', 16, bold=True)
        except:
            self.font = pygame.font.Font(None, 16)
            self.font_small = pygame.font.Font(None, 14)
            self.font_title = pygame.font.Font(None, 18)
        
        # Surfaces
        self.game_surface = pygame.Surface((160, 144))
        self.tilemap_surface = pygame.Surface((256, 256))
        self.tiles_surface = pygame.Surface((128, 192))
        
        # Frame skip for performance
        self.frame_skip = 2  # 0 = no skip, 1 = skip every other, 2 = skip 2 of 3
        self.frame_counter = 0
        self.turbo_mode = True  # Default ON for faster gameplay
        
        # AI Agent system
        self.agent_manager = None
        self.agent_enabled = False
        self.agent_status_text = "AI: Starting..."
        self.agent_thinking_log: list = ["[Init] AI Agent initializing..."]
        self.max_thinking_display = 12  # Lines to show in panel
        self.show_ai_panel = True  # Always show AI panel by default
        
        # Attempt tracking
        self.current_attempt = 1
        self.max_attempts = 3
        self.attempt_results: list = []  # Track success/failure per attempt
        
        if AGENT_AVAILABLE:
            self.agent_manager = AgentManager(emulator)
        
        # UI Buttons
        self.buttons: dict = {}
        self._init_buttons()
        
        # Auto-start AI agent
        self._auto_start_ai = True
        
        # State
        self.running = True
    
    def _init_buttons(self):
        """Initialize UI button positions."""
        # Button dimensions
        btn_w, btn_h = 70, 24
        btn_y = self.game_height + 15  # Below game display
        btn_x = 10
        spacing = 5
        
        # Create button rectangles
        self.buttons = {
            'turbo': pygame.Rect(btn_x, btn_y, btn_w, btn_h),
            'ai': pygame.Rect(btn_x + btn_w + spacing, btn_y, btn_w, btn_h),
            'ai_panel': pygame.Rect(btn_x + (btn_w + spacing) * 2, btn_y, btn_w + 10, btn_h),
            'reset': pygame.Rect(btn_x + (btn_w + spacing) * 3 + 10, btn_y, btn_w - 10, btn_h),
        }
        self.show_tilemap = True
        self.show_tiles = True
        self.show_debug = True
        self.selected_tilemap = 0  # 0 or 1
        self.selected_vram_bank = 0
        
        # Key mapping
        self.key_map = {
            pygame.K_z: 'a',
            pygame.K_x: 'b',
            pygame.K_RETURN: 'start',
            pygame.K_RSHIFT: 'select',
            pygame.K_UP: 'up',
            pygame.K_DOWN: 'down',
            pygame.K_LEFT: 'left',
            pygame.K_RIGHT: 'right',
        }
        
        # FPS tracking
        self.fps_samples = []
        self.last_fps = 0
    
    def run(self):
        """Main GUI loop."""
        self.running = True
        
        # Auto-start AI on first frame
        if self._auto_start_ai and self.agent_manager:
            self._toggle_agent()
            self.agent_thinking_log.append(f"[Attempt {self.current_attempt}/{self.max_attempts}] Starting...")
        
        while self.running:
            self._handle_events()
            
            if not self.emulator.paused:
                # Run frame(s) with optional frame skip
                frames_to_run = 1 + self.frame_skip if self.turbo_mode else 1
                
                # In turbo mode with AI, run more frames but process AI less frequently
                if self.turbo_mode and self.agent_enabled:
                    frames_to_run = 4  # Run 4 frames per loop iteration in turbo
                
                for i in range(frames_to_run):
                    frame = self.emulator.run_frame()
                    
                    # Let AI agent process frame (in turbo, only process every Nth frame)
                    if self.agent_manager and self.agent_enabled:
                        # In turbo mode, process AI on first frame of batch only
                        # This prevents AI from getting overwhelmed
                        should_process_ai = (not self.turbo_mode) or (i == 0)
                        
                        if should_process_ai:
                            action = self.agent_manager.process_frame(frame, turbo=self.turbo_mode)
                            if action and action.reasoning:
                                turbo_str = " [TURBO]" if self.turbo_mode else ""
                                self.agent_status_text = f"AI{turbo_str}: {action.reasoning[:35]}"
                        
                        # Update thinking log from agent
                        if self.agent_manager.agent:
                            if hasattr(self.agent_manager.agent, 'get_thinking_output'):
                                self.agent_thinking_log = self.agent_manager.agent.get_thinking_output()
                            elif hasattr(self.agent_manager.agent, 'thinking_history'):
                                self.agent_thinking_log = self.agent_manager.agent.thinking_history
                            
                            # Check for goal completion (find_professor is target)
                            if hasattr(self.agent_manager.agent, 'goal_system'):
                                goal = self.agent_manager.agent.goal_system.get_current_goal()
                                if goal and goal.id == "get_starter":
                                    # Reached final goal - attempt successful!
                                    self._handle_attempt_complete(True)
                
                # Only update display on last frame
                self._update_game_surface(frame)
            
            self._draw()
            
            # FPS limiting (skip in turbo mode)
            if not self.turbo_mode:
                self.clock.tick(60)
            else:
                self.clock.tick(0)  # Unlimited
            
            # Track FPS
            self.fps_samples.append(self.clock.get_fps())
            if len(self.fps_samples) > 30:
                self.fps_samples.pop(0)
                self.last_fps = sum(self.fps_samples) / len(self.fps_samples)
        
        # Cleanup
        if self.agent_manager:
            self.agent_manager.stop()
        pygame.quit()
    
    def _toggle_agent(self):
        """Toggle AI agent on/off."""
        if not self.agent_manager:
            self.agent_status_text = "AI: Not available"
            return
        
        if self.agent_enabled:
            self.agent_manager.stop()
            self.agent_enabled = False
            self.agent_status_text = "AI: Stopped"
            self.agent_thinking_log.append("[Stopped] Agent disabled")
        else:
            if not self.agent_manager.agent:
                # Create default agent (Toolkit-based - discovers and requests actions)
                agent = self.agent_manager.create_agent('toolkit')
                if agent:
                    self.agent_manager.set_agent(agent)
                    self.agent_thinking_log.append("[Init] Created ToolkitAgent (action-based AI)")
            
            if self.agent_manager.start():
                self.agent_enabled = True
                self.agent_status_text = f"AI: {self.agent_manager.agent.name} running"
                self.agent_thinking_log.append(f"[Started] {self.agent_manager.agent.name}")
            else:
                self.agent_status_text = "AI: Failed to start (check Ollama)"
                self.agent_thinking_log.append("[Error] Failed to start - is Ollama running?")
    
    def _cycle_agent(self):
        """Cycle through available agent types."""
        if not self.agent_manager:
            return
        
        # Stop current agent
        if self.agent_enabled:
            self.agent_manager.stop()
            self.agent_enabled = False
        
        agents = self.agent_manager.get_available_agents()
        current_name = self.agent_manager.agent.name if self.agent_manager.agent else None
        
        # Find current index and get next
        try:
            current_idx = next(i for i, a in enumerate(agents) 
                             if a in (current_name or '').lower())
            next_idx = (current_idx + 1) % len(agents)
        except StopIteration:
            next_idx = 0
        
        agent_type = agents[next_idx]
        agent = self.agent_manager.create_agent(agent_type)
        if agent:
            self.agent_manager.set_agent(agent)
            self.agent_status_text = f"AI: {agent.name} selected (F2 to start)"
            self.agent_thinking_log.append(f"[Switch] Changed to {agent.name}")
    
    def _advance_agent_stage(self):
        """Manually advance the AI agent's stage (for testing)."""
        if not self.agent_manager or not self.agent_manager.agent:
            return
        
        if hasattr(self.agent_manager.agent, 'manual_advance_stage'):
            self.agent_manager.agent.manual_advance_stage()
            self.agent_thinking_log.append("[Manual] Stage advanced by user")
    
    def _handle_button_click(self, pos):
        """Handle mouse clicks on UI buttons."""
        for name, rect in self.buttons.items():
            if rect.collidepoint(pos):
                if name == 'turbo':
                    self.turbo_mode = not self.turbo_mode
                    self.frame_skip = 2 if self.turbo_mode else 0
                    mode_str = "TURBO ON ⚡" if self.turbo_mode else "Normal speed"
                    self.agent_thinking_log.append(f"[Speed] {mode_str}")
                
                elif name == 'ai':
                    self._toggle_agent()
                
                elif name == 'ai_panel':
                    self.show_ai_panel = not self.show_ai_panel
                
                elif name == 'reset':
                    self._start_new_attempt()
                
                break
    
    def _handle_attempt_complete(self, success: bool):
        """Handle completion of an attempt."""
        self.attempt_results.append(success)
        
        if success:
            self.agent_thinking_log.append(f"🎉 ATTEMPT {self.current_attempt} SUCCESSFUL!")
        else:
            self.agent_thinking_log.append(f"❌ Attempt {self.current_attempt} failed")
        
        # Check if more attempts remain
        if self.current_attempt < self.max_attempts:
            self.current_attempt += 1
            self._start_new_attempt()
        else:
            # All attempts done
            successes = sum(self.attempt_results)
            self.agent_thinking_log.append(f"=== RESULTS: {successes}/{self.max_attempts} successful ===")
            self.agent_enabled = False
    
    def _start_new_attempt(self):
        """Start a new attempt - reset game and agent."""
        self.emulator.reset()
        
        # Reset agent goals
        if self.agent_manager and self.agent_manager.agent:
            if hasattr(self.agent_manager.agent, 'goal_system'):
                self.agent_manager.agent.goal_system.reset()
            self.agent_manager.reset()
        
        self.agent_thinking_log.append(f"[Attempt {self.current_attempt}/{self.max_attempts}] Starting fresh...")
    
    def _handle_events(self):
        """Handle input events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self.emulator.running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self._handle_button_click(event.pos)
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    self.emulator.running = False
                
                elif event.key == pygame.K_SPACE:
                    self.emulator.paused = not self.emulator.paused
                
                elif event.key == pygame.K_r:
                    self.emulator.reset()
                
                elif event.key == pygame.K_t:
                    self.selected_tilemap = 1 - self.selected_tilemap
                
                elif event.key == pygame.K_v:
                    self.selected_vram_bank = 1 - self.selected_vram_bank
                
                elif event.key == pygame.K_F1:
                    self.show_debug = not self.show_debug
                
                elif event.key == pygame.K_TAB:
                    self.turbo_mode = not self.turbo_mode
                    self.frame_skip = 2 if self.turbo_mode else 0
                    if self.agent_enabled:
                        mode_str = "TURBO ON ⚡" if self.turbo_mode else "Normal speed"
                        self.agent_thinking_log.append(f"[Speed] {mode_str}")
                
                elif event.key == pygame.K_F2:
                    # Toggle AI agent
                    self._toggle_agent()
                
                elif event.key == pygame.K_F3:
                    # Cycle through agent types
                    self._cycle_agent()
                
                elif event.key == pygame.K_F4:
                    # Manually advance AI stage (for testing)
                    self._advance_agent_stage()
                
                elif event.key == pygame.K_n and self.emulator.paused:
                    # Step one instruction
                    self.emulator.step()
                
                elif event.key in self.key_map:
                    self.emulator.press_button(self.key_map[event.key])
            
            elif event.type == pygame.KEYUP:
                if event.key in self.key_map:
                    self.emulator.release_button(self.key_map[event.key])
    
    def _update_game_surface(self, frame: np.ndarray):
        """Update the game display surface."""
        # Convert numpy array to pygame surface
        pygame.surfarray.blit_array(self.game_surface, frame.swapaxes(0, 1))
    
    def _draw(self):
        """Draw all GUI elements."""
        self.screen.fill(self.BG_COLOR)
        
        # Draw game display (left side)
        self._draw_game_display()
        
        # Draw UI buttons (below game)
        self._draw_buttons()
        
        # Draw AI panel on the RIGHT side (goal + thinking)
        if self.show_ai_panel:
            self._draw_ai_panel_right()
        
        # Draw help at bottom
        self._draw_help()
        
        pygame.display.flip()
    
    def _draw_buttons(self):
        """Draw clickable UI buttons."""
        for name, rect in self.buttons.items():
            # Determine button state/color
            if name == 'turbo':
                active = self.turbo_mode
                label = "⚡ TURBO" if active else "TURBO"
                color = (80, 150, 80) if active else (60, 60, 70)
            elif name == 'ai':
                active = self.agent_enabled
                label = "🤖 AI ON" if active else "AI OFF"
                color = (80, 120, 180) if active else (60, 60, 70)
            elif name == 'ai_panel':
                active = self.show_ai_panel
                label = "📊 PANEL" if active else "PANEL"
                color = (100, 100, 120) if active else (60, 60, 70)
            elif name == 'reset':
                active = False
                label = "↺ RESET"
                color = (120, 60, 60)
            else:
                continue
            
            # Draw button background
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, (100, 100, 110), rect, 1)
            
            # Draw button label
            text = self.font_small.render(label, True, (220, 220, 220))
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)
    
    def _draw_game_display(self):
        """Draw the main game display."""
        x, y = 10, 10
        
        # Title
        title = self.font_title.render(f"Game Display ({self.emulator.rom_title or 'No ROM'})", True, self.TEXT_COLOR)
        self.screen.blit(title, (x, y))
        y += 25
        
        # Border
        pygame.draw.rect(self.screen, self.BORDER_COLOR, 
                        (x - 2, y - 2, self.game_width + 4, self.game_height + 4), 2)
        
        # Scaled game display
        scaled = pygame.transform.scale(self.game_surface, (self.game_width, self.game_height))
        self.screen.blit(scaled, (x, y))
        
        # Pause indicator
        if self.emulator.paused:
            pause_text = self.font_title.render("PAUSED", True, (255, 100, 100))
            pause_rect = pause_text.get_rect(center=(x + self.game_width // 2, y + self.game_height // 2))
            pygame.draw.rect(self.screen, (0, 0, 0, 180), pause_rect.inflate(20, 10))
            self.screen.blit(pause_text, pause_rect)
        
        # FPS
        fps_text = self.font_small.render(f"FPS: {self.last_fps:.1f}", True, self.TEXT_COLOR)
        self.screen.blit(fps_text, (x, y + self.game_height + 5))
    
    def _draw_debug_panel(self):
        """Draw the debug information panel."""
        x = self.game_width + 20
        y = 10
        panel_width = self.debug_panel_width
        
        # CPU State
        y = self._draw_cpu_state(x, y)
        y += 15
        
        # PPU State
        y = self._draw_ppu_state(x, y)
        y += 15
        
        # Tilemap viewer
        y = self._draw_tilemap_viewer(x, y)
        y += 15
        
        # Tiles viewer (below tilemap)
        self._draw_tiles_viewer(x, y)
    
    def _draw_cpu_state(self, x: int, y: int) -> int:
        """Draw CPU register state."""
        title = self.font_title.render("CPU Registers", True, self.HIGHLIGHT_COLOR)
        self.screen.blit(title, (x, y))
        y += 22
        
        state = self.emulator.get_cpu_state()
        
        # Draw registers in columns
        col_width = 100
        regs = [
            ('AF', state['AF']), ('BC', state['BC']),
            ('DE', state['DE']), ('HL', state['HL']),
            ('SP', state['SP']), ('PC', state['PC']),
        ]
        
        for i, (name, value) in enumerate(regs):
            col = i % 3
            row = i // 3
            rx = x + col * col_width
            ry = y + row * 18
            
            text = self.font.render(f"{name}: {value:04X}", True, self.TEXT_COLOR)
            self.screen.blit(text, (rx, ry))
        
        y += 40
        
        # Flags
        flags = state['Flags']
        flag_text = f"Flags: Z={int(flags['Z'])} N={int(flags['N'])} H={int(flags['H'])} C={int(flags['C'])}"
        text = self.font.render(flag_text, True, self.TEXT_COLOR)
        self.screen.blit(text, (x, y))
        y += 18
        
        # IME and Halted
        ime_text = f"IME: {int(state['IME'])}  Halted: {int(state['Halted'])}"
        text = self.font.render(ime_text, True, self.TEXT_COLOR)
        self.screen.blit(text, (x, y))
        
        return y + 20
    
    def _draw_ppu_state(self, x: int, y: int) -> int:
        """Draw PPU state."""
        title = self.font_title.render("PPU State", True, self.HIGHLIGHT_COLOR)
        self.screen.blit(title, (x, y))
        y += 22
        
        state = self.emulator.get_ppu_state()
        
        # Mode names
        mode_names = ['H-Blank', 'V-Blank', 'OAM', 'Transfer']
        
        lines = [
            f"LY: {state['LY']:3d}  LYC: {state['LYC']:3d}  Mode: {mode_names[state['Mode']]}",
            f"SCX: {state['SCX']:3d}  SCY: {state['SCY']:3d}  WX: {state['WX']:3d}  WY: {state['WY']:3d}",
            f"LCDC: {state['LCDC']:02X}  STAT: {state['STAT']:02X}  BGP: {state['BGP']:02X}",
        ]
        
        for line in lines:
            text = self.font.render(line, True, self.TEXT_COLOR)
            self.screen.blit(text, (x, y))
            y += 18
        
        return y + 5
    
    def _draw_tilemap_viewer(self, x: int, y: int) -> int:
        """Draw the tilemap viewer."""
        # Title with toggle info
        map_addr = "0x9C00" if self.selected_tilemap else "0x9800"
        title = self.font_title.render(f"Tilemap ({map_addr}) [T to toggle]", True, self.HIGHLIGHT_COLOR)
        self.screen.blit(title, (x, y))
        y += 22
        
        # Get tilemap image
        tilemap = self.emulator.get_tilemap(self.selected_tilemap)
        
        # Convert to pygame surface
        pygame.surfarray.blit_array(self.tilemap_surface, tilemap.swapaxes(0, 1))
        
        # Draw border
        display_size = 200  # Scaled down to fit
        pygame.draw.rect(self.screen, self.BORDER_COLOR,
                        (x - 2, y - 2, display_size + 4, display_size + 4), 1)
        
        # Scale and draw
        scaled = pygame.transform.scale(self.tilemap_surface, (display_size, display_size))
        self.screen.blit(scaled, (x, y))
        
        # Draw viewport rectangle (shows visible area)
        scx = self.emulator.ppu.scx
        scy = self.emulator.ppu.scy
        scale_factor = display_size / 256
        
        vp_x = int(x + scx * scale_factor)
        vp_y = int(y + scy * scale_factor)
        vp_w = int(160 * scale_factor)
        vp_h = int(144 * scale_factor)
        
        # Handle wrap-around
        pygame.draw.rect(self.screen, (255, 100, 100), 
                        (vp_x, vp_y, vp_w, vp_h), 2)
        
        return y + display_size + 5
    
    def _draw_tiles_viewer(self, x: int, y: int):
        """Draw the tiles viewer."""
        bank_text = f"Bank {self.selected_vram_bank}" if self.emulator.cgb_mode else ""
        title = self.font_title.render(f"VRAM Tiles {bank_text} [V to toggle]", True, self.HIGHLIGHT_COLOR)
        self.screen.blit(title, (x, y))
        y += 22
        
        # Get tiles image
        tiles = self.emulator.get_tiles(self.selected_vram_bank)
        
        # Convert to pygame surface
        pygame.surfarray.blit_array(self.tiles_surface, tiles.swapaxes(0, 1))
        
        # Draw border
        pygame.draw.rect(self.screen, self.BORDER_COLOR,
                        (x - 2, y - 2, self.tiles_width + 4, self.tiles_height + 4), 1)
        
        # Draw tiles
        self.screen.blit(self.tiles_surface, (x, y))
    
    def _draw_ai_panel_right(self):
        """Draw AI goal and thinking panel on the right side."""
        # Position on right side of screen
        x = self.game_width + 20
        y = 10
        panel_width = self.ai_panel_width - 10
        panel_height = self.game_height + 60
        
        # Panel background
        pygame.draw.rect(self.screen, (30, 32, 38), 
                        (x - 5, y - 5, panel_width + 10, panel_height + 10))
        pygame.draw.rect(self.screen, self.BORDER_COLOR,
                        (x - 5, y - 5, panel_width + 10, panel_height + 10), 1)
        
        # === ATTEMPT COUNTER ===
        attempt_color = (100, 200, 100) if self.agent_enabled else (150, 150, 150)
        attempt_text = self.font_title.render(
            f"Attempt {self.current_attempt}/{self.max_attempts}", True, attempt_color
        )
        self.screen.blit(attempt_text, (x, y))
        y += 25
        
        # === CURRENT GOAL (prominent) ===
        goal_name = "No Goal"
        goal_desc = ""
        goal_progress = ""
        
        if self.agent_manager and self.agent_manager.agent:
            if hasattr(self.agent_manager.agent, 'get_current_stage_info'):
                info = self.agent_manager.agent.get_current_stage_info()
                goal_name = info.get('name', 'Unknown')
                goal_desc = info.get('goal', '')
                goal_progress = info.get('progress', '')
        
        # Goal box
        goal_box_h = 70
        pygame.draw.rect(self.screen, (40, 60, 50), (x, y, panel_width, goal_box_h))
        pygame.draw.rect(self.screen, (80, 150, 80), (x, y, panel_width, goal_box_h), 2)
        
        # Goal title
        goal_title = self.font_title.render(f"🎯 {goal_name}", True, (120, 220, 120))
        self.screen.blit(goal_title, (x + 5, y + 5))
        
        # Goal description
        desc_lines = [goal_desc[i:i+45] for i in range(0, len(goal_desc), 45)][:2]
        for i, line in enumerate(desc_lines):
            desc_text = self.font_small.render(line, True, (180, 200, 180))
            self.screen.blit(desc_text, (x + 5, y + 28 + i * 14))
        
        # Progress
        if goal_progress:
            prog_text = self.font_small.render(goal_progress, True, (150, 150, 100))
            self.screen.blit(prog_text, (x + panel_width - 80, y + 5))
        
        y += goal_box_h + 10
        
        # === GAME STATE ===
        if self.agent_manager and self.agent_manager.agent:
            if hasattr(self.agent_manager.agent, 'get_memory_state'):
                mem_state = self.agent_manager.agent.get_memory_state()
                if mem_state:
                    pos = mem_state.player_position
                    state_text = self.font_small.render(
                        f"📍 ({pos.x},{pos.y}) {pos.map_name[:20]}", True, (150, 180, 200)
                    )
                    self.screen.blit(state_text, (x, y))
                    y += 16
                    
                    # Party/battle status
                    if mem_state.battle.in_battle:
                        b = mem_state.battle
                        battle_text = self.font_small.render(
                            f"⚔️ BATTLE vs {b.enemy_name} Lv{b.enemy_level}", True, (250, 150, 150)
                        )
                        self.screen.blit(battle_text, (x, y))
                    elif mem_state.party_count > 0:
                        party_str = ", ".join(f"{p.species_name[:6]}" for p in mem_state.party[:3])
                        party_text = self.font_small.render(f"Party: {party_str}", True, (150, 200, 150))
                        self.screen.blit(party_text, (x, y))
                    else:
                        no_party = self.font_small.render("No Pokemon yet", True, (150, 150, 150))
                        self.screen.blit(no_party, (x, y))
                    y += 18
        
        # Divider
        pygame.draw.line(self.screen, (60, 60, 70), (x, y), (x + panel_width, y), 1)
        y += 8
        
        # === THINKING LOG ===
        thinking_title = self.font_small.render("💭 AI Thinking:", True, self.HIGHLIGHT_COLOR)
        self.screen.blit(thinking_title, (x, y))
        y += 18
        
        # Draw thinking log
        if self.agent_thinking_log:
            recent = self.agent_thinking_log[-self.max_thinking_display:]
            for line in recent:
                # Truncate long lines
                max_chars = 42
                display_line = line[-max_chars:] if len(line) > max_chars else line
                
                # Color code
                if '✓' in line or 'COMPLETE' in line:
                    color = (100, 220, 100)
                elif '✗' in line or 'fail' in line.lower() or 'error' in line.lower():
                    color = (220, 100, 100)
                elif 'GOAL' in line or '===' in line:
                    color = (220, 200, 100)
                elif 'Attempt' in line:
                    color = (150, 200, 250)
                else:
                    color = (170, 170, 180)
                
                text = self.font_small.render(display_line, True, color)
                self.screen.blit(text, (x, y))
                y += 14
        else:
            no_log = self.font_small.render("Waiting for AI...", True, (100, 100, 100))
            self.screen.blit(no_log, (x, y))
    
    def _draw_help(self):
        """Draw control help at bottom."""
        y = self.window_height - 80
        x = 10
        
        turbo_str = "ON" if self.turbo_mode else "OFF"
        agent_str = "ON" if self.agent_enabled else "OFF"
        help_lines = [
            "Controls: Arrow Keys = D-Pad | Z = A | X = B | Enter = Start | RShift = Select",
            "Space = Pause | R = Reset | TAB = Turbo | F2 = AI On/Off | F3 = Cycle Agent | F4 = Skip Stage",
            f"Frame: {self.emulator.total_frames} | Turbo: {turbo_str} | {self.agent_status_text}"
        ]
        
        for line in help_lines:
            text = self.font_small.render(line, True, (120, 130, 140))
            self.screen.blit(text, (x, y))
            y += 16


class SimplerGUI:
    """
    Simpler GUI focused on just the game display and tilemap.
    """
    
    def __init__(self, emulator, scale: int = 4):
        self.emulator = emulator
        self.scale = scale
        
        pygame.init()
        
        # Calculate layout
        game_w = 160 * scale
        game_h = 144 * scale
        tilemap_size = 256 * 2  # 2x scale for tilemap
        tiles_w = 128 * 2
        tiles_h = 192 * 2
        
        padding = 20
        
        # Window layout: Game | Tilemap | Tiles
        self.window_width = game_w + tilemap_size + tiles_w + padding * 4
        self.window_height = max(game_h, tilemap_size, tiles_h) + padding * 2
        
        pygame.display.set_caption(f"GBC Emulator - {emulator.rom_title or 'No ROM'}")
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.clock = pygame.time.Clock()
        
        # Positions
        self.game_pos = (padding, padding)
        self.tilemap_pos = (game_w + padding * 2, padding)
        self.tiles_pos = (game_w + tilemap_size + padding * 3, padding)
        
        # Surfaces
        self.game_surface = pygame.Surface((160, 144))
        self.tilemap_surface = pygame.Surface((256, 256))
        self.tiles_surface = pygame.Surface((128, 192))
        
        self.running = True
        self.selected_tilemap = 0
        
        # Key mapping
        self.key_map = {
            pygame.K_z: 'a',
            pygame.K_x: 'b',
            pygame.K_RETURN: 'start',
            pygame.K_RSHIFT: 'select',
            pygame.K_UP: 'up',
            pygame.K_DOWN: 'down',
            pygame.K_LEFT: 'left',
            pygame.K_RIGHT: 'right',
        }
    
    def run(self):
        """Main loop."""
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_t:
                        self.selected_tilemap = 1 - self.selected_tilemap
                    elif event.key == pygame.K_SPACE:
                        self.emulator.paused = not self.emulator.paused
                    elif event.key in self.key_map:
                        self.emulator.press_button(self.key_map[event.key])
                elif event.type == pygame.KEYUP:
                    if event.key in self.key_map:
                        self.emulator.release_button(self.key_map[event.key])
            
            if not self.emulator.paused:
                frame = self.emulator.run_frame()
                pygame.surfarray.blit_array(self.game_surface, frame.swapaxes(0, 1))
            
            # Draw
            self.screen.fill((20, 25, 30))
            
            # Game
            scaled_game = pygame.transform.scale(
                self.game_surface, (160 * self.scale, 144 * self.scale))
            self.screen.blit(scaled_game, self.game_pos)
            
            # Tilemap
            tilemap = self.emulator.get_tilemap(self.selected_tilemap)
            pygame.surfarray.blit_array(self.tilemap_surface, tilemap.swapaxes(0, 1))
            scaled_tilemap = pygame.transform.scale(self.tilemap_surface, (512, 512))
            self.screen.blit(scaled_tilemap, self.tilemap_pos)
            
            # Draw viewport on tilemap
            scx, scy = self.emulator.ppu.scx, self.emulator.ppu.scy
            vp_scale = 2
            pygame.draw.rect(self.screen, (255, 80, 80),
                           (self.tilemap_pos[0] + scx * vp_scale,
                            self.tilemap_pos[1] + scy * vp_scale,
                            160 * vp_scale, 144 * vp_scale), 2)
            
            # Tiles
            tiles = self.emulator.get_tiles(0)
            pygame.surfarray.blit_array(self.tiles_surface, tiles.swapaxes(0, 1))
            scaled_tiles = pygame.transform.scale(self.tiles_surface, (256, 384))
            self.screen.blit(scaled_tiles, self.tiles_pos)
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()


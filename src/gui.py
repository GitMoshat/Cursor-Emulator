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
        
        # Debug visualization
        self.debug_scan_enabled = True  # Show AI scan rectangles (default ON)
        self.scan_rects: list = []  # Rectangles to draw
        self.scan_points: list = []  # Points of interest
        
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
        btn_w, btn_h = 60, 24
        btn_y = self.game_height + 15  # Below game display
        btn_x = 10
        spacing = 4
        
        # Create button rectangles
        self.buttons = {
            'turbo': pygame.Rect(btn_x, btn_y, btn_w, btn_h),
            'ai': pygame.Rect(btn_x + (btn_w + spacing), btn_y, btn_w, btn_h),
            'debug': pygame.Rect(btn_x + (btn_w + spacing) * 2, btn_y, btn_w, btn_h),
            'reset': pygame.Rect(btn_x + (btn_w + spacing) * 3, btn_y, btn_w, btn_h),
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
                    mode_str = "TURBO ON" if self.turbo_mode else "Normal"
                    self.agent_thinking_log.append(f"[Speed] {mode_str}")
                
                elif name == 'ai':
                    self._toggle_agent()
                
                elif name == 'debug':
                    self.debug_scan_enabled = not self.debug_scan_enabled
                    mode_str = "ON" if self.debug_scan_enabled else "OFF"
                    self.agent_thinking_log.append(f"[Debug] Scan visualization {mode_str}")
                
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
                label = "TURBO" if active else "TURBO"
                color = (80, 150, 80) if active else (60, 60, 70)
            elif name == 'ai':
                active = self.agent_enabled
                label = "AI ON" if active else "AI OFF"
                color = (80, 120, 180) if active else (60, 60, 70)
            elif name == 'debug':
                active = self.debug_scan_enabled
                label = "DEBUG" if active else "DEBUG"
                color = (180, 60, 60) if active else (60, 60, 70)
            elif name == 'reset':
                active = False
                label = "RESET"
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
        
        # Draw debug scan overlay
        if self.debug_scan_enabled and self.agent_enabled:
            self._draw_debug_scan_overlay(x, y)
        
        # Pause indicator
        if self.emulator.paused:
            pause_text = self.font_title.render("PAUSED", True, (255, 100, 100))
            pause_rect = pause_text.get_rect(center=(x + self.game_width // 2, y + self.game_height // 2))
            pygame.draw.rect(self.screen, (0, 0, 0, 180), pause_rect.inflate(20, 10))
            self.screen.blit(pause_text, pause_rect)
        
        # FPS
        fps_text = self.font_small.render(f"FPS: {self.last_fps:.1f}", True, self.TEXT_COLOR)
        self.screen.blit(fps_text, (x, y + self.game_height + 5))
    
    def _draw_debug_scan_overlay(self, display_x: int, display_y: int):
        """Draw debug rectangles showing what the AI is currently interacting with."""
        if not self.agent_manager or not self.agent_manager.agent:
            return
        
        # Scale factor from game (160x144) to display
        scale_x = self.game_width / 160
        scale_y = self.game_height / 144
        
        # Get memory state for scanning info
        mem_state = None
        if hasattr(self.agent_manager.agent, 'get_memory_state'):
            mem_state = self.agent_manager.agent.get_memory_state()
        
        if not mem_state:
            return
        
        # Colors
        COLOR_FOCUS = (255, 50, 50)       # Red - current focus/selection
        COLOR_MENU = (255, 100, 100)      # Light red - menu area
        COLOR_CURSOR = (255, 255, 0)      # Yellow - cursor position
        COLOR_OPTION = (255, 200, 50)     # Orange - selectable options
        
        import time
        pulse = (int(time.time() * 6) % 2) == 0  # Faster pulse
        
        # Determine what mode we're in and highlight accordingly
        if mem_state.battle.in_battle:
            # === BATTLE MODE ===
            self._draw_battle_debug(display_x, display_y, scale_x, scale_y, mem_state, pulse)
        
        elif mem_state.menu.text_active or mem_state.menu.in_menu:
            # === MENU/DIALOG MODE ===
            self._draw_menu_debug(display_x, display_y, scale_x, scale_y, mem_state, pulse)
        
        else:
            # === OVERWORLD MODE ===
            self._draw_overworld_debug(display_x, display_y, scale_x, scale_y, mem_state, pulse)
        
        # Always show current state label
        state_label = "BATTLE" if mem_state.battle.in_battle else \
                     "MENU" if mem_state.menu.in_menu else \
                     "DIALOG" if mem_state.menu.text_active else "EXPLORE"
        color = (255, 100, 100) if pulse else (200, 80, 80)
        state_text = self.font_small.render(f"[{state_label}]", True, color)
        self.screen.blit(state_text, (display_x + 5, display_y + 5))
    
    def _draw_menu_debug(self, dx, dy, sx, sy, mem_state, pulse):
        """Draw debug overlay for menu/dialog screens that follows the actual selection."""
        COLOR_FOCUS = (255, 50, 50) if pulse else (200, 40, 40)
        COLOR_BOX = (255, 100, 100)
        COLOR_ARROW = (255, 255, 0)
        
        cursor_pos = mem_state.menu.cursor_position
        cursor_x = mem_state.menu.cursor_x
        cursor_y = mem_state.menu.cursor_y
        screen_type = mem_state.menu.screen_type
        selection_y = mem_state.menu.selection_y_pixel
        
        # Screen type label
        type_label = self.font_small.render(f"[{screen_type.upper()}]", True, (200, 150, 150))
        self.screen.blit(type_label, (dx + self.game_width - 100, dy + 22))
        
        # Draw based on screen type
        if screen_type == "gender_select":
            # Gender select: BOY/GIRL options
            # These are typically displayed in a window in the center-right
            option_x = int(88 * sx)  # Right side of screen
            option_w = int(48 * sx)
            option_h = int(14 * sy)
            
            labels = ["BOY", "GIRL"]
            for i in range(2):
                opt_y = dy + int((56 + i * 16) * sy)
                is_selected = (i == cursor_pos)
                
                if is_selected:
                    # Draw prominent selection box
                    pygame.draw.rect(self.screen, (60, 15, 15), 
                                    (dx + option_x - 4, opt_y - 2, option_w + 8, option_h + 4))
                    pygame.draw.rect(self.screen, COLOR_FOCUS, 
                                    (dx + option_x - 4, opt_y - 2, option_w + 8, option_h + 4), 3)
                    # Selection arrow
                    arrow_x = dx + option_x - 16
                    pygame.draw.polygon(self.screen, COLOR_ARROW, [
                        (arrow_x, opt_y + option_h//2),
                        (arrow_x + 10, opt_y - 2),
                        (arrow_x + 10, opt_y + option_h + 2)
                    ])
                    # Label the selection
                    sel_text = self.font_small.render(f"→ {labels[i]}", True, COLOR_ARROW)
                    self.screen.blit(sel_text, (dx + 5, dy + 36))
                else:
                    pygame.draw.rect(self.screen, (100, 60, 60), 
                                    (dx + option_x, opt_y, option_w, option_h), 1)
        
        elif screen_type == "option_menu":
            # Yes/No style menus
            option_x = int(96 * sx)  # Usually right side
            option_w = int(48 * sx)
            option_h = int(14 * sy)
            
            for i in range(2):
                opt_y = dy + int((72 + i * 16) * sy)
                is_selected = (i == cursor_pos)
                
                if is_selected:
                    pygame.draw.rect(self.screen, (60, 15, 15), 
                                    (dx + option_x - 4, opt_y - 2, option_w + 8, option_h + 4))
                    pygame.draw.rect(self.screen, COLOR_FOCUS, 
                                    (dx + option_x - 4, opt_y - 2, option_w + 8, option_h + 4), 3)
                    # Arrow
                    pygame.draw.polygon(self.screen, COLOR_ARROW, [
                        (dx + option_x - 14, opt_y + option_h//2),
                        (dx + option_x - 6, opt_y),
                        (dx + option_x - 6, opt_y + option_h)
                    ])
                else:
                    pygame.draw.rect(self.screen, (100, 60, 60), 
                                    (dx + option_x, opt_y, option_w, option_h), 1)
        
        elif screen_type == "name_entry":
            # Name entry grid
            grid_cols = 10
            grid_start_x = 8
            grid_start_y = 48
            tile_size = 16
            
            grid_x = cursor_pos % grid_cols
            grid_y_pos = cursor_pos // grid_cols
            
            px = dx + int((grid_start_x + grid_x * tile_size) * sx)
            py = dy + int((grid_start_y + grid_y_pos * tile_size) * sy)
            pw = int(tile_size * sx)
            ph = int(tile_size * sy)
            
            # Draw grid position highlight
            pygame.draw.rect(self.screen, COLOR_FOCUS, (px - 2, py - 2, pw + 4, ph + 4), 3)
            
            # Grid position label
            pos_text = self.font_small.render(f"Grid: ({grid_x},{grid_y_pos})", True, COLOR_ARROW)
            self.screen.blit(pos_text, (dx + 5, dy + 36))
        
        elif screen_type == "dialog":
            # Dialog box - highlight the text area
            text_box_y = dy + int(96 * sy)
            text_box_h = int(48 * sy)
            pygame.draw.rect(self.screen, COLOR_BOX, 
                            (dx + 4, text_box_y, self.game_width - 8, text_box_h), 2)
            
            # "Press A" indicator if text is waiting
            pygame.draw.polygon(self.screen, COLOR_ARROW, [
                (dx + self.game_width - 20, text_box_y + text_box_h - 8),
                (dx + self.game_width - 12, text_box_y + text_box_h - 16),
                (dx + self.game_width - 28, text_box_y + text_box_h - 16)
            ])
        
        else:
            # Generic menu - use calculated Y position
            option_x = int(16 * sx)
            option_w = int(120 * sx)
            option_h = int(14 * sy)
            opt_y = dy + int(selection_y * sy)
            
            pygame.draw.rect(self.screen, COLOR_FOCUS, 
                            (dx + option_x, opt_y - 2, option_w, option_h + 4), 2)
        
        # Always show cursor position info
        cursor_text = self.font_small.render(f"C:{cursor_pos} XY:({cursor_x},{cursor_y})", True, (255, 200, 100))
        self.screen.blit(cursor_text, (dx + 5, dy + 22))
    
    def _draw_battle_debug(self, dx, dy, sx, sy, mem_state, pulse):
        """Draw debug overlay for battle screens that highlights the selected option."""
        COLOR_FOCUS = (255, 50, 50) if pulse else (200, 40, 40)
        COLOR_ARROW = (255, 255, 0)
        
        battle = mem_state.battle
        menu_state = battle.menu_state
        
        # Battle menu is a 2x2 grid at bottom-right
        # FIGHT | PKMN
        # ITEM  | RUN
        menu_base_x = 80  # Pixels from left
        menu_base_y = 112  # Pixels from top
        cell_w = 40
        cell_h = 16
        
        options = ["FIGHT", "PKMN", "ITEM", "RUN"]
        
        # Draw all four options with appropriate highlighting
        for i in range(4):
            col = i % 2
            row = i // 2
            opt_x = dx + int((menu_base_x + col * cell_w) * sx)
            opt_y = dy + int((menu_base_y + row * cell_h) * sy)
            opt_w = int(cell_w * sx) - 4
            opt_h = int(cell_h * sy) - 2
            
            if i == menu_state:
                # Selected option - prominent highlight
                pygame.draw.rect(self.screen, (50, 15, 15), 
                                (opt_x - 3, opt_y - 2, opt_w + 6, opt_h + 4))
                pygame.draw.rect(self.screen, COLOR_FOCUS, 
                                (opt_x - 3, opt_y - 2, opt_w + 6, opt_h + 4), 3)
                # Selection arrow
                pygame.draw.polygon(self.screen, COLOR_ARROW, [
                    (opt_x - 10, opt_y + opt_h//2),
                    (opt_x - 4, opt_y),
                    (opt_x - 4, opt_y + opt_h)
                ])
            else:
                # Non-selected options - dim outline
                pygame.draw.rect(self.screen, (80, 50, 50), 
                                (opt_x, opt_y, opt_w, opt_h), 1)
        
        # Current selection label
        if menu_state < len(options):
            label = self.font_small.render(f"→ {options[menu_state]}", True, COLOR_ARROW)
            self.screen.blit(label, (dx + 5, dy + 36))
        
        # Enemy info box (top-left area)
        enemy_box_x = dx + int(8 * sx)
        enemy_box_y = dy + int(8 * sy)
        enemy_box_w = int(96 * sx)
        enemy_box_h = int(32 * sy)
        pygame.draw.rect(self.screen, (255, 150, 50), 
                        (enemy_box_x, enemy_box_y, enemy_box_w, enemy_box_h), 1)
        
        # Player info box (right side, middle)
        player_box_x = dx + int(56 * sx)
        player_box_y = dy + int(56 * sy)
        player_box_w = int(96 * sx)
        player_box_h = int(32 * sy)
        pygame.draw.rect(self.screen, (100, 255, 100), 
                        (player_box_x, player_box_y, player_box_w, player_box_h), 1)
        
        # Battle status
        enemy_info = f"vs {battle.enemy_name} Lv{battle.enemy_level}"
        info_text = self.font_small.render(enemy_info, True, (255, 150, 150))
        self.screen.blit(info_text, (dx + 5, dy + 22))
    
    def _draw_overworld_debug(self, dx, dy, sx, sy, mem_state, pulse):
        """Draw debug overlay for overworld exploration."""
        COLOR_PLAYER = (255, 50, 50) if pulse else (200, 40, 40)
        COLOR_INTERACT = (255, 255, 50)
        
        # Player is typically centered on screen
        px = dx + int(80 * sx)
        py = dy + int(72 * sy)
        pw = int(16 * sx)
        ph = int(16 * sy)
        
        # Player box
        pygame.draw.rect(self.screen, COLOR_PLAYER, 
                        (px - pw//2, py - ph//2, pw, ph), 2)
        
        # Facing direction and interaction zone
        facing = mem_state.player_position.facing
        tile_size = int(16 * sx)
        interact_x, interact_y = px, py
        
        if facing == "up":
            interact_y -= tile_size
            pygame.draw.line(self.screen, COLOR_INTERACT, (px, py - ph//2), (px, py - ph//2 - 10), 2)
        elif facing == "down":
            interact_y += tile_size
            pygame.draw.line(self.screen, COLOR_INTERACT, (px, py + ph//2), (px, py + ph//2 + 10), 2)
        elif facing == "left":
            interact_x -= tile_size
            pygame.draw.line(self.screen, COLOR_INTERACT, (px - pw//2, py), (px - pw//2 - 10, py), 2)
        elif facing == "right":
            interact_x += tile_size
            pygame.draw.line(self.screen, COLOR_INTERACT, (px + pw//2, py), (px + pw//2 + 10, py), 2)
        
        # Interaction zone
        pygame.draw.rect(self.screen, COLOR_INTERACT,
                        (interact_x - tile_size//2, interact_y - tile_size//2, tile_size, tile_size), 2)
        
        # Position
        pos = mem_state.player_position
        pos_text = self.font_small.render(f"({pos.x},{pos.y}) {pos.facing}", True, (200, 200, 200))
        self.screen.blit(pos_text, (dx + 5, dy + 22))
    
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


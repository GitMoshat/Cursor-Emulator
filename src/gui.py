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
        
        # Total window size (extra height for AI thinking panel)
        self.window_width = self.game_width + self.debug_panel_width + 30
        self.window_height = max(self.game_height + 280, 750)
        
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
        self.frame_skip = 0  # 0 = no skip, 1 = skip every other, 2 = skip 2 of 3
        self.frame_counter = 0
        self.turbo_mode = False
        
        # AI Agent system
        self.agent_manager = None
        self.agent_enabled = False
        self.agent_status_text = "AI: Not loaded"
        self.agent_thinking_log: list = []
        self.max_thinking_display = 8  # Lines to show in panel
        if AGENT_AVAILABLE:
            self.agent_manager = AgentManager(emulator)
        
        # State
        self.running = True
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
                # Create default agent (Memory-based for fast, accurate play)
                agent = self.agent_manager.create_agent('memory')
                if agent:
                    self.agent_manager.set_agent(agent)
                    self.agent_thinking_log.append("[Init] Created MemoryAgent (fast, no vision needed)")
            
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
    
    def _handle_events(self):
        """Handle input events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self.emulator.running = False
            
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
        
        # Draw game display
        self._draw_game_display()
        
        # Draw debug panel
        if self.show_debug:
            self._draw_debug_panel()
        
        # Draw AI thinking panel (below game display)
        if self.agent_enabled or self.agent_thinking_log:
            self._draw_thinking_panel()
        
        # Draw help
        self._draw_help()
        
        pygame.display.flip()
    
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
    
    def _draw_thinking_panel(self):
        """Draw the AI thinking/reasoning panel with memory state."""
        # Position below game display
        x = 10
        y = self.game_height + 50
        panel_width = self.game_width + 200  # Wider for memory info
        panel_height = 160
        
        # Background
        pygame.draw.rect(self.screen, self.PANEL_BG, 
                        (x - 5, y - 5, panel_width + 10, panel_height + 10))
        pygame.draw.rect(self.screen, self.BORDER_COLOR,
                        (x - 5, y - 5, panel_width + 10, panel_height + 10), 1)
        
        # Title with stage info
        stage_info = ""
        memory_info = ""
        if self.agent_manager and self.agent_manager.agent:
            if hasattr(self.agent_manager.agent, 'get_current_stage_info'):
                info = self.agent_manager.agent.get_current_stage_info()
                stage_info = f" - {info.get('name', '?')}"
            
            # Get memory state if available
            if hasattr(self.agent_manager.agent, 'get_memory_state'):
                mem_state = self.agent_manager.agent.get_memory_state()
                if mem_state:
                    pos = mem_state.player_position
                    memory_info = f" | Pos:({pos.x},{pos.y}) Map:{pos.map_name[:15]}"
                    if mem_state.battle.in_battle:
                        memory_info += " [BATTLE]"
        
        title_color = (100, 200, 100) if self.agent_enabled else self.HIGHLIGHT_COLOR
        title = self.font_title.render(f"🤖 AI{stage_info}{memory_info}", True, title_color)
        self.screen.blit(title, (x, y))
        y += 20
        
        # Draw memory state summary if available
        if self.agent_manager and self.agent_manager.agent and hasattr(self.agent_manager.agent, 'get_memory_state'):
            mem_state = self.agent_manager.agent.get_memory_state()
            if mem_state:
                # Party info
                if mem_state.party:
                    party_str = " | ".join(
                        f"{p.species_name[:8]}:{p.current_hp}/{p.max_hp}"
                        for p in mem_state.party[:3]
                    )
                    party_text = self.font_small.render(f"Party: {party_str}", True, (150, 200, 150))
                    self.screen.blit(party_text, (x, y))
                    y += 14
                
                # Status line
                status_parts = []
                if mem_state.badges > 0:
                    status_parts.append(f"Badges:{mem_state.badges}")
                if mem_state.money > 0:
                    status_parts.append(f"${mem_state.money}")
                if not mem_state.has_starter:
                    status_parts.append("NO STARTER")
                if mem_state.battle.in_battle:
                    b = mem_state.battle
                    status_parts.append(f"vs {b.enemy_name} Lv{b.enemy_level}")
                
                if status_parts:
                    status_text = self.font_small.render(" | ".join(status_parts), True, (200, 200, 150))
                    self.screen.blit(status_text, (x, y))
                    y += 14
        
        # Draw goal if available
        if self.agent_manager and self.agent_manager.agent:
            if hasattr(self.agent_manager.agent, 'get_current_stage_info'):
                info = self.agent_manager.agent.get_current_stage_info()
                goal = info.get('goal', '')[:70]
                goal_text = self.font_small.render(f"Goal: {goal}", True, (180, 180, 100))
                self.screen.blit(goal_text, (x, y))
                y += 14
        
        # Divider
        y += 2
        
        # Draw thinking log
        if self.agent_thinking_log:
            recent = self.agent_thinking_log[-self.max_thinking_display:]
            for line in recent:
                # Truncate long lines
                display_line = line[-80:] if len(line) > 80 else line
                
                # Color code based on content
                if '✓' in line or 'success' in line.lower():
                    color = (100, 200, 100)
                elif '✗' in line or 'fail' in line.lower() or 'error' in line.lower():
                    color = (200, 100, 100)
                elif 'Goal:' in line or 'Stage:' in line or 'Rule:' in line:
                    color = (200, 200, 100)
                elif 'LLM' in line:
                    color = (150, 150, 250)
                else:
                    color = self.TEXT_COLOR
                
                text = self.font_small.render(display_line, True, color)
                self.screen.blit(text, (x, y))
                y += 13
        else:
            no_log = self.font_small.render("Press F2 to start AI agent (memory-based, no vision needed)", True, (120, 120, 120))
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


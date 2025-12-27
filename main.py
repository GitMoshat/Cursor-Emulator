#!/usr/bin/env python3
"""
Game Boy Color Emulator
A fully-featured GBC emulator with tilemap viewer and debug tools.

Usage:
    python main.py <rom_file>
    python main.py                  # Uses default Dokemon ROM path
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def show_loading_screen():
    """Show a loading screen while heavy modules load."""
    import pygame
    pygame.init()
    
    # Small loading window
    screen = pygame.display.set_mode((400, 200))
    pygame.display.set_caption("GBC Emulator - Loading...")
    
    # Colors
    BG_COLOR = (20, 25, 35)
    BAR_BG = (40, 45, 55)
    BAR_FG = (100, 180, 100)
    TEXT_COLOR = (200, 200, 200)
    
    font = pygame.font.Font(None, 24)
    small_font = pygame.font.Font(None, 18)
    
    def update_progress(progress: float, message: str):
        """Update the loading screen."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
        
        screen.fill(BG_COLOR)
        
        # Title
        title = font.render("GBC Emulator", True, TEXT_COLOR)
        screen.blit(title, (400//2 - title.get_width()//2, 40))
        
        # Status message
        status = small_font.render(message, True, (150, 150, 150))
        screen.blit(status, (400//2 - status.get_width()//2, 80))
        
        # Progress bar background
        bar_x, bar_y = 50, 120
        bar_w, bar_h = 300, 24
        pygame.draw.rect(screen, BAR_BG, (bar_x, bar_y, bar_w, bar_h))
        
        # Progress bar fill
        fill_w = int(bar_w * progress)
        if fill_w > 0:
            pygame.draw.rect(screen, BAR_FG, (bar_x, bar_y, fill_w, bar_h))
        
        # Progress bar border
        pygame.draw.rect(screen, (80, 85, 95), (bar_x, bar_y, bar_w, bar_h), 2)
        
        # Percentage
        pct_text = font.render(f"{int(progress * 100)}%", True, TEXT_COLOR)
        screen.blit(pct_text, (400//2 - pct_text.get_width()//2, 155))
        
        pygame.display.flip()
    
    return update_progress, screen


def main():
    # Default ROM path
    default_rom = r"E:\ISO's\Dokemon.gbc"
    
    # Get ROM path from args or use default
    if len(sys.argv) > 1:
        rom_path = sys.argv[1]
    else:
        rom_path = default_rom
    
    # Show loading screen immediately (pygame is light)
    update_progress, loading_screen = show_loading_screen()
    update_progress(0.05, "Initializing...")
    
    # Check if ROM exists
    if not os.path.exists(rom_path):
        import pygame
        pygame.quit()
        print(f"ROM file not found: {rom_path}")
        print("\nUsage: python main.py <rom_file>")
        return 1
    
    # Lazy load heavy modules with progress updates
    update_progress(0.10, "Loading NumPy...")
    import numpy as np
    
    update_progress(0.25, "Loading Numba JIT compiler...")
    # Numba is the heaviest - it compiles on first import
    try:
        from numba import njit
    except ImportError:
        pass  # Numba optional
    
    update_progress(0.45, "Loading emulator core...")
    from src.emulator import Emulator
    
    update_progress(0.60, "Loading PPU renderer...")
    from src.ppu_fast import PPU_Fast
    
    update_progress(0.70, "Loading GUI system...")
    from src.gui import EmulatorGUI
    
    update_progress(0.80, "Creating emulator instance...")
    emulator = Emulator()
    
    update_progress(0.85, f"Loading ROM: {os.path.basename(rom_path)}...")
    if not emulator.load_rom(rom_path):
        import pygame
        pygame.quit()
        print("Failed to load ROM!")
        return 1
    
    update_progress(0.95, "Initializing AI agent system...")
    # Pre-import agent modules
    from src.agent import AgentManager
    
    update_progress(1.0, "Ready!")
    
    # Small delay to show 100%
    import pygame
    pygame.time.wait(200)
    
    # Close loading screen
    pygame.quit()
    
    # Print controls to console
    print("=" * 60)
    print("  GBC Emulator - Ready!")
    print("=" * 60)
    print("\nControls: Arrow=D-Pad, Z=A, X=B, Enter=Start, RShift=Select")
    print("TAB=Turbo, F2=AI On/Off, F3=Cycle Agent, Space=Pause, ESC=Quit")
    print()
    
    # Create GUI and run (re-initializes pygame)
    try:
        gui = EmulatorGUI(emulator, scale=3)
        gui.run()
    except Exception as e:
        print(f"Error running emulator: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

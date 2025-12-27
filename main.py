#!/usr/bin/env python3
"""
Game Boy Color Emulator
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Show splash IMMEDIATELY using tkinter (faster than pygame)
def show_splash():
    """Show splash screen instantly."""
    try:
        import tkinter as tk
        
        splash = tk.Tk()
        splash.title("Loading...")
        splash.overrideredirect(True)  # No window decorations
        
        # Center on screen
        w, h = 300, 100
        sw = splash.winfo_screenwidth()
        sh = splash.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        splash.geometry(f"{w}x{h}+{x}+{y}")
        
        # Dark background
        splash.configure(bg='#1a1a2e')
        
        # Title
        title = tk.Label(splash, text="GBC Emulator", font=("Arial", 16, "bold"),
                        fg='#a0a0a0', bg='#1a1a2e')
        title.pack(pady=15)
        
        # Status
        status_var = tk.StringVar(value="Starting...")
        status = tk.Label(splash, textvariable=status_var, font=("Arial", 10),
                         fg='#707070', bg='#1a1a2e')
        status.pack()
        
        # Progress bar frame
        bar_frame = tk.Frame(splash, bg='#2a2a3e', height=10)
        bar_frame.pack(fill='x', padx=30, pady=10)
        bar_frame.pack_propagate(False)
        
        progress_bar = tk.Frame(bar_frame, bg='#4a9f4a', height=10, width=0)
        progress_bar.place(x=0, y=0, height=10)
        
        splash.update()
        
        def update(pct, msg):
            status_var.set(msg)
            progress_bar.configure(width=int(240 * pct))
            splash.update()
        
        return splash, update
    except:
        # Tkinter not available, return dummy
        return None, lambda p, m: None


def main():
    # Show splash FIRST
    splash, update_splash = show_splash()
    update_splash(0.05, "Initializing...")
    
    # Default ROM path
    default_rom = r"E:\ISO's\Dokemon.gbc"
    rom_path = sys.argv[1] if len(sys.argv) > 1 else default_rom
    
    # Check ROM
    if not os.path.exists(rom_path):
        if splash:
            splash.destroy()
        print(f"ROM file not found: {rom_path}")
        return 1
    
    try:
        update_splash(0.10, "Loading NumPy...")
        import numpy as np
        
        update_splash(0.30, "Loading Numba JIT...")
        # Import numba - this is slow
        from numba import njit
        
        update_splash(0.50, "Loading emulator...")
        from src.emulator import Emulator
        
        update_splash(0.65, "Loading renderer...")
        from src.ppu_fast import PPU_Fast
        
        update_splash(0.75, "Loading GUI...")
        from src.gui import EmulatorGUI
        
        update_splash(0.85, "Creating emulator...")
        emulator = Emulator()
        
        update_splash(0.90, f"Loading ROM...")
        if not emulator.load_rom(rom_path):
            if splash:
                splash.destroy()
            print("Failed to load ROM!")
            return 1
        
        update_splash(0.95, "Loading AI system...")
        from src.agent import AgentManager
        
        update_splash(1.0, "Ready!")
        
        # Close splash
        if splash:
            splash.after(100, splash.destroy)
            splash.mainloop()
        
    except Exception as e:
        if splash:
            splash.destroy()
        print(f"Startup error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Print controls
    print("GBC Emulator Ready!")
    print("Controls: Arrows=D-Pad, Z=A, X=B, Enter=Start, TAB=Turbo, F2=AI")
    
    # Run main GUI
    try:
        gui = EmulatorGUI(emulator, scale=3)
        gui.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

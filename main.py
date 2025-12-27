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

from src.emulator import Emulator
from src.gui import EmulatorGUI, SimplerGUI


def main():
    # Default ROM path
    default_rom = r"E:\ISO's\Dokemon.gbc"
    
    # Get ROM path from args or use default
    if len(sys.argv) > 1:
        rom_path = sys.argv[1]
    else:
        rom_path = default_rom
    
    # Check if ROM exists
    if not os.path.exists(rom_path):
        print(f"ROM file not found: {rom_path}")
        print("\nUsage: python main.py <rom_file>")
        return 1
    
    print("=" * 60)
    print("  GBC Emulator")
    print("=" * 60)
    print()
    
    # Create emulator
    emulator = Emulator()
    
    # Load ROM
    print(f"Loading ROM: {rom_path}")
    if not emulator.load_rom(rom_path):
        print("Failed to load ROM!")
        return 1
    
    print()
    print("Controls:")
    print("  Arrow Keys  - D-Pad")
    print("  Z           - A Button")
    print("  X           - B Button")
    print("  Enter       - Start")
    print("  Right Shift - Select")
    print("  Space       - Pause/Resume")
    print("  R           - Reset")
    print("  T           - Toggle Tilemap (0x9800/0x9C00)")
    print("  V           - Toggle VRAM Bank (GBC)")
    print("  N           - Step (when paused)")
    print("  TAB         - Turbo Mode (fast forward)")
    print("  ESC         - Quit")
    print()
    print("=" * 60)
    
    # Create GUI and run
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


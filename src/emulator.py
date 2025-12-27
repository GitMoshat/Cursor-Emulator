"""
Game Boy Color Emulator Core
Integrates CPU, Memory, and PPU components.
"""

import time
from typing import Optional, Callable
import numpy as np

from .memory import Memory
from .cpu import CPU
from .ppu import PPU

# Note: JIT PPU available in ppu_fast.py if needed
print("Using standard CPU and PPU")


class Emulator:
    """
    GBC Emulator - Main emulation loop and component integration.
    
    Clock speed: 4.194304 MHz (8.388608 MHz in double speed mode)
    Cycles per frame: 70224 (at 59.73 FPS)
    """
    
    CLOCK_SPEED = 4194304  # Hz
    DOUBLE_SPEED = 8388608  # Hz
    CYCLES_PER_FRAME = 70224
    TARGET_FPS = 59.73
    
    def __init__(self):
        self.memory = Memory()
        self.cpu = CPU(self.memory)
        self.ppu = PPU(self.memory)
        
        # Connect components
        self.ppu.on_vblank = self._on_vblank
        
        # State
        self.running = False
        self.paused = False
        self.cgb_mode = False
        
        # Timing
        self.frame_cycles = 0
        self.total_frames = 0
        self.last_frame_time = 0
        
        # Callbacks
        self.on_frame: Optional[Callable] = None
        
        # Debug info
        self.debug_enabled = False
        self.breakpoints = set()
        
        # ROM info
        self.rom_title = ""
        self.rom_loaded = False
    
    def load_rom(self, filepath: str) -> bool:
        """Load a ROM file."""
        try:
            with open(filepath, 'rb') as f:
                rom_data = f.read()
            
            self.cgb_mode = self.memory.load_rom(rom_data)
            
            # Initialize CPU for GBC mode
            if self.cgb_mode:
                self.cpu.init_gbc_mode()
            
            # Extract ROM title (filter out non-printable chars)
            title_bytes = rom_data[0x134:0x144]
            self.rom_title = ''.join(
                chr(b) if 32 <= b < 127 else '' 
                for b in title_bytes
            ).strip()
            
            self.rom_loaded = True
            print(f"Loaded ROM: {self.rom_title}")
            print(f"GBC Mode: {self.cgb_mode}")
            
            return True
            
        except Exception as e:
            print(f"Failed to load ROM: {e}")
            return False
    
    def reset(self):
        """Reset the emulator."""
        self.cpu = CPU(self.memory)
        self.ppu = PPU(self.memory)
        self.ppu.on_vblank = self._on_vblank
        
        if self.cgb_mode:
            self.cpu.init_gbc_mode()
        
        self.frame_cycles = 0
        self.running = False
        self.paused = False
    
    def step(self) -> int:
        """Execute one CPU instruction. Returns cycles consumed."""
        cycles = self.cpu.step()
        
        # Update timer
        self.memory.update_timer(cycles)
        
        # Update PPU
        interrupts = self.ppu.step(cycles)
        
        # Request interrupts
        if interrupts & 0x01:  # V-Blank
            self.cpu.request_interrupt(0)
        if interrupts & 0x02:  # STAT
            self.cpu.request_interrupt(1)
        
        # Timer interrupt
        if int(self.memory.io[0x0F]) & 0x04:
            self.cpu.request_interrupt(2)
            self.memory.io[0x0F] = int(self.memory.io[0x0F]) & 0xFB  # Clear bit 2
        
        self.frame_cycles += cycles
        
        return cycles
    
    def run_frame(self) -> np.ndarray:
        """Run emulation for one frame. Returns the framebuffer."""
        target_cycles = self.CYCLES_PER_FRAME
        
        while self.frame_cycles < target_cycles:
            if self.debug_enabled and self.cpu.pc in self.breakpoints:
                self.paused = True
                break
            
            self.step()
        
        self.frame_cycles -= target_cycles
        self.total_frames += 1
        
        return self.ppu.framebuffer.copy()
    
    def run(self):
        """Main emulation loop."""
        self.running = True
        frame_time = 1.0 / self.TARGET_FPS
        
        while self.running:
            if not self.paused:
                start = time.perf_counter()
                
                frame = self.run_frame()
                
                if self.on_frame:
                    self.on_frame(frame)
                
                # Frame timing
                elapsed = time.perf_counter() - start
                sleep_time = frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
    
    def _on_vblank(self):
        """Called when V-Blank starts."""
        pass
    
    # Input handling
    def press_button(self, button: str):
        """Press a button. Buttons: a, b, start, select, up, down, left, right"""
        button_map = {
            'a': 0, 'b': 1, 'select': 2, 'start': 3,
            'right': 4, 'left': 5, 'up': 6, 'down': 7
        }
        if button.lower() in button_map:
            self.memory.set_button(button_map[button.lower()], True)
            # Request joypad interrupt
            self.cpu.request_interrupt(4)
    
    def release_button(self, button: str):
        """Release a button."""
        button_map = {
            'a': 0, 'b': 1, 'select': 2, 'start': 3,
            'right': 4, 'left': 5, 'up': 6, 'down': 7
        }
        if button.lower() in button_map:
            self.memory.set_button(button_map[button.lower()], False)
    
    # Debug methods
    def get_cpu_state(self) -> dict:
        """Get current CPU state for debugging."""
        return {
            'A': self.cpu.a,
            'F': self.cpu.f,
            'B': self.cpu.b,
            'C': self.cpu.c,
            'D': self.cpu.d,
            'E': self.cpu.e,
            'H': self.cpu.h,
            'L': self.cpu.l,
            'SP': self.cpu.sp,
            'PC': self.cpu.pc,
            'AF': self.cpu.af,
            'BC': self.cpu.bc,
            'DE': self.cpu.de,
            'HL': self.cpu.hl,
            'IME': self.cpu.ime,
            'Halted': self.cpu.halted,
            'Flags': {
                'Z': self.cpu.flag_z,
                'N': self.cpu.flag_n,
                'H': self.cpu.flag_h,
                'C': self.cpu.flag_c,
            }
        }
    
    def get_ppu_state(self) -> dict:
        """Get current PPU state for debugging."""
        return {
            'LY': self.ppu.ly,
            'LYC': self.ppu.lyc,
            'Mode': self.ppu.mode,
            'LCDC': self.ppu.lcdc,
            'STAT': self.ppu.stat,
            'SCX': self.ppu.scx,
            'SCY': self.ppu.scy,
            'WX': self.ppu.wx,
            'WY': self.ppu.wy,
            'BGP': self.ppu.bgp,
            'OBP0': self.ppu.obp0,
            'OBP1': self.ppu.obp1,
            'LCD Enabled': self.ppu.lcd_enabled,
            'Window Enabled': self.ppu.window_enabled,
            'Sprites Enabled': self.ppu.sprites_enabled,
            'BG Enabled': self.ppu.bg_enabled,
        }
    
    def get_memory_dump(self, start: int, length: int) -> bytes:
        """Dump memory region."""
        data = bytearray()
        for i in range(length):
            data.append(self.memory.read(start + i))
        return bytes(data)
    
    def get_tilemap(self, map_select: int = 0) -> np.ndarray:
        """Get tilemap image (0 = 0x9800, 1 = 0x9C00)."""
        addr = 0x9C00 if map_select else 0x9800
        return self.ppu.get_tilemap_image(addr)
    
    def get_tiles(self, bank: int = 0) -> np.ndarray:
        """Get all tiles image."""
        return self.ppu.get_tiles_image(bank)


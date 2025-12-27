"""
Game Boy Color Memory Management Unit
Handles memory mapping, banking, and I/O registers.
"""

import numpy as np
from typing import Optional, Callable, List


class MBC:
    """Base Memory Bank Controller class."""
    
    def __init__(self, rom_data: bytes, ram_size: int = 0):
        self.rom = np.frombuffer(rom_data, dtype=np.uint8).copy()
        self.rom_bank = 1
        self.ram_bank = 0
        self.ram_enabled = False
        
        # External RAM
        if ram_size > 0:
            self.ram = np.zeros(ram_size, dtype=np.uint8)
        else:
            self.ram = np.zeros(0x2000, dtype=np.uint8)  # Default 8KB
    
    def read_rom(self, addr: int) -> int:
        if addr < 0x4000:
            return self.rom[addr]
        else:
            bank_addr = (self.rom_bank * 0x4000) + (addr - 0x4000)
            if bank_addr < len(self.rom):
                return self.rom[bank_addr]
            return 0xFF
    
    def read_ram(self, addr: int) -> int:
        if self.ram_enabled and len(self.ram) > 0:
            ram_addr = (self.ram_bank * 0x2000) + (addr - 0xA000)
            if ram_addr < len(self.ram):
                return self.ram[ram_addr]
        return 0xFF
    
    def write_rom(self, addr: int, value: int):
        """Handle writes to ROM area (for MBC control)."""
        pass
    
    def write_ram(self, addr: int, value: int):
        if self.ram_enabled and len(self.ram) > 0:
            ram_addr = (self.ram_bank * 0x2000) + (addr - 0xA000)
            if ram_addr < len(self.ram):
                self.ram[ram_addr] = value


class MBC_None(MBC):
    """No MBC - 32KB ROM only."""
    pass


class MBC1(MBC):
    """MBC1 - Up to 2MB ROM, 32KB RAM."""
    
    def __init__(self, rom_data: bytes, ram_size: int = 0):
        super().__init__(rom_data, ram_size)
        self.mode = 0  # 0 = ROM mode, 1 = RAM mode
        self.rom_bank_low = 1
        self.rom_bank_high = 0
    
    def write_rom(self, addr: int, value: int):
        if addr < 0x2000:
            # RAM Enable
            self.ram_enabled = (value & 0x0F) == 0x0A
        elif addr < 0x4000:
            # ROM Bank Number (lower 5 bits)
            self.rom_bank_low = value & 0x1F
            if self.rom_bank_low == 0:
                self.rom_bank_low = 1
            self._update_banks()
        elif addr < 0x6000:
            # RAM Bank / Upper ROM Bank
            self.rom_bank_high = value & 0x03
            self._update_banks()
        else:
            # Banking Mode
            self.mode = value & 0x01
            self._update_banks()
    
    def _update_banks(self):
        if self.mode == 0:
            # ROM Banking Mode
            self.rom_bank = (self.rom_bank_high << 5) | self.rom_bank_low
            self.ram_bank = 0
        else:
            # RAM Banking Mode
            self.rom_bank = self.rom_bank_low
            self.ram_bank = self.rom_bank_high


class MBC2(MBC):
    """MBC2 - Up to 256KB ROM, 512x4 bits RAM."""
    
    def __init__(self, rom_data: bytes, ram_size: int = 512):
        super().__init__(rom_data, ram_size)
        self.ram = np.zeros(512, dtype=np.uint8)
    
    def write_rom(self, addr: int, value: int):
        if addr < 0x4000:
            if addr & 0x0100:
                # ROM Bank
                self.rom_bank = value & 0x0F
                if self.rom_bank == 0:
                    self.rom_bank = 1
            else:
                # RAM Enable
                self.ram_enabled = (value & 0x0F) == 0x0A
    
    def read_ram(self, addr: int) -> int:
        if self.ram_enabled:
            return self.ram[(addr - 0xA000) & 0x1FF] | 0xF0
        return 0xFF
    
    def write_ram(self, addr: int, value: int):
        if self.ram_enabled:
            self.ram[(addr - 0xA000) & 0x1FF] = value & 0x0F


class MBC3(MBC):
    """MBC3 - Up to 2MB ROM, 32KB RAM, RTC."""
    
    def __init__(self, rom_data: bytes, ram_size: int = 0x8000):
        super().__init__(rom_data, ram_size)
        self.rtc_registers = [0] * 5  # S, M, H, DL, DH
        self.rtc_latched = [0] * 5
        self.rtc_latch_state = 0
        self.ram_rtc_select = 0
    
    def write_rom(self, addr: int, value: int):
        if addr < 0x2000:
            self.ram_enabled = (value & 0x0F) == 0x0A
        elif addr < 0x4000:
            self.rom_bank = value & 0x7F
            if self.rom_bank == 0:
                self.rom_bank = 1
        elif addr < 0x6000:
            self.ram_rtc_select = value
            if value <= 0x03:
                self.ram_bank = value
        else:
            # RTC Latch
            if self.rtc_latch_state == 0 and value == 0x00:
                self.rtc_latch_state = 1
            elif self.rtc_latch_state == 1 and value == 0x01:
                self.rtc_latched = self.rtc_registers.copy()
                self.rtc_latch_state = 0
    
    def read_ram(self, addr: int) -> int:
        if not self.ram_enabled:
            return 0xFF
        if self.ram_rtc_select >= 0x08 and self.ram_rtc_select <= 0x0C:
            return self.rtc_latched[self.ram_rtc_select - 0x08]
        return super().read_ram(addr)
    
    def write_ram(self, addr: int, value: int):
        if not self.ram_enabled:
            return
        if self.ram_rtc_select >= 0x08 and self.ram_rtc_select <= 0x0C:
            self.rtc_registers[self.ram_rtc_select - 0x08] = value
        else:
            super().write_ram(addr, value)


class MBC5(MBC):
    """MBC5 - Up to 8MB ROM, 128KB RAM."""
    
    def __init__(self, rom_data: bytes, ram_size: int = 0x20000):
        super().__init__(rom_data, ram_size)
        self.rom_bank_low = 1
        self.rom_bank_high = 0
    
    def write_rom(self, addr: int, value: int):
        if addr < 0x2000:
            self.ram_enabled = (value & 0x0F) == 0x0A
        elif addr < 0x3000:
            self.rom_bank_low = value
            self.rom_bank = (self.rom_bank_high << 8) | self.rom_bank_low
        elif addr < 0x4000:
            self.rom_bank_high = value & 0x01
            self.rom_bank = (self.rom_bank_high << 8) | self.rom_bank_low
        elif addr < 0x6000:
            self.ram_bank = value & 0x0F


class Memory:
    """
    Game Boy Color Memory Management Unit.
    
    Memory Map:
    0x0000-0x3FFF: ROM Bank 0 (16KB)
    0x4000-0x7FFF: ROM Bank N (16KB, switchable)
    0x8000-0x9FFF: VRAM (8KB, bank switchable on GBC)
    0xA000-0xBFFF: External RAM (8KB, bank switchable)
    0xC000-0xCFFF: WRAM Bank 0 (4KB)
    0xD000-0xDFFF: WRAM Bank N (4KB, bank switchable on GBC)
    0xE000-0xFDFF: Echo RAM (mirror of C000-DDFF)
    0xFE00-0xFE9F: OAM (Sprite Attribute Table)
    0xFEA0-0xFEFF: Not usable
    0xFF00-0xFF7F: I/O Registers
    0xFF80-0xFFFE: HRAM (High RAM)
    0xFFFF: Interrupt Enable Register
    """
    
    def __init__(self):
        # VRAM (2 banks for GBC)
        self.vram = np.zeros((2, 0x2000), dtype=np.uint8)
        self.vram_bank = 0
        
        # WRAM (8 banks for GBC, bank 0 fixed)
        self.wram = np.zeros((8, 0x1000), dtype=np.uint8)
        self.wram_bank = 1
        
        # OAM
        self.oam = np.zeros(0xA0, dtype=np.uint8)
        
        # HRAM
        self.hram = np.zeros(0x7F, dtype=np.uint8)
        
        # I/O Registers
        self.io = np.zeros(0x80, dtype=np.uint8)
        
        # Interrupt Enable
        self.ie = 0x00
        
        # MBC
        self.mbc: Optional[MBC] = None
        
        # GBC mode
        self.cgb_mode = False
        
        # GBC Palette RAM
        self.bg_palette_ram = np.zeros(64, dtype=np.uint8)
        self.obj_palette_ram = np.zeros(64, dtype=np.uint8)
        self.bg_palette_index = 0
        self.obj_palette_index = 0
        self.bg_palette_auto_inc = False
        self.obj_palette_auto_inc = False
        
        # DMA
        self.dma_source = 0
        self.dma_dest = 0
        self.dma_length = 0
        self.hdma_active = False
        self.hdma_mode = 0  # 0 = GDMA, 1 = HDMA
        
        # Joypad
        self.joypad_select = 0x30
        self.joypad_state = 0xFF  # All buttons released
        
        # Timer
        self.div_counter = 0
        self.tima_counter = 0
        
        # Callbacks
        self.on_vram_write: Optional[Callable] = None
        self.on_oam_write: Optional[Callable] = None
    
    def load_rom(self, rom_data: bytes):
        """Load a ROM and initialize the appropriate MBC."""
        if len(rom_data) < 0x150:
            raise ValueError("Invalid ROM: too small")
        
        # Read cartridge header
        cart_type = rom_data[0x147]
        rom_size_code = rom_data[0x148]
        ram_size_code = rom_data[0x149]
        
        # Calculate sizes
        rom_size = 0x8000 << rom_size_code
        ram_sizes = {0: 0, 1: 0x800, 2: 0x2000, 3: 0x8000, 4: 0x20000, 5: 0x10000}
        ram_size = ram_sizes.get(ram_size_code, 0)
        
        # Check for GBC support
        cgb_flag = rom_data[0x143]
        self.cgb_mode = (cgb_flag & 0x80) != 0
        
        # Create appropriate MBC
        mbc_types = {
            0x00: MBC_None,
            0x01: MBC1, 0x02: MBC1, 0x03: MBC1,
            0x05: MBC2, 0x06: MBC2,
            0x0F: MBC3, 0x10: MBC3, 0x11: MBC3, 0x12: MBC3, 0x13: MBC3,
            0x19: MBC5, 0x1A: MBC5, 0x1B: MBC5, 0x1C: MBC5, 0x1D: MBC5, 0x1E: MBC5,
        }
        
        mbc_class = mbc_types.get(cart_type, MBC5)
        self.mbc = mbc_class(rom_data, ram_size)
        
        # Initialize I/O registers
        self._init_io()
        
        return self.cgb_mode
    
    def _init_io(self):
        """Initialize I/O registers to boot values."""
        self.io[0x00] = 0xCF  # P1 (Joypad)
        self.io[0x01] = 0x00  # SB (Serial transfer data)
        self.io[0x02] = 0x7E  # SC (Serial transfer control)
        self.io[0x04] = 0xAB  # DIV
        self.io[0x05] = 0x00  # TIMA
        self.io[0x06] = 0x00  # TMA
        self.io[0x07] = 0xF8  # TAC
        self.io[0x0F] = 0xE1  # IF
        self.io[0x10] = 0x80  # NR10
        self.io[0x11] = 0xBF  # NR11
        self.io[0x12] = 0xF3  # NR12
        self.io[0x13] = 0xFF  # NR13
        self.io[0x14] = 0xBF  # NR14
        self.io[0x16] = 0x3F  # NR21
        self.io[0x17] = 0x00  # NR22
        self.io[0x18] = 0xFF  # NR23
        self.io[0x19] = 0xBF  # NR24
        self.io[0x1A] = 0x7F  # NR30
        self.io[0x1B] = 0xFF  # NR31
        self.io[0x1C] = 0x9F  # NR32
        self.io[0x1D] = 0xFF  # NR33
        self.io[0x1E] = 0xBF  # NR34
        self.io[0x20] = 0xFF  # NR41
        self.io[0x21] = 0x00  # NR42
        self.io[0x22] = 0x00  # NR43
        self.io[0x23] = 0xBF  # NR44
        self.io[0x24] = 0x77  # NR50
        self.io[0x25] = 0xF3  # NR51
        self.io[0x26] = 0xF1  # NR52
        self.io[0x40] = 0x91  # LCDC
        self.io[0x41] = 0x85  # STAT
        self.io[0x42] = 0x00  # SCY
        self.io[0x43] = 0x00  # SCX
        self.io[0x44] = 0x00  # LY
        self.io[0x45] = 0x00  # LYC
        self.io[0x46] = 0xFF  # DMA
        self.io[0x47] = 0xFC  # BGP
        self.io[0x48] = 0x00  # OBP0
        self.io[0x49] = 0x00  # OBP1
        self.io[0x4A] = 0x00  # WY
        self.io[0x4B] = 0x00  # WX
        
        if self.cgb_mode:
            self.io[0x4D] = 0x7E  # KEY1 (Speed switch)
            self.io[0x4F] = 0xFE  # VBK (VRAM bank)
            self.io[0x51] = 0xFF  # HDMA1
            self.io[0x52] = 0xFF  # HDMA2
            self.io[0x53] = 0xFF  # HDMA3
            self.io[0x54] = 0xFF  # HDMA4
            self.io[0x55] = 0xFF  # HDMA5
            self.io[0x68] = 0xC0  # BCPS
            self.io[0x69] = 0xFF  # BCPD
            self.io[0x6A] = 0xC1  # OCPS
            self.io[0x6B] = 0xFF  # OCPD
            self.io[0x70] = 0xF8  # SVBK (WRAM bank)
    
    def read(self, addr: int) -> int:
        """Read a byte from memory."""
        addr = int(addr) & 0xFFFF
        
        # ROM
        if addr < 0x8000:
            if self.mbc:
                return self.mbc.read_rom(addr)
            return 0xFF
        
        # VRAM
        if addr < 0xA000:
            return self.vram[self.vram_bank][addr - 0x8000]
        
        # External RAM
        if addr < 0xC000:
            if self.mbc:
                return self.mbc.read_ram(addr)
            return 0xFF
        
        # WRAM Bank 0
        if addr < 0xD000:
            return self.wram[0][addr - 0xC000]
        
        # WRAM Bank N
        if addr < 0xE000:
            return self.wram[self.wram_bank][addr - 0xD000]
        
        # Echo RAM
        if addr < 0xFE00:
            return self.read(addr - 0x2000)
        
        # OAM
        if addr < 0xFEA0:
            return self.oam[addr - 0xFE00]
        
        # Not usable
        if addr < 0xFF00:
            return 0xFF
        
        # I/O Registers
        if addr < 0xFF80:
            return self._read_io(addr)
        
        # HRAM
        if addr < 0xFFFF:
            return self.hram[addr - 0xFF80]
        
        # IE Register
        return self.ie
    
    def write(self, addr: int, value: int):
        """Write a byte to memory."""
        addr = int(addr) & 0xFFFF
        value = int(value) & 0xFF
        
        # ROM (MBC control)
        if addr < 0x8000:
            if self.mbc:
                self.mbc.write_rom(addr, value)
            return
        
        # VRAM
        if addr < 0xA000:
            self.vram[self.vram_bank][addr - 0x8000] = value
            if self.on_vram_write:
                self.on_vram_write(addr, value)
            return
        
        # External RAM
        if addr < 0xC000:
            if self.mbc:
                self.mbc.write_ram(addr, value)
            return
        
        # WRAM Bank 0
        if addr < 0xD000:
            self.wram[0][addr - 0xC000] = value
            return
        
        # WRAM Bank N
        if addr < 0xE000:
            self.wram[self.wram_bank][addr - 0xD000] = value
            return
        
        # Echo RAM
        if addr < 0xFE00:
            self.write(addr - 0x2000, value)
            return
        
        # OAM
        if addr < 0xFEA0:
            self.oam[addr - 0xFE00] = value
            if self.on_oam_write:
                self.on_oam_write(addr, value)
            return
        
        # Not usable
        if addr < 0xFF00:
            return
        
        # I/O Registers
        if addr < 0xFF80:
            self._write_io(addr, value)
            return
        
        # HRAM
        if addr < 0xFFFF:
            self.hram[addr - 0xFF80] = value
            return
        
        # IE Register
        self.ie = value
    
    def _read_io(self, addr: int) -> int:
        """Read from I/O registers."""
        reg = addr & 0x7F
        
        # Joypad
        if reg == 0x00:
            return self._read_joypad()
        
        # GBC Palette data
        if reg == 0x69:  # BCPD
            return self.bg_palette_ram[self.bg_palette_index]
        if reg == 0x6B:  # OCPD
            return self.obj_palette_ram[self.obj_palette_index]
        
        return self.io[reg]
    
    def _write_io(self, addr: int, value: int):
        """Write to I/O registers."""
        reg = addr & 0x7F
        
        # Joypad
        if reg == 0x00:
            self.joypad_select = value & 0x30
            return
        
        # DIV - Writing any value resets it
        if reg == 0x04:
            self.io[0x04] = 0
            self.div_counter = 0
            return
        
        # DMA
        if reg == 0x46:
            self._do_oam_dma(value)
            return
        
        # GBC VRAM Bank
        if reg == 0x4F and self.cgb_mode:
            self.vram_bank = value & 0x01
            self.io[0x4F] = value | 0xFE
            return
        
        # GBC WRAM Bank
        if reg == 0x70 and self.cgb_mode:
            self.wram_bank = value & 0x07
            if self.wram_bank == 0:
                self.wram_bank = 1
            self.io[0x70] = value | 0xF8
            return
        
        # GBC Palette index
        if reg == 0x68:  # BCPS
            self.bg_palette_index = value & 0x3F
            self.bg_palette_auto_inc = (value & 0x80) != 0
            self.io[0x68] = value | 0x40
            return
        if reg == 0x6A:  # OCPS
            self.obj_palette_index = value & 0x3F
            self.obj_palette_auto_inc = (value & 0x80) != 0
            self.io[0x6A] = value | 0x40
            return
        
        # GBC Palette data
        if reg == 0x69:  # BCPD
            self.bg_palette_ram[self.bg_palette_index] = value
            if self.bg_palette_auto_inc:
                self.bg_palette_index = (self.bg_palette_index + 1) & 0x3F
            return
        if reg == 0x6B:  # OCPD
            self.obj_palette_ram[self.obj_palette_index] = value
            if self.obj_palette_auto_inc:
                self.obj_palette_index = (self.obj_palette_index + 1) & 0x3F
            return
        
        # GBC HDMA
        if reg == 0x55 and self.cgb_mode:
            self._start_hdma(value)
            return
        
        self.io[reg] = value
    
    def _read_joypad(self) -> int:
        """Read joypad state based on selection."""
        result = self.joypad_select | 0x0F
        
        if not (self.joypad_select & 0x10):
            # Direction keys
            result &= (self.joypad_state >> 4) | 0xF0
        if not (self.joypad_select & 0x20):
            # Button keys
            result &= (self.joypad_state & 0x0F) | 0xF0
        
        return result
    
    def set_button(self, button: int, pressed: bool):
        """
        Set button state.
        Buttons: 0=A, 1=B, 2=Select, 3=Start, 4=Right, 5=Left, 6=Up, 7=Down
        """
        if pressed:
            self.joypad_state &= ~(1 << button)
        else:
            self.joypad_state |= (1 << button)
    
    def _do_oam_dma(self, value: int):
        """Perform OAM DMA transfer."""
        source = value << 8
        for i in range(0xA0):
            self.oam[i] = self.read(source + i)
        self.io[0x46] = value
    
    def _start_hdma(self, value: int):
        """Start GBC HDMA transfer."""
        if self.hdma_active and (value & 0x80) == 0:
            # Cancel HDMA
            self.hdma_active = False
            self.io[0x55] = value | 0x80
            return
        
        source = ((int(self.io[0x51]) << 8) | int(self.io[0x52])) & 0xFFF0
        dest = ((int(self.io[0x53]) << 8) | int(self.io[0x54])) & 0x1FF0
        length = ((value & 0x7F) + 1) * 16
        
        if value & 0x80:
            # HDMA (H-Blank DMA)
            self.hdma_active = True
            self.hdma_mode = 1
            self.dma_source = source
            self.dma_dest = dest
            self.dma_length = length
            self.io[0x55] = value & 0x7F
        else:
            # GDMA (General DMA)
            for i in range(length):
                self.vram[self.vram_bank][(dest + i) & 0x1FFF] = self.read(source + i)
            self.io[0x55] = 0xFF
    
    def do_hdma_transfer(self):
        """Perform one HDMA block transfer (called during H-Blank)."""
        if not self.hdma_active:
            return
        
        # Transfer 16 bytes
        for i in range(16):
            self.vram[self.vram_bank][(self.dma_dest + i) & 0x1FFF] = self.read(self.dma_source + i)
        
        self.dma_source += 16
        self.dma_dest += 16
        self.dma_length -= 16
        
        if self.dma_length <= 0:
            self.hdma_active = False
            self.io[0x55] = 0xFF
        else:
            self.io[0x55] = ((self.dma_length // 16) - 1) & 0x7F
    
    def update_timer(self, cycles: int):
        """Update timer registers."""
        # DIV increments every 256 cycles
        self.div_counter += cycles
        while self.div_counter >= 256:
            self.div_counter -= 256
            self.io[0x04] = (int(self.io[0x04]) + 1) & 0xFF
        
        # TIMA
        tac = int(self.io[0x07])
        if tac & 0x04:  # Timer enabled
            freq_dividers = [1024, 16, 64, 256]
            divider = freq_dividers[tac & 0x03]
            
            self.tima_counter += cycles
            while self.tima_counter >= divider:
                self.tima_counter -= divider
                tima = (int(self.io[0x05]) + 1) & 0xFF
                self.io[0x05] = tima
                if tima == 0:  # Overflow
                    self.io[0x05] = self.io[0x06]  # Reset to TMA
                    self.io[0x0F] = int(self.io[0x0F]) | 0x04  # Request timer interrupt


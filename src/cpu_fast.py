"""
JIT-Compiled Game Boy Color CPU using Numba.
Achieves near-native speed by compiling the hot instruction loop.
"""

import numpy as np
from numba import njit, uint8, uint16, int8, boolean
from numba.core.types import UniTuple
from typing import Tuple


# Register indices in the state array
REG_A = 0
REG_F = 1
REG_B = 2
REG_C = 3
REG_D = 4
REG_E = 5
REG_H = 6
REG_L = 7
REG_SP_LO = 8
REG_SP_HI = 9
REG_PC_LO = 10
REG_PC_HI = 11

# State flags indices
STATE_IME = 0
STATE_HALTED = 1
STATE_IME_SCHEDULED = 2
STATE_DOUBLE_SPEED = 3

# Flag bit positions
FLAG_Z = 0x80
FLAG_N = 0x40
FLAG_H = 0x20
FLAG_C = 0x10


@njit(cache=True)
def get_af(regs):
    return (regs[REG_A] << 8) | (regs[REG_F] & 0xF0)

@njit(cache=True)
def set_af(regs, value):
    regs[REG_A] = (value >> 8) & 0xFF
    regs[REG_F] = value & 0xF0

@njit(cache=True)
def get_bc(regs):
    return (regs[REG_B] << 8) | regs[REG_C]

@njit(cache=True)
def set_bc(regs, value):
    regs[REG_B] = (value >> 8) & 0xFF
    regs[REG_C] = value & 0xFF

@njit(cache=True)
def get_de(regs):
    return (regs[REG_D] << 8) | regs[REG_E]

@njit(cache=True)
def set_de(regs, value):
    regs[REG_D] = (value >> 8) & 0xFF
    regs[REG_E] = value & 0xFF

@njit(cache=True)
def get_hl(regs):
    return (regs[REG_H] << 8) | regs[REG_L]

@njit(cache=True)
def set_hl(regs, value):
    regs[REG_H] = (value >> 8) & 0xFF
    regs[REG_L] = value & 0xFF

@njit(cache=True)
def get_sp(regs):
    return (regs[REG_SP_HI] << 8) | regs[REG_SP_LO]

@njit(cache=True)
def set_sp(regs, value):
    regs[REG_SP_LO] = value & 0xFF
    regs[REG_SP_HI] = (value >> 8) & 0xFF

@njit(cache=True)
def get_pc(regs):
    return (regs[REG_PC_HI] << 8) | regs[REG_PC_LO]

@njit(cache=True)
def set_pc(regs, value):
    regs[REG_PC_LO] = value & 0xFF
    regs[REG_PC_HI] = (value >> 8) & 0xFF

@njit(cache=True)
def get_flag_z(regs):
    return (regs[REG_F] & FLAG_Z) != 0

@njit(cache=True)
def get_flag_n(regs):
    return (regs[REG_F] & FLAG_N) != 0

@njit(cache=True)
def get_flag_h(regs):
    return (regs[REG_F] & FLAG_H) != 0

@njit(cache=True)
def get_flag_c(regs):
    return (regs[REG_F] & FLAG_C) != 0

@njit(cache=True)
def set_flag(regs, flag, value):
    if value:
        regs[REG_F] |= flag
    else:
        regs[REG_F] &= ~flag & 0xFF
    regs[REG_F] &= 0xF0


@njit(cache=True)
def read_mem(rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank, 
             wram, wram_bank, oam, io, hram, ie, addr):
    """Read a byte from memory."""
    addr = addr & 0xFFFF
    
    if addr < 0x4000:
        # ROM bank 0
        if addr < len(rom):
            return rom[addr]
        return 0xFF
    elif addr < 0x8000:
        # ROM bank N
        bank_addr = rom_bank * 0x4000 + (addr - 0x4000)
        if bank_addr < len(rom):
            return rom[bank_addr]
        return 0xFF
    elif addr < 0xA000:
        # VRAM
        return vram[vram_bank, addr - 0x8000]
    elif addr < 0xC000:
        # External RAM
        if ram_enabled and len(ram) > 0:
            ram_addr = ram_bank * 0x2000 + (addr - 0xA000)
            if ram_addr < len(ram):
                return ram[ram_addr]
        return 0xFF
    elif addr < 0xD000:
        # WRAM bank 0
        return wram[0, addr - 0xC000]
    elif addr < 0xE000:
        # WRAM bank N
        return wram[wram_bank, addr - 0xD000]
    elif addr < 0xFE00:
        # Echo RAM
        return read_mem(rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                       wram, wram_bank, oam, io, hram, ie, addr - 0x2000)
    elif addr < 0xFEA0:
        # OAM
        return oam[addr - 0xFE00]
    elif addr < 0xFF00:
        # Not usable
        return 0xFF
    elif addr < 0xFF80:
        # I/O registers
        return io[addr - 0xFF00]
    elif addr < 0xFFFF:
        # HRAM
        return hram[addr - 0xFF80]
    else:
        # IE register
        return ie[0]


@njit(cache=True)
def write_mem(rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr, 
              vram, vram_bank, wram, wram_bank, oam, io, hram, ie, 
              mbc_type, addr, value):
    """Write a byte to memory. Returns updated bank values."""
    addr = addr & 0xFFFF
    value = value & 0xFF
    
    rom_bank = rom_bank_ptr[0]
    ram_bank = ram_bank_ptr[0]
    ram_enabled = ram_enabled_ptr[0]
    
    if addr < 0x2000:
        # RAM enable (MBC)
        ram_enabled_ptr[0] = (value & 0x0F) == 0x0A
    elif addr < 0x4000:
        # ROM bank select
        if mbc_type == 1:  # MBC1
            bank = value & 0x1F
            if bank == 0:
                bank = 1
            rom_bank_ptr[0] = bank
        elif mbc_type == 5:  # MBC5
            rom_bank_ptr[0] = value
    elif addr < 0x6000:
        # RAM bank select
        if mbc_type == 1:
            ram_bank_ptr[0] = value & 0x03
        elif mbc_type == 5:
            ram_bank_ptr[0] = value & 0x0F
    elif addr < 0x8000:
        # MBC mode select
        pass
    elif addr < 0xA000:
        # VRAM
        vram[vram_bank, addr - 0x8000] = value
    elif addr < 0xC000:
        # External RAM
        if ram_enabled_ptr[0] and len(ram) > 0:
            ram_addr = ram_bank_ptr[0] * 0x2000 + (addr - 0xA000)
            if ram_addr < len(ram):
                ram[ram_addr] = value
    elif addr < 0xD000:
        # WRAM bank 0
        wram[0, addr - 0xC000] = value
    elif addr < 0xE000:
        # WRAM bank N
        wram[wram_bank, addr - 0xD000] = value
    elif addr < 0xFE00:
        # Echo RAM
        write_mem(rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                 vram, vram_bank, wram, wram_bank, oam, io, hram, ie,
                 mbc_type, addr - 0x2000, value)
    elif addr < 0xFEA0:
        # OAM
        oam[addr - 0xFE00] = value
    elif addr < 0xFF00:
        # Not usable
        pass
    elif addr < 0xFF80:
        # I/O registers
        io[addr - 0xFF00] = value
    elif addr < 0xFFFF:
        # HRAM
        hram[addr - 0xFF80] = value
    else:
        # IE register
        ie[0] = value


@njit(cache=True)
def fetch_byte(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
               wram, wram_bank, oam, io, hram, ie):
    """Fetch a byte from PC and increment PC."""
    pc = get_pc(regs)
    value = read_mem(rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                    wram, wram_bank, oam, io, hram, ie, pc)
    set_pc(regs, (pc + 1) & 0xFFFF)
    return value


@njit(cache=True)
def fetch_word(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
               wram, wram_bank, oam, io, hram, ie):
    """Fetch a 16-bit word from PC."""
    lo = fetch_byte(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                   wram, wram_bank, oam, io, hram, ie)
    hi = fetch_byte(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                   wram, wram_bank, oam, io, hram, ie)
    return (hi << 8) | lo


@njit(cache=True)
def push_word(regs, rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
              vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, value):
    """Push a 16-bit value to stack."""
    sp = get_sp(regs) - 2
    set_sp(regs, sp & 0xFFFF)
    write_mem(rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
             vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, sp, value & 0xFF)
    write_mem(rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
             vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, sp + 1, (value >> 8) & 0xFF)


@njit(cache=True)
def pop_word(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
             wram, wram_bank, oam, io, hram, ie):
    """Pop a 16-bit value from stack."""
    sp = get_sp(regs)
    lo = read_mem(rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                 wram, wram_bank, oam, io, hram, ie, sp)
    hi = read_mem(rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                 wram, wram_bank, oam, io, hram, ie, sp + 1)
    set_sp(regs, (sp + 2) & 0xFFFF)
    return (hi << 8) | lo


# ALU Operations
@njit(cache=True)
def alu_add(regs, value):
    a = regs[REG_A]
    result = a + value
    regs[REG_F] = 0
    if (result & 0xFF) == 0:
        regs[REG_F] |= FLAG_Z
    if ((a & 0x0F) + (value & 0x0F)) > 0x0F:
        regs[REG_F] |= FLAG_H
    if result > 0xFF:
        regs[REG_F] |= FLAG_C
    regs[REG_A] = result & 0xFF

@njit(cache=True)
def alu_adc(regs, value):
    a = regs[REG_A]
    carry = 1 if (regs[REG_F] & FLAG_C) else 0
    result = a + value + carry
    regs[REG_F] = 0
    if (result & 0xFF) == 0:
        regs[REG_F] |= FLAG_Z
    if ((a & 0x0F) + (value & 0x0F) + carry) > 0x0F:
        regs[REG_F] |= FLAG_H
    if result > 0xFF:
        regs[REG_F] |= FLAG_C
    regs[REG_A] = result & 0xFF

@njit(cache=True)
def alu_sub(regs, value):
    a = regs[REG_A]
    result = a - value
    regs[REG_F] = FLAG_N
    if (result & 0xFF) == 0:
        regs[REG_F] |= FLAG_Z
    if (a & 0x0F) < (value & 0x0F):
        regs[REG_F] |= FLAG_H
    if result < 0:
        regs[REG_F] |= FLAG_C
    regs[REG_A] = result & 0xFF

@njit(cache=True)
def alu_sbc(regs, value):
    a = regs[REG_A]
    carry = 1 if (regs[REG_F] & FLAG_C) else 0
    result = a - value - carry
    regs[REG_F] = FLAG_N
    if (result & 0xFF) == 0:
        regs[REG_F] |= FLAG_Z
    if (a & 0x0F) < (value & 0x0F) + carry:
        regs[REG_F] |= FLAG_H
    if result < 0:
        regs[REG_F] |= FLAG_C
    regs[REG_A] = result & 0xFF

@njit(cache=True)
def alu_and(regs, value):
    regs[REG_A] &= value
    regs[REG_F] = FLAG_H
    if regs[REG_A] == 0:
        regs[REG_F] |= FLAG_Z

@njit(cache=True)
def alu_xor(regs, value):
    regs[REG_A] ^= value
    regs[REG_F] = 0
    if regs[REG_A] == 0:
        regs[REG_F] |= FLAG_Z

@njit(cache=True)
def alu_or(regs, value):
    regs[REG_A] |= value
    regs[REG_F] = 0
    if regs[REG_A] == 0:
        regs[REG_F] |= FLAG_Z

@njit(cache=True)
def alu_cp(regs, value):
    a = regs[REG_A]
    result = a - value
    regs[REG_F] = FLAG_N
    if (result & 0xFF) == 0:
        regs[REG_F] |= FLAG_Z
    if (a & 0x0F) < (value & 0x0F):
        regs[REG_F] |= FLAG_H
    if result < 0:
        regs[REG_F] |= FLAG_C

@njit(cache=True)
def alu_inc(regs, value):
    result = (value + 1) & 0xFF
    regs[REG_F] = (regs[REG_F] & FLAG_C)
    if result == 0:
        regs[REG_F] |= FLAG_Z
    if (value & 0x0F) == 0x0F:
        regs[REG_F] |= FLAG_H
    return result

@njit(cache=True)
def alu_dec(regs, value):
    result = (value - 1) & 0xFF
    regs[REG_F] = (regs[REG_F] & FLAG_C) | FLAG_N
    if result == 0:
        regs[REG_F] |= FLAG_Z
    if (value & 0x0F) == 0x00:
        regs[REG_F] |= FLAG_H
    return result


# CB prefix operations
@njit(cache=True)
def cb_rlc(regs, value):
    carry = (value >> 7) & 1
    result = ((value << 1) | carry) & 0xFF
    regs[REG_F] = 0
    if result == 0:
        regs[REG_F] |= FLAG_Z
    if carry:
        regs[REG_F] |= FLAG_C
    return result

@njit(cache=True)
def cb_rrc(regs, value):
    carry = value & 1
    result = ((value >> 1) | (carry << 7)) & 0xFF
    regs[REG_F] = 0
    if result == 0:
        regs[REG_F] |= FLAG_Z
    if carry:
        regs[REG_F] |= FLAG_C
    return result

@njit(cache=True)
def cb_rl(regs, value):
    old_carry = 1 if (regs[REG_F] & FLAG_C) else 0
    carry = (value >> 7) & 1
    result = ((value << 1) | old_carry) & 0xFF
    regs[REG_F] = 0
    if result == 0:
        regs[REG_F] |= FLAG_Z
    if carry:
        regs[REG_F] |= FLAG_C
    return result

@njit(cache=True)
def cb_rr(regs, value):
    old_carry = 0x80 if (regs[REG_F] & FLAG_C) else 0
    carry = value & 1
    result = ((value >> 1) | old_carry) & 0xFF
    regs[REG_F] = 0
    if result == 0:
        regs[REG_F] |= FLAG_Z
    if carry:
        regs[REG_F] |= FLAG_C
    return result

@njit(cache=True)
def cb_sla(regs, value):
    carry = (value >> 7) & 1
    result = (value << 1) & 0xFF
    regs[REG_F] = 0
    if result == 0:
        regs[REG_F] |= FLAG_Z
    if carry:
        regs[REG_F] |= FLAG_C
    return result

@njit(cache=True)
def cb_sra(regs, value):
    carry = value & 1
    result = ((value >> 1) | (value & 0x80)) & 0xFF
    regs[REG_F] = 0
    if result == 0:
        regs[REG_F] |= FLAG_Z
    if carry:
        regs[REG_F] |= FLAG_C
    return result

@njit(cache=True)
def cb_swap(regs, value):
    result = ((value >> 4) | (value << 4)) & 0xFF
    regs[REG_F] = 0
    if result == 0:
        regs[REG_F] |= FLAG_Z
    return result

@njit(cache=True)
def cb_srl(regs, value):
    carry = value & 1
    result = (value >> 1) & 0xFF
    regs[REG_F] = 0
    if result == 0:
        regs[REG_F] |= FLAG_Z
    if carry:
        regs[REG_F] |= FLAG_C
    return result

@njit(cache=True)
def cb_bit(regs, bit, value):
    regs[REG_F] = (regs[REG_F] & FLAG_C) | FLAG_H
    if (value & (1 << bit)) == 0:
        regs[REG_F] |= FLAG_Z


@njit(cache=True)
def execute_instruction(regs, state, rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                        vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type):
    """Execute one CPU instruction. Returns cycles consumed."""
    
    rom_bank = rom_bank_ptr[0]
    ram_bank = ram_bank_ptr[0]
    ram_enabled = ram_enabled_ptr[0]
    
    # Handle IME scheduling
    if state[STATE_IME_SCHEDULED]:
        state[STATE_IME] = 1
        state[STATE_IME_SCHEDULED] = 0
    
    # Handle HALT
    if state[STATE_HALTED]:
        ie_val = ie[0]
        if_val = io[0x0F]
        if ie_val & if_val & 0x1F:
            state[STATE_HALTED] = 0
        else:
            return 4
    
    # Fetch opcode
    opcode = fetch_byte(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                       wram, wram_bank, oam, io, hram, ie)
    
    cycles = 4  # Default cycle count
    
    # Decode and execute
    if opcode == 0x00:  # NOP
        cycles = 4
    
    # LD r16, nn
    elif opcode == 0x01:  # LD BC, nn
        nn = fetch_word(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                       wram, wram_bank, oam, io, hram, ie)
        set_bc(regs, nn)
        cycles = 12
    elif opcode == 0x11:  # LD DE, nn
        nn = fetch_word(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                       wram, wram_bank, oam, io, hram, ie)
        set_de(regs, nn)
        cycles = 12
    elif opcode == 0x21:  # LD HL, nn
        nn = fetch_word(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                       wram, wram_bank, oam, io, hram, ie)
        set_hl(regs, nn)
        cycles = 12
    elif opcode == 0x31:  # LD SP, nn
        nn = fetch_word(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                       wram, wram_bank, oam, io, hram, ie)
        set_sp(regs, nn)
        cycles = 12
    
    # LD (r16), A
    elif opcode == 0x02:  # LD (BC), A
        write_mem(rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                 vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, get_bc(regs), regs[REG_A])
        cycles = 8
    elif opcode == 0x12:  # LD (DE), A
        write_mem(rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                 vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, get_de(regs), regs[REG_A])
        cycles = 8
    elif opcode == 0x22:  # LD (HL+), A
        hl = get_hl(regs)
        write_mem(rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                 vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, hl, regs[REG_A])
        set_hl(regs, (hl + 1) & 0xFFFF)
        cycles = 8
    elif opcode == 0x32:  # LD (HL-), A
        hl = get_hl(regs)
        write_mem(rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                 vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, hl, regs[REG_A])
        set_hl(regs, (hl - 1) & 0xFFFF)
        cycles = 8
    
    # INC r16
    elif opcode == 0x03:  # INC BC
        set_bc(regs, (get_bc(regs) + 1) & 0xFFFF)
        cycles = 8
    elif opcode == 0x13:  # INC DE
        set_de(regs, (get_de(regs) + 1) & 0xFFFF)
        cycles = 8
    elif opcode == 0x23:  # INC HL
        set_hl(regs, (get_hl(regs) + 1) & 0xFFFF)
        cycles = 8
    elif opcode == 0x33:  # INC SP
        set_sp(regs, (get_sp(regs) + 1) & 0xFFFF)
        cycles = 8
    
    # DEC r16
    elif opcode == 0x0B:  # DEC BC
        set_bc(regs, (get_bc(regs) - 1) & 0xFFFF)
        cycles = 8
    elif opcode == 0x1B:  # DEC DE
        set_de(regs, (get_de(regs) - 1) & 0xFFFF)
        cycles = 8
    elif opcode == 0x2B:  # DEC HL
        set_hl(regs, (get_hl(regs) - 1) & 0xFFFF)
        cycles = 8
    elif opcode == 0x3B:  # DEC SP
        set_sp(regs, (get_sp(regs) - 1) & 0xFFFF)
        cycles = 8
    
    # INC r8
    elif opcode == 0x04:
        regs[REG_B] = alu_inc(regs, regs[REG_B])
        cycles = 4
    elif opcode == 0x0C:
        regs[REG_C] = alu_inc(regs, regs[REG_C])
        cycles = 4
    elif opcode == 0x14:
        regs[REG_D] = alu_inc(regs, regs[REG_D])
        cycles = 4
    elif opcode == 0x1C:
        regs[REG_E] = alu_inc(regs, regs[REG_E])
        cycles = 4
    elif opcode == 0x24:
        regs[REG_H] = alu_inc(regs, regs[REG_H])
        cycles = 4
    elif opcode == 0x2C:
        regs[REG_L] = alu_inc(regs, regs[REG_L])
        cycles = 4
    elif opcode == 0x34:  # INC (HL)
        hl = get_hl(regs)
        val = read_mem(rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                      wram, wram_bank, oam, io, hram, ie, hl)
        write_mem(rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                 vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, hl, alu_inc(regs, val))
        cycles = 12
    elif opcode == 0x3C:
        regs[REG_A] = alu_inc(regs, regs[REG_A])
        cycles = 4
    
    # DEC r8
    elif opcode == 0x05:
        regs[REG_B] = alu_dec(regs, regs[REG_B])
        cycles = 4
    elif opcode == 0x0D:
        regs[REG_C] = alu_dec(regs, regs[REG_C])
        cycles = 4
    elif opcode == 0x15:
        regs[REG_D] = alu_dec(regs, regs[REG_D])
        cycles = 4
    elif opcode == 0x1D:
        regs[REG_E] = alu_dec(regs, regs[REG_E])
        cycles = 4
    elif opcode == 0x25:
        regs[REG_H] = alu_dec(regs, regs[REG_H])
        cycles = 4
    elif opcode == 0x2D:
        regs[REG_L] = alu_dec(regs, regs[REG_L])
        cycles = 4
    elif opcode == 0x35:  # DEC (HL)
        hl = get_hl(regs)
        val = read_mem(rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                      wram, wram_bank, oam, io, hram, ie, hl)
        write_mem(rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                 vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, hl, alu_dec(regs, val))
        cycles = 12
    elif opcode == 0x3D:
        regs[REG_A] = alu_dec(regs, regs[REG_A])
        cycles = 4
    
    # LD r8, n
    elif opcode == 0x06:
        regs[REG_B] = fetch_byte(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                                wram, wram_bank, oam, io, hram, ie)
        cycles = 8
    elif opcode == 0x0E:
        regs[REG_C] = fetch_byte(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                                wram, wram_bank, oam, io, hram, ie)
        cycles = 8
    elif opcode == 0x16:
        regs[REG_D] = fetch_byte(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                                wram, wram_bank, oam, io, hram, ie)
        cycles = 8
    elif opcode == 0x1E:
        regs[REG_E] = fetch_byte(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                                wram, wram_bank, oam, io, hram, ie)
        cycles = 8
    elif opcode == 0x26:
        regs[REG_H] = fetch_byte(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                                wram, wram_bank, oam, io, hram, ie)
        cycles = 8
    elif opcode == 0x2E:
        regs[REG_L] = fetch_byte(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                                wram, wram_bank, oam, io, hram, ie)
        cycles = 8
    elif opcode == 0x36:  # LD (HL), n
        n = fetch_byte(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                      wram, wram_bank, oam, io, hram, ie)
        write_mem(rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                 vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, get_hl(regs), n)
        cycles = 12
    elif opcode == 0x3E:
        regs[REG_A] = fetch_byte(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                                wram, wram_bank, oam, io, hram, ie)
        cycles = 8
    
    # Rotates A
    elif opcode == 0x07:  # RLCA
        a = regs[REG_A]
        carry = (a >> 7) & 1
        regs[REG_A] = ((a << 1) | carry) & 0xFF
        regs[REG_F] = FLAG_C if carry else 0
        cycles = 4
    elif opcode == 0x0F:  # RRCA
        a = regs[REG_A]
        carry = a & 1
        regs[REG_A] = ((a >> 1) | (carry << 7)) & 0xFF
        regs[REG_F] = FLAG_C if carry else 0
        cycles = 4
    elif opcode == 0x17:  # RLA
        a = regs[REG_A]
        old_c = 1 if (regs[REG_F] & FLAG_C) else 0
        carry = (a >> 7) & 1
        regs[REG_A] = ((a << 1) | old_c) & 0xFF
        regs[REG_F] = FLAG_C if carry else 0
        cycles = 4
    elif opcode == 0x1F:  # RRA
        a = regs[REG_A]
        old_c = 0x80 if (regs[REG_F] & FLAG_C) else 0
        carry = a & 1
        regs[REG_A] = ((a >> 1) | old_c) & 0xFF
        regs[REG_F] = FLAG_C if carry else 0
        cycles = 4
    
    # JR
    elif opcode == 0x18:  # JR n
        offset = fetch_byte(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                           wram, wram_bank, oam, io, hram, ie)
        if offset > 127:
            offset -= 256
        set_pc(regs, (get_pc(regs) + offset) & 0xFFFF)
        cycles = 12
    elif opcode == 0x20:  # JR NZ, n
        offset = fetch_byte(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                           wram, wram_bank, oam, io, hram, ie)
        if offset > 127:
            offset -= 256
        if not get_flag_z(regs):
            set_pc(regs, (get_pc(regs) + offset) & 0xFFFF)
            cycles = 12
        else:
            cycles = 8
    elif opcode == 0x28:  # JR Z, n
        offset = fetch_byte(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                           wram, wram_bank, oam, io, hram, ie)
        if offset > 127:
            offset -= 256
        if get_flag_z(regs):
            set_pc(regs, (get_pc(regs) + offset) & 0xFFFF)
            cycles = 12
        else:
            cycles = 8
    elif opcode == 0x30:  # JR NC, n
        offset = fetch_byte(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                           wram, wram_bank, oam, io, hram, ie)
        if offset > 127:
            offset -= 256
        if not get_flag_c(regs):
            set_pc(regs, (get_pc(regs) + offset) & 0xFFFF)
            cycles = 12
        else:
            cycles = 8
    elif opcode == 0x38:  # JR C, n
        offset = fetch_byte(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                           wram, wram_bank, oam, io, hram, ie)
        if offset > 127:
            offset -= 256
        if get_flag_c(regs):
            set_pc(regs, (get_pc(regs) + offset) & 0xFFFF)
            cycles = 12
        else:
            cycles = 8
    
    # ADD HL, r16
    elif opcode == 0x09:  # ADD HL, BC
        hl = get_hl(regs)
        bc = get_bc(regs)
        result = hl + bc
        regs[REG_F] = (regs[REG_F] & FLAG_Z)
        if ((hl & 0x0FFF) + (bc & 0x0FFF)) > 0x0FFF:
            regs[REG_F] |= FLAG_H
        if result > 0xFFFF:
            regs[REG_F] |= FLAG_C
        set_hl(regs, result & 0xFFFF)
        cycles = 8
    elif opcode == 0x19:  # ADD HL, DE
        hl = get_hl(regs)
        de = get_de(regs)
        result = hl + de
        regs[REG_F] = (regs[REG_F] & FLAG_Z)
        if ((hl & 0x0FFF) + (de & 0x0FFF)) > 0x0FFF:
            regs[REG_F] |= FLAG_H
        if result > 0xFFFF:
            regs[REG_F] |= FLAG_C
        set_hl(regs, result & 0xFFFF)
        cycles = 8
    elif opcode == 0x29:  # ADD HL, HL
        hl = get_hl(regs)
        result = hl + hl
        regs[REG_F] = (regs[REG_F] & FLAG_Z)
        if ((hl & 0x0FFF) + (hl & 0x0FFF)) > 0x0FFF:
            regs[REG_F] |= FLAG_H
        if result > 0xFFFF:
            regs[REG_F] |= FLAG_C
        set_hl(regs, result & 0xFFFF)
        cycles = 8
    elif opcode == 0x39:  # ADD HL, SP
        hl = get_hl(regs)
        sp = get_sp(regs)
        result = hl + sp
        regs[REG_F] = (regs[REG_F] & FLAG_Z)
        if ((hl & 0x0FFF) + (sp & 0x0FFF)) > 0x0FFF:
            regs[REG_F] |= FLAG_H
        if result > 0xFFFF:
            regs[REG_F] |= FLAG_C
        set_hl(regs, result & 0xFFFF)
        cycles = 8
    
    # LD A, (r16)
    elif opcode == 0x0A:  # LD A, (BC)
        regs[REG_A] = read_mem(rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                              wram, wram_bank, oam, io, hram, ie, get_bc(regs))
        cycles = 8
    elif opcode == 0x1A:  # LD A, (DE)
        regs[REG_A] = read_mem(rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                              wram, wram_bank, oam, io, hram, ie, get_de(regs))
        cycles = 8
    elif opcode == 0x2A:  # LD A, (HL+)
        hl = get_hl(regs)
        regs[REG_A] = read_mem(rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                              wram, wram_bank, oam, io, hram, ie, hl)
        set_hl(regs, (hl + 1) & 0xFFFF)
        cycles = 8
    elif opcode == 0x3A:  # LD A, (HL-)
        hl = get_hl(regs)
        regs[REG_A] = read_mem(rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                              wram, wram_bank, oam, io, hram, ie, hl)
        set_hl(regs, (hl - 1) & 0xFFFF)
        cycles = 8
    
    # DAA
    elif opcode == 0x27:
        a = regs[REG_A]
        if not get_flag_n(regs):
            if get_flag_c(regs) or a > 0x99:
                a = (a + 0x60) & 0xFF
                regs[REG_F] |= FLAG_C
            if get_flag_h(regs) or (a & 0x0F) > 0x09:
                a = (a + 0x06) & 0xFF
        else:
            if get_flag_c(regs):
                a = (a - 0x60) & 0xFF
            if get_flag_h(regs):
                a = (a - 0x06) & 0xFF
        regs[REG_A] = a
        if a == 0:
            regs[REG_F] |= FLAG_Z
        else:
            regs[REG_F] &= ~FLAG_Z & 0xFF
        regs[REG_F] &= ~FLAG_H & 0xFF
        cycles = 4
    
    # CPL
    elif opcode == 0x2F:
        regs[REG_A] ^= 0xFF
        regs[REG_F] |= FLAG_N | FLAG_H
        cycles = 4
    
    # SCF
    elif opcode == 0x37:
        regs[REG_F] = (regs[REG_F] & FLAG_Z) | FLAG_C
        cycles = 4
    
    # CCF
    elif opcode == 0x3F:
        regs[REG_F] = (regs[REG_F] & FLAG_Z) | (0 if (regs[REG_F] & FLAG_C) else FLAG_C)
        cycles = 4
    
    # LD (nn), SP
    elif opcode == 0x08:
        nn = fetch_word(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                       wram, wram_bank, oam, io, hram, ie)
        sp = get_sp(regs)
        write_mem(rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                 vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, nn, sp & 0xFF)
        write_mem(rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                 vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, nn + 1, (sp >> 8) & 0xFF)
        cycles = 20
    
    # HALT
    elif opcode == 0x76:
        state[STATE_HALTED] = 1
        cycles = 4
    
    # STOP
    elif opcode == 0x10:
        fetch_byte(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                  wram, wram_bank, oam, io, hram, ie)  # Consume extra byte
        cycles = 4
    
    # LD r, r (0x40-0x7F except HALT)
    elif 0x40 <= opcode <= 0x7F and opcode != 0x76:
        src = opcode & 0x07
        dst = (opcode >> 3) & 0x07
        
        # Get source value
        if src == 0:
            val = regs[REG_B]
        elif src == 1:
            val = regs[REG_C]
        elif src == 2:
            val = regs[REG_D]
        elif src == 3:
            val = regs[REG_E]
        elif src == 4:
            val = regs[REG_H]
        elif src == 5:
            val = regs[REG_L]
        elif src == 6:
            val = read_mem(rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                          wram, wram_bank, oam, io, hram, ie, get_hl(regs))
            cycles = 8
        else:
            val = regs[REG_A]
        
        # Set destination
        if dst == 0:
            regs[REG_B] = val
        elif dst == 1:
            regs[REG_C] = val
        elif dst == 2:
            regs[REG_D] = val
        elif dst == 3:
            regs[REG_E] = val
        elif dst == 4:
            regs[REG_H] = val
        elif dst == 5:
            regs[REG_L] = val
        elif dst == 6:
            write_mem(rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                     vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, get_hl(regs), val)
            cycles = 8
        else:
            regs[REG_A] = val
        
        if cycles == 4 and (src == 6 or dst == 6):
            cycles = 8
    
    # ALU A, r (0x80-0xBF)
    elif 0x80 <= opcode <= 0xBF:
        src = opcode & 0x07
        op = (opcode >> 3) & 0x07
        
        # Get source value
        if src == 0:
            val = regs[REG_B]
        elif src == 1:
            val = regs[REG_C]
        elif src == 2:
            val = regs[REG_D]
        elif src == 3:
            val = regs[REG_E]
        elif src == 4:
            val = regs[REG_H]
        elif src == 5:
            val = regs[REG_L]
        elif src == 6:
            val = read_mem(rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                          wram, wram_bank, oam, io, hram, ie, get_hl(regs))
            cycles = 8
        else:
            val = regs[REG_A]
        
        if op == 0:
            alu_add(regs, val)
        elif op == 1:
            alu_adc(regs, val)
        elif op == 2:
            alu_sub(regs, val)
        elif op == 3:
            alu_sbc(regs, val)
        elif op == 4:
            alu_and(regs, val)
        elif op == 5:
            alu_xor(regs, val)
        elif op == 6:
            alu_or(regs, val)
        else:
            alu_cp(regs, val)
    
    # RET cc
    elif opcode == 0xC0:  # RET NZ
        if not get_flag_z(regs):
            set_pc(regs, pop_word(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                                 wram, wram_bank, oam, io, hram, ie))
            cycles = 20
        else:
            cycles = 8
    elif opcode == 0xC8:  # RET Z
        if get_flag_z(regs):
            set_pc(regs, pop_word(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                                 wram, wram_bank, oam, io, hram, ie))
            cycles = 20
        else:
            cycles = 8
    elif opcode == 0xD0:  # RET NC
        if not get_flag_c(regs):
            set_pc(regs, pop_word(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                                 wram, wram_bank, oam, io, hram, ie))
            cycles = 20
        else:
            cycles = 8
    elif opcode == 0xD8:  # RET C
        if get_flag_c(regs):
            set_pc(regs, pop_word(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                                 wram, wram_bank, oam, io, hram, ie))
            cycles = 20
        else:
            cycles = 8
    
    # RET / RETI
    elif opcode == 0xC9:  # RET
        set_pc(regs, pop_word(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                             wram, wram_bank, oam, io, hram, ie))
        cycles = 16
    elif opcode == 0xD9:  # RETI
        set_pc(regs, pop_word(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                             wram, wram_bank, oam, io, hram, ie))
        state[STATE_IME] = 1
        cycles = 16
    
    # POP r16
    elif opcode == 0xC1:  # POP BC
        set_bc(regs, pop_word(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                             wram, wram_bank, oam, io, hram, ie))
        cycles = 12
    elif opcode == 0xD1:  # POP DE
        set_de(regs, pop_word(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                             wram, wram_bank, oam, io, hram, ie))
        cycles = 12
    elif opcode == 0xE1:  # POP HL
        set_hl(regs, pop_word(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                             wram, wram_bank, oam, io, hram, ie))
        cycles = 12
    elif opcode == 0xF1:  # POP AF
        set_af(regs, pop_word(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                             wram, wram_bank, oam, io, hram, ie))
        cycles = 12
    
    # PUSH r16
    elif opcode == 0xC5:  # PUSH BC
        push_word(regs, rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                 vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, get_bc(regs))
        cycles = 16
    elif opcode == 0xD5:  # PUSH DE
        push_word(regs, rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                 vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, get_de(regs))
        cycles = 16
    elif opcode == 0xE5:  # PUSH HL
        push_word(regs, rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                 vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, get_hl(regs))
        cycles = 16
    elif opcode == 0xF5:  # PUSH AF
        push_word(regs, rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                 vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, get_af(regs))
        cycles = 16
    
    # JP cc, nn
    elif opcode == 0xC2:  # JP NZ, nn
        nn = fetch_word(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                       wram, wram_bank, oam, io, hram, ie)
        if not get_flag_z(regs):
            set_pc(regs, nn)
            cycles = 16
        else:
            cycles = 12
    elif opcode == 0xCA:  # JP Z, nn
        nn = fetch_word(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                       wram, wram_bank, oam, io, hram, ie)
        if get_flag_z(regs):
            set_pc(regs, nn)
            cycles = 16
        else:
            cycles = 12
    elif opcode == 0xD2:  # JP NC, nn
        nn = fetch_word(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                       wram, wram_bank, oam, io, hram, ie)
        if not get_flag_c(regs):
            set_pc(regs, nn)
            cycles = 16
        else:
            cycles = 12
    elif opcode == 0xDA:  # JP C, nn
        nn = fetch_word(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                       wram, wram_bank, oam, io, hram, ie)
        if get_flag_c(regs):
            set_pc(regs, nn)
            cycles = 16
        else:
            cycles = 12
    
    # JP nn
    elif opcode == 0xC3:
        set_pc(regs, fetch_word(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                               wram, wram_bank, oam, io, hram, ie))
        cycles = 16
    
    # JP (HL)
    elif opcode == 0xE9:
        set_pc(regs, get_hl(regs))
        cycles = 4
    
    # CALL cc, nn
    elif opcode == 0xC4:  # CALL NZ, nn
        nn = fetch_word(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                       wram, wram_bank, oam, io, hram, ie)
        if not get_flag_z(regs):
            push_word(regs, rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                     vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, get_pc(regs))
            set_pc(regs, nn)
            cycles = 24
        else:
            cycles = 12
    elif opcode == 0xCC:  # CALL Z, nn
        nn = fetch_word(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                       wram, wram_bank, oam, io, hram, ie)
        if get_flag_z(regs):
            push_word(regs, rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                     vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, get_pc(regs))
            set_pc(regs, nn)
            cycles = 24
        else:
            cycles = 12
    elif opcode == 0xD4:  # CALL NC, nn
        nn = fetch_word(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                       wram, wram_bank, oam, io, hram, ie)
        if not get_flag_c(regs):
            push_word(regs, rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                     vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, get_pc(regs))
            set_pc(regs, nn)
            cycles = 24
        else:
            cycles = 12
    elif opcode == 0xDC:  # CALL C, nn
        nn = fetch_word(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                       wram, wram_bank, oam, io, hram, ie)
        if get_flag_c(regs):
            push_word(regs, rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                     vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, get_pc(regs))
            set_pc(regs, nn)
            cycles = 24
        else:
            cycles = 12
    
    # CALL nn
    elif opcode == 0xCD:
        nn = fetch_word(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                       wram, wram_bank, oam, io, hram, ie)
        push_word(regs, rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                 vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, get_pc(regs))
        set_pc(regs, nn)
        cycles = 24
    
    # RST
    elif opcode == 0xC7:  # RST 00
        push_word(regs, rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                 vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, get_pc(regs))
        set_pc(regs, 0x00)
        cycles = 16
    elif opcode == 0xCF:  # RST 08
        push_word(regs, rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                 vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, get_pc(regs))
        set_pc(regs, 0x08)
        cycles = 16
    elif opcode == 0xD7:  # RST 10
        push_word(regs, rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                 vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, get_pc(regs))
        set_pc(regs, 0x10)
        cycles = 16
    elif opcode == 0xDF:  # RST 18
        push_word(regs, rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                 vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, get_pc(regs))
        set_pc(regs, 0x18)
        cycles = 16
    elif opcode == 0xE7:  # RST 20
        push_word(regs, rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                 vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, get_pc(regs))
        set_pc(regs, 0x20)
        cycles = 16
    elif opcode == 0xEF:  # RST 28
        push_word(regs, rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                 vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, get_pc(regs))
        set_pc(regs, 0x28)
        cycles = 16
    elif opcode == 0xF7:  # RST 30
        push_word(regs, rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                 vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, get_pc(regs))
        set_pc(regs, 0x30)
        cycles = 16
    elif opcode == 0xFF:  # RST 38
        push_word(regs, rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                 vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, get_pc(regs))
        set_pc(regs, 0x38)
        cycles = 16
    
    # ALU A, n
    elif opcode == 0xC6:  # ADD A, n
        alu_add(regs, fetch_byte(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                                wram, wram_bank, oam, io, hram, ie))
        cycles = 8
    elif opcode == 0xCE:  # ADC A, n
        alu_adc(regs, fetch_byte(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                                wram, wram_bank, oam, io, hram, ie))
        cycles = 8
    elif opcode == 0xD6:  # SUB n
        alu_sub(regs, fetch_byte(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                                wram, wram_bank, oam, io, hram, ie))
        cycles = 8
    elif opcode == 0xDE:  # SBC A, n
        alu_sbc(regs, fetch_byte(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                                wram, wram_bank, oam, io, hram, ie))
        cycles = 8
    elif opcode == 0xE6:  # AND n
        alu_and(regs, fetch_byte(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                                wram, wram_bank, oam, io, hram, ie))
        cycles = 8
    elif opcode == 0xEE:  # XOR n
        alu_xor(regs, fetch_byte(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                                wram, wram_bank, oam, io, hram, ie))
        cycles = 8
    elif opcode == 0xF6:  # OR n
        alu_or(regs, fetch_byte(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                               wram, wram_bank, oam, io, hram, ie))
        cycles = 8
    elif opcode == 0xFE:  # CP n
        alu_cp(regs, fetch_byte(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                               wram, wram_bank, oam, io, hram, ie))
        cycles = 8
    
    # LDH
    elif opcode == 0xE0:  # LDH (n), A
        n = fetch_byte(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                      wram, wram_bank, oam, io, hram, ie)
        write_mem(rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                 vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, 0xFF00 + n, regs[REG_A])
        cycles = 12
    elif opcode == 0xF0:  # LDH A, (n)
        n = fetch_byte(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                      wram, wram_bank, oam, io, hram, ie)
        regs[REG_A] = read_mem(rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                              wram, wram_bank, oam, io, hram, ie, 0xFF00 + n)
        cycles = 12
    elif opcode == 0xE2:  # LD (C), A
        write_mem(rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                 vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, 0xFF00 + regs[REG_C], regs[REG_A])
        cycles = 8
    elif opcode == 0xF2:  # LD A, (C)
        regs[REG_A] = read_mem(rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                              wram, wram_bank, oam, io, hram, ie, 0xFF00 + regs[REG_C])
        cycles = 8
    
    # LD (nn), A and LD A, (nn)
    elif opcode == 0xEA:  # LD (nn), A
        nn = fetch_word(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                       wram, wram_bank, oam, io, hram, ie)
        write_mem(rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                 vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, nn, regs[REG_A])
        cycles = 16
    elif opcode == 0xFA:  # LD A, (nn)
        nn = fetch_word(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                       wram, wram_bank, oam, io, hram, ie)
        regs[REG_A] = read_mem(rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                              wram, wram_bank, oam, io, hram, ie, nn)
        cycles = 16
    
    # ADD SP, n
    elif opcode == 0xE8:
        n = fetch_byte(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                      wram, wram_bank, oam, io, hram, ie)
        if n > 127:
            n -= 256
        sp = get_sp(regs)
        result = sp + n
        regs[REG_F] = 0
        if ((sp & 0x0F) + (n & 0x0F)) > 0x0F:
            regs[REG_F] |= FLAG_H
        if ((sp & 0xFF) + (n & 0xFF)) > 0xFF:
            regs[REG_F] |= FLAG_C
        set_sp(regs, result & 0xFFFF)
        cycles = 16
    
    # LD HL, SP+n
    elif opcode == 0xF8:
        n = fetch_byte(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                      wram, wram_bank, oam, io, hram, ie)
        if n > 127:
            n -= 256
        sp = get_sp(regs)
        result = sp + n
        regs[REG_F] = 0
        if ((sp & 0x0F) + (n & 0x0F)) > 0x0F:
            regs[REG_F] |= FLAG_H
        if ((sp & 0xFF) + (n & 0xFF)) > 0xFF:
            regs[REG_F] |= FLAG_C
        set_hl(regs, result & 0xFFFF)
        cycles = 12
    
    # LD SP, HL
    elif opcode == 0xF9:
        set_sp(regs, get_hl(regs))
        cycles = 8
    
    # DI / EI
    elif opcode == 0xF3:  # DI
        state[STATE_IME] = 0
        cycles = 4
    elif opcode == 0xFB:  # EI
        state[STATE_IME_SCHEDULED] = 1
        cycles = 4
    
    # CB prefix
    elif opcode == 0xCB:
        cb_opcode = fetch_byte(regs, rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                              wram, wram_bank, oam, io, hram, ie)
        
        reg = cb_opcode & 0x07
        op = cb_opcode >> 3
        
        # Get value
        if reg == 0:
            val = regs[REG_B]
        elif reg == 1:
            val = regs[REG_C]
        elif reg == 2:
            val = regs[REG_D]
        elif reg == 3:
            val = regs[REG_E]
        elif reg == 4:
            val = regs[REG_H]
        elif reg == 5:
            val = regs[REG_L]
        elif reg == 6:
            val = read_mem(rom, rom_bank, ram, ram_bank, ram_enabled, vram, vram_bank,
                          wram, wram_bank, oam, io, hram, ie, get_hl(regs))
        else:
            val = regs[REG_A]
        
        # Execute operation
        if op < 8:  # Rotate/shift
            if op == 0:
                val = cb_rlc(regs, val)
            elif op == 1:
                val = cb_rrc(regs, val)
            elif op == 2:
                val = cb_rl(regs, val)
            elif op == 3:
                val = cb_rr(regs, val)
            elif op == 4:
                val = cb_sla(regs, val)
            elif op == 5:
                val = cb_sra(regs, val)
            elif op == 6:
                val = cb_swap(regs, val)
            else:
                val = cb_srl(regs, val)
            
            # Write back
            if reg == 0:
                regs[REG_B] = val
            elif reg == 1:
                regs[REG_C] = val
            elif reg == 2:
                regs[REG_D] = val
            elif reg == 3:
                regs[REG_E] = val
            elif reg == 4:
                regs[REG_H] = val
            elif reg == 5:
                regs[REG_L] = val
            elif reg == 6:
                write_mem(rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                         vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, get_hl(regs), val)
            else:
                regs[REG_A] = val
            
            cycles = 16 if reg == 6 else 8
            
        elif op < 16:  # BIT
            bit = op - 8
            cb_bit(regs, bit, val)
            cycles = 12 if reg == 6 else 8
            
        elif op < 24:  # RES
            bit = op - 16
            val &= ~(1 << bit) & 0xFF
            
            if reg == 0:
                regs[REG_B] = val
            elif reg == 1:
                regs[REG_C] = val
            elif reg == 2:
                regs[REG_D] = val
            elif reg == 3:
                regs[REG_E] = val
            elif reg == 4:
                regs[REG_H] = val
            elif reg == 5:
                regs[REG_L] = val
            elif reg == 6:
                write_mem(rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                         vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, get_hl(regs), val)
            else:
                regs[REG_A] = val
            
            cycles = 16 if reg == 6 else 8
            
        else:  # SET
            bit = op - 24
            val |= (1 << bit)
            
            if reg == 0:
                regs[REG_B] = val
            elif reg == 1:
                regs[REG_C] = val
            elif reg == 2:
                regs[REG_D] = val
            elif reg == 3:
                regs[REG_E] = val
            elif reg == 4:
                regs[REG_H] = val
            elif reg == 5:
                regs[REG_L] = val
            elif reg == 6:
                write_mem(rom, rom_bank_ptr, ram, ram_bank_ptr, ram_enabled_ptr,
                         vram, vram_bank, wram, wram_bank, oam, io, hram, ie, mbc_type, get_hl(regs), val)
            else:
                regs[REG_A] = val
            
            cycles = 16 if reg == 6 else 8
    
    else:
        # Undefined opcode - treat as NOP
        cycles = 4
    
    return cycles


class CPUFast:
    """
    JIT-compiled CPU wrapper class.
    Uses Numba-compiled functions for near-native speed.
    """
    
    def __init__(self, memory):
        self.memory = memory
        
        # Registers as numpy array for JIT
        self.regs = np.zeros(12, dtype=np.uint8)
        
        # State flags
        self.state = np.zeros(4, dtype=np.uint8)
        
        # Bank pointers (for JIT modification)
        self.rom_bank_ptr = np.array([1], dtype=np.int32)
        self.ram_bank_ptr = np.array([0], dtype=np.int32)
        self.ram_enabled_ptr = np.array([0], dtype=np.uint8)
        
        # MBC type
        self.mbc_type = 5  # Default to MBC5
        
        # IE register
        self.ie = np.array([0], dtype=np.uint8)
        
        # Cycle counter
        self.cycles = 0
        self.total_cycles = 0
        
        # Initialize registers
        self._init_regs()
    
    def _init_regs(self):
        """Initialize registers to post-boot values."""
        self.regs[REG_A] = 0x01
        self.regs[REG_F] = 0xB0
        self.regs[REG_B] = 0x00
        self.regs[REG_C] = 0x13
        self.regs[REG_D] = 0x00
        self.regs[REG_E] = 0xD8
        self.regs[REG_H] = 0x01
        self.regs[REG_L] = 0x4D
        set_sp(self.regs, 0xFFFE)
        set_pc(self.regs, 0x0100)
    
    def init_gbc_mode(self):
        """Initialize for GBC mode."""
        self.regs[REG_A] = 0x11
        self.regs[REG_F] = 0x80
        self.regs[REG_B] = 0x00
        self.regs[REG_C] = 0x00
        self.regs[REG_D] = 0xFF
        self.regs[REG_E] = 0x56
        self.regs[REG_H] = 0x00
        self.regs[REG_L] = 0x0D
    
    @property
    def pc(self):
        return get_pc(self.regs)
    
    @pc.setter
    def pc(self, value):
        set_pc(self.regs, value)
    
    @property
    def sp(self):
        return get_sp(self.regs)
    
    @property
    def a(self):
        return self.regs[REG_A]
    
    @property
    def f(self):
        return self.regs[REG_F]
    
    @property
    def b(self):
        return self.regs[REG_B]
    
    @property
    def c(self):
        return self.regs[REG_C]
    
    @property
    def d(self):
        return self.regs[REG_D]
    
    @property
    def e(self):
        return self.regs[REG_E]
    
    @property
    def h(self):
        return self.regs[REG_H]
    
    @property
    def l(self):
        return self.regs[REG_L]
    
    @property
    def af(self):
        return get_af(self.regs)
    
    @property
    def bc(self):
        return get_bc(self.regs)
    
    @property
    def de(self):
        return get_de(self.regs)
    
    @property
    def hl(self):
        return get_hl(self.regs)
    
    @property
    def ime(self):
        return self.state[STATE_IME] != 0
    
    @property
    def halted(self):
        return self.state[STATE_HALTED] != 0
    
    @property
    def flag_z(self):
        return get_flag_z(self.regs)
    
    @property
    def flag_n(self):
        return get_flag_n(self.regs)
    
    @property
    def flag_h(self):
        return get_flag_h(self.regs)
    
    @property
    def flag_c(self):
        return get_flag_c(self.regs)
    
    @property
    def double_speed(self):
        return self.state[STATE_DOUBLE_SPEED] != 0
    
    def step(self) -> int:
        """Execute one instruction."""
        # Get memory arrays
        rom = self.memory.mbc.rom if self.memory.mbc else np.zeros(0x8000, dtype=np.uint8)
        ram = self.memory.mbc.ram if self.memory.mbc else np.zeros(0x2000, dtype=np.uint8)
        
        # Sync bank values
        if self.memory.mbc:
            self.rom_bank_ptr[0] = self.memory.mbc.rom_bank
            self.ram_bank_ptr[0] = self.memory.mbc.ram_bank
            self.ram_enabled_ptr[0] = 1 if self.memory.mbc.ram_enabled else 0
        
        # Execute
        cycles = execute_instruction(
            self.regs, self.state,
            rom, self.rom_bank_ptr, ram, self.ram_bank_ptr, self.ram_enabled_ptr,
            self.memory.vram, self.memory.vram_bank,
            self.memory.wram, self.memory.wram_bank,
            self.memory.oam, self.memory.io, self.memory.hram, self.ie,
            self.mbc_type
        )
        
        # Sync bank values back
        if self.memory.mbc:
            self.memory.mbc.rom_bank = self.rom_bank_ptr[0]
            self.memory.mbc.ram_bank = self.ram_bank_ptr[0]
            self.memory.mbc.ram_enabled = self.ram_enabled_ptr[0] != 0
        
        # Sync IE
        self.memory.ie = self.ie[0]
        
        self.cycles = cycles
        self.total_cycles += cycles
        return cycles
    
    def request_interrupt(self, bit: int):
        """Request an interrupt."""
        self.memory.io[0x0F] = int(self.memory.io[0x0F]) | (1 << bit)
    
    def read_byte(self, addr: int) -> int:
        return int(self.memory.read(addr))
    
    def write_byte(self, addr: int, value: int):
        self.memory.write(addr, value)


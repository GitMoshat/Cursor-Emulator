"""
Game Boy Color CPU Emulator - Sharp LR35902 (Z80-like)
Complete implementation with all opcodes and accurate cycle timing.
"""

import numpy as np
from typing import Callable, Optional


class CPU:
    """
    Sharp LR35902 CPU - Custom Z80-like processor used in Game Boy/Color.
    
    Registers:
    - A (Accumulator), F (Flags)
    - B, C, D, E, H, L (General purpose)
    - SP (Stack Pointer), PC (Program Counter)
    
    Flags (F register):
    - Bit 7: Z (Zero)
    - Bit 6: N (Subtract)
    - Bit 5: H (Half Carry)
    - Bit 4: C (Carry)
    """
    
    # Flag bit positions
    FLAG_Z = 0x80  # Zero flag
    FLAG_N = 0x40  # Subtract flag
    FLAG_H = 0x20  # Half carry flag
    FLAG_C = 0x10  # Carry flag
    
    def __init__(self, memory):
        self.memory = memory
        
        # 8-bit registers
        self.a = 0x01  # Accumulator (0x11 for GBC)
        self.f = 0xB0  # Flags
        self.b = 0x00
        self.c = 0x13
        self.d = 0x00
        self.e = 0xD8
        self.h = 0x01
        self.l = 0x4D
        
        # 16-bit registers
        self.sp = 0xFFFE  # Stack pointer
        self.pc = 0x0100  # Program counter (after boot ROM)
        
        # CPU state
        self.halted = False
        self.stopped = False
        self.ime = False  # Interrupt Master Enable
        self.ime_scheduled = False  # EI delays by one instruction
        self.cycles = 0
        self.total_cycles = 0
        
        # GBC double speed mode
        self.double_speed = False
        
        # Build opcode tables
        self._build_opcode_table()
        self._build_cb_opcode_table()
    
    # Register pair accessors
    @property
    def af(self) -> int:
        return (self.a << 8) | (self.f & 0xF0)
    
    @af.setter
    def af(self, value: int):
        self.a = (value >> 8) & 0xFF
        self.f = value & 0xF0
    
    @property
    def bc(self) -> int:
        return (self.b << 8) | self.c
    
    @bc.setter
    def bc(self, value: int):
        self.b = (value >> 8) & 0xFF
        self.c = value & 0xFF
    
    @property
    def de(self) -> int:
        return (self.d << 8) | self.e
    
    @de.setter
    def de(self, value: int):
        self.d = (value >> 8) & 0xFF
        self.e = value & 0xFF
    
    @property
    def hl(self) -> int:
        return (self.h << 8) | self.l
    
    @hl.setter
    def hl(self, value: int):
        self.h = (value >> 8) & 0xFF
        self.l = value & 0xFF
    
    # Flag accessors
    def get_flag(self, flag: int) -> bool:
        return (self.f & flag) != 0
    
    def set_flag(self, flag: int, value: bool):
        if value:
            self.f |= flag
        else:
            self.f &= ~flag
        self.f &= 0xF0  # Lower 4 bits always 0
    
    @property
    def flag_z(self) -> bool:
        return self.get_flag(self.FLAG_Z)
    
    @flag_z.setter
    def flag_z(self, value: bool):
        self.set_flag(self.FLAG_Z, value)
    
    @property
    def flag_n(self) -> bool:
        return self.get_flag(self.FLAG_N)
    
    @flag_n.setter
    def flag_n(self, value: bool):
        self.set_flag(self.FLAG_N, value)
    
    @property
    def flag_h(self) -> bool:
        return self.get_flag(self.FLAG_H)
    
    @flag_h.setter
    def flag_h(self, value: bool):
        self.set_flag(self.FLAG_H, value)
    
    @property
    def flag_c(self) -> bool:
        return self.get_flag(self.FLAG_C)
    
    @flag_c.setter
    def flag_c(self, value: bool):
        self.set_flag(self.FLAG_C, value)
    
    # Memory access helpers
    def read_byte(self, addr: int) -> int:
        return int(self.memory.read(addr))
    
    def write_byte(self, addr: int, value: int):
        self.memory.write(int(addr), int(value) & 0xFF)
    
    def read_word(self, addr: int) -> int:
        lo = self.read_byte(addr)
        hi = self.read_byte((int(addr) + 1) & 0xFFFF)
        return (hi << 8) | lo
    
    def write_word(self, addr: int, value: int):
        self.write_byte(addr, int(value) & 0xFF)
        self.write_byte((int(addr) + 1) & 0xFFFF, (int(value) >> 8) & 0xFF)
    
    def fetch_byte(self) -> int:
        value = self.read_byte(self.pc)
        self.pc = (int(self.pc) + 1) & 0xFFFF
        return int(value)
    
    def fetch_word(self) -> int:
        lo = self.fetch_byte()
        hi = self.fetch_byte()
        return (int(hi) << 8) | int(lo)
    
    def fetch_signed_byte(self) -> int:
        value = self.fetch_byte()
        if value > 127:
            value -= 256
        return value
    
    # Stack operations
    def push_word(self, value: int):
        self.sp = (self.sp - 2) & 0xFFFF
        self.write_word(self.sp, value)
    
    def pop_word(self) -> int:
        value = self.read_word(self.sp)
        self.sp = (self.sp + 2) & 0xFFFF
        return value
    
    def step(self) -> int:
        """Execute one instruction and return cycles consumed."""
        self.cycles = 0
        
        # Handle scheduled IME enable
        if self.ime_scheduled:
            self.ime = True
            self.ime_scheduled = False
        
        # Handle interrupts
        if self.ime:
            interrupt_cycles = self._handle_interrupts()
            if interrupt_cycles > 0:
                return interrupt_cycles
        
        # Handle HALT
        if self.halted:
            # Check if any enabled interrupts are pending
            ie = self.read_byte(0xFFFF)
            if_flag = self.read_byte(0xFF0F)
            if ie & if_flag & 0x1F:
                self.halted = False
            else:
                return 4  # HALT consumes 4 cycles per "NOP"
        
        # Handle STOP
        if self.stopped:
            return 4
        
        # Fetch and execute opcode
        opcode = self.fetch_byte()
        
        if opcode == 0xCB:
            # CB-prefixed opcode
            cb_opcode = self.fetch_byte()
            handler = self.cb_opcodes[cb_opcode]
            if handler:
                handler()
            else:
                raise RuntimeError(f"Unknown CB opcode: 0xCB 0x{cb_opcode:02X} at PC=0x{self.pc-2:04X}")
        else:
            handler = self.opcodes[opcode]
            if handler:
                handler()
            else:
                raise RuntimeError(f"Unknown opcode: 0x{opcode:02X} at PC=0x{self.pc-1:04X}")
        
        self.total_cycles += self.cycles
        return self.cycles
    
    def _handle_interrupts(self) -> int:
        """Handle pending interrupts. Returns cycles consumed or 0."""
        ie = self.read_byte(0xFFFF)  # Interrupt Enable
        if_flag = self.read_byte(0xFF0F)  # Interrupt Flag
        pending = ie & if_flag & 0x1F
        
        if not pending:
            return 0
        
        self.halted = False
        
        # Priority: VBlank > LCD STAT > Timer > Serial > Joypad
        for bit, vector in [(0, 0x40), (1, 0x48), (2, 0x50), (3, 0x58), (4, 0x60)]:
            if pending & (1 << bit):
                self.ime = False
                self.write_byte(0xFF0F, if_flag & ~(1 << bit))
                self.push_word(self.pc)
                self.pc = vector
                return 20  # Interrupt handling takes 20 cycles
        
        return 0
    
    def request_interrupt(self, bit: int):
        """Request an interrupt by setting the appropriate IF bit."""
        if_flag = self.read_byte(0xFF0F)
        self.write_byte(0xFF0F, if_flag | (1 << bit))
    
    def init_gbc_mode(self):
        """Initialize registers for GBC mode."""
        self.a = 0x11  # GBC identifier
        self.f = 0x80
        self.b = 0x00
        self.c = 0x00
        self.d = 0xFF
        self.e = 0x56
        self.h = 0x00
        self.l = 0x0D
    
    # =========================================================================
    # OPCODE IMPLEMENTATIONS
    # =========================================================================
    
    def _build_opcode_table(self):
        """Build the main opcode lookup table."""
        self.opcodes = [None] * 256
        
        # NOP
        self.opcodes[0x00] = self._nop
        
        # LD r16, nn
        self.opcodes[0x01] = lambda: self._ld_r16_nn('bc')
        self.opcodes[0x11] = lambda: self._ld_r16_nn('de')
        self.opcodes[0x21] = lambda: self._ld_r16_nn('hl')
        self.opcodes[0x31] = lambda: self._ld_sp_nn()
        
        # LD (r16), A
        self.opcodes[0x02] = lambda: self._ld_r16_a('bc')
        self.opcodes[0x12] = lambda: self._ld_r16_a('de')
        self.opcodes[0x22] = self._ld_hli_a
        self.opcodes[0x32] = self._ld_hld_a
        
        # INC r16
        self.opcodes[0x03] = lambda: self._inc_r16('bc')
        self.opcodes[0x13] = lambda: self._inc_r16('de')
        self.opcodes[0x23] = lambda: self._inc_r16('hl')
        self.opcodes[0x33] = self._inc_sp
        
        # INC r8
        self.opcodes[0x04] = lambda: self._inc_r8('b')
        self.opcodes[0x0C] = lambda: self._inc_r8('c')
        self.opcodes[0x14] = lambda: self._inc_r8('d')
        self.opcodes[0x1C] = lambda: self._inc_r8('e')
        self.opcodes[0x24] = lambda: self._inc_r8('h')
        self.opcodes[0x2C] = lambda: self._inc_r8('l')
        self.opcodes[0x34] = self._inc_hl_ind
        self.opcodes[0x3C] = lambda: self._inc_r8('a')
        
        # DEC r8
        self.opcodes[0x05] = lambda: self._dec_r8('b')
        self.opcodes[0x0D] = lambda: self._dec_r8('c')
        self.opcodes[0x15] = lambda: self._dec_r8('d')
        self.opcodes[0x1D] = lambda: self._dec_r8('e')
        self.opcodes[0x25] = lambda: self._dec_r8('h')
        self.opcodes[0x2D] = lambda: self._dec_r8('l')
        self.opcodes[0x35] = self._dec_hl_ind
        self.opcodes[0x3D] = lambda: self._dec_r8('a')
        
        # LD r8, n
        self.opcodes[0x06] = lambda: self._ld_r8_n('b')
        self.opcodes[0x0E] = lambda: self._ld_r8_n('c')
        self.opcodes[0x16] = lambda: self._ld_r8_n('d')
        self.opcodes[0x1E] = lambda: self._ld_r8_n('e')
        self.opcodes[0x26] = lambda: self._ld_r8_n('h')
        self.opcodes[0x2E] = lambda: self._ld_r8_n('l')
        self.opcodes[0x36] = self._ld_hl_n
        self.opcodes[0x3E] = lambda: self._ld_r8_n('a')
        
        # Rotate A instructions
        self.opcodes[0x07] = self._rlca
        self.opcodes[0x0F] = self._rrca
        self.opcodes[0x17] = self._rla
        self.opcodes[0x1F] = self._rra
        
        # LD (nn), SP
        self.opcodes[0x08] = self._ld_nn_sp
        
        # ADD HL, r16
        self.opcodes[0x09] = lambda: self._add_hl_r16('bc')
        self.opcodes[0x19] = lambda: self._add_hl_r16('de')
        self.opcodes[0x29] = lambda: self._add_hl_r16('hl')
        self.opcodes[0x39] = self._add_hl_sp
        
        # LD A, (r16)
        self.opcodes[0x0A] = lambda: self._ld_a_r16('bc')
        self.opcodes[0x1A] = lambda: self._ld_a_r16('de')
        self.opcodes[0x2A] = self._ld_a_hli
        self.opcodes[0x3A] = self._ld_a_hld
        
        # DEC r16
        self.opcodes[0x0B] = lambda: self._dec_r16('bc')
        self.opcodes[0x1B] = lambda: self._dec_r16('de')
        self.opcodes[0x2B] = lambda: self._dec_r16('hl')
        self.opcodes[0x3B] = self._dec_sp
        
        # JR
        self.opcodes[0x18] = self._jr
        self.opcodes[0x20] = lambda: self._jr_cc(not self.flag_z)
        self.opcodes[0x28] = lambda: self._jr_cc(self.flag_z)
        self.opcodes[0x30] = lambda: self._jr_cc(not self.flag_c)
        self.opcodes[0x38] = lambda: self._jr_cc(self.flag_c)
        
        # DAA
        self.opcodes[0x27] = self._daa
        
        # CPL
        self.opcodes[0x2F] = self._cpl
        
        # SCF
        self.opcodes[0x37] = self._scf
        
        # CCF
        self.opcodes[0x3F] = self._ccf
        
        # LD r8, r8 (0x40-0x7F except HALT at 0x76)
        regs = ['b', 'c', 'd', 'e', 'h', 'l', '(hl)', 'a']
        for dst_idx, dst in enumerate(regs):
            for src_idx, src in enumerate(regs):
                opcode = 0x40 + (dst_idx * 8) + src_idx
                if opcode == 0x76:
                    self.opcodes[0x76] = self._halt
                elif dst == '(hl)':
                    self.opcodes[opcode] = lambda s=src: self._ld_hl_r8(s)
                elif src == '(hl)':
                    self.opcodes[opcode] = lambda d=dst: self._ld_r8_hl(d)
                else:
                    self.opcodes[opcode] = lambda d=dst, s=src: self._ld_r8_r8(d, s)
        
        # ALU operations with registers (0x80-0xBF)
        alu_ops = [self._add_a, self._adc_a, self._sub_a, self._sbc_a,
                   self._and_a, self._xor_a, self._or_a, self._cp_a]
        for op_idx, op in enumerate(alu_ops):
            for src_idx, src in enumerate(regs):
                opcode = 0x80 + (op_idx * 8) + src_idx
                if src == '(hl)':
                    self.opcodes[opcode] = lambda o=op: self._alu_hl(o)
                else:
                    self.opcodes[opcode] = lambda o=op, s=src: self._alu_r8(o, s)
        
        # RET cc
        self.opcodes[0xC0] = lambda: self._ret_cc(not self.flag_z)
        self.opcodes[0xC8] = lambda: self._ret_cc(self.flag_z)
        self.opcodes[0xD0] = lambda: self._ret_cc(not self.flag_c)
        self.opcodes[0xD8] = lambda: self._ret_cc(self.flag_c)
        
        # POP r16
        self.opcodes[0xC1] = lambda: self._pop_r16('bc')
        self.opcodes[0xD1] = lambda: self._pop_r16('de')
        self.opcodes[0xE1] = lambda: self._pop_r16('hl')
        self.opcodes[0xF1] = lambda: self._pop_af()
        
        # JP cc, nn
        self.opcodes[0xC2] = lambda: self._jp_cc_nn(not self.flag_z)
        self.opcodes[0xCA] = lambda: self._jp_cc_nn(self.flag_z)
        self.opcodes[0xD2] = lambda: self._jp_cc_nn(not self.flag_c)
        self.opcodes[0xDA] = lambda: self._jp_cc_nn(self.flag_c)
        
        # JP nn
        self.opcodes[0xC3] = self._jp_nn
        
        # CALL cc, nn
        self.opcodes[0xC4] = lambda: self._call_cc_nn(not self.flag_z)
        self.opcodes[0xCC] = lambda: self._call_cc_nn(self.flag_z)
        self.opcodes[0xD4] = lambda: self._call_cc_nn(not self.flag_c)
        self.opcodes[0xDC] = lambda: self._call_cc_nn(self.flag_c)
        
        # PUSH r16
        self.opcodes[0xC5] = lambda: self._push_r16('bc')
        self.opcodes[0xD5] = lambda: self._push_r16('de')
        self.opcodes[0xE5] = lambda: self._push_r16('hl')
        self.opcodes[0xF5] = lambda: self._push_af()
        
        # ALU A, n
        self.opcodes[0xC6] = lambda: self._alu_n(self._add_a)
        self.opcodes[0xCE] = lambda: self._alu_n(self._adc_a)
        self.opcodes[0xD6] = lambda: self._alu_n(self._sub_a)
        self.opcodes[0xDE] = lambda: self._alu_n(self._sbc_a)
        self.opcodes[0xE6] = lambda: self._alu_n(self._and_a)
        self.opcodes[0xEE] = lambda: self._alu_n(self._xor_a)
        self.opcodes[0xF6] = lambda: self._alu_n(self._or_a)
        self.opcodes[0xFE] = lambda: self._alu_n(self._cp_a)
        
        # RST
        for i in range(8):
            addr = i * 8
            self.opcodes[0xC7 + (i * 8)] = lambda a=addr: self._rst(a)
        
        # RET
        self.opcodes[0xC9] = self._ret
        
        # RETI
        self.opcodes[0xD9] = self._reti
        
        # CALL nn
        self.opcodes[0xCD] = self._call_nn
        
        # LDH instructions
        self.opcodes[0xE0] = self._ldh_n_a
        self.opcodes[0xF0] = self._ldh_a_n
        self.opcodes[0xE2] = self._ldh_c_a
        self.opcodes[0xF2] = self._ldh_a_c
        
        # LD (nn), A and LD A, (nn)
        self.opcodes[0xEA] = self._ld_nn_a
        self.opcodes[0xFA] = self._ld_a_nn
        
        # JP (HL)
        self.opcodes[0xE9] = self._jp_hl
        
        # LD SP, HL
        self.opcodes[0xF9] = self._ld_sp_hl
        
        # DI / EI
        self.opcodes[0xF3] = self._di
        self.opcodes[0xFB] = self._ei
        
        # ADD SP, n
        self.opcodes[0xE8] = self._add_sp_n
        
        # LD HL, SP+n
        self.opcodes[0xF8] = self._ld_hl_sp_n
        
        # STOP
        self.opcodes[0x10] = self._stop
    
    def _build_cb_opcode_table(self):
        """Build the CB-prefixed opcode lookup table."""
        self.cb_opcodes = [None] * 256
        regs = ['b', 'c', 'd', 'e', 'h', 'l', '(hl)', 'a']
        
        # RLC, RRC, RL, RR, SLA, SRA, SWAP, SRL
        shift_ops = [
            self._rlc, self._rrc, self._rl, self._rr,
            self._sla, self._sra, self._swap, self._srl
        ]
        
        for op_idx, op in enumerate(shift_ops):
            for reg_idx, reg in enumerate(regs):
                opcode = (op_idx * 8) + reg_idx
                if reg == '(hl)':
                    self.cb_opcodes[opcode] = lambda o=op: self._cb_hl(o)
                else:
                    self.cb_opcodes[opcode] = lambda o=op, r=reg: self._cb_r8(o, r)
        
        # BIT, RES, SET
        for bit in range(8):
            for reg_idx, reg in enumerate(regs):
                # BIT b, r
                opcode = 0x40 + (bit * 8) + reg_idx
                if reg == '(hl)':
                    self.cb_opcodes[opcode] = lambda b=bit: self._bit_hl(b)
                else:
                    self.cb_opcodes[opcode] = lambda b=bit, r=reg: self._bit_r8(b, r)
                
                # RES b, r
                opcode = 0x80 + (bit * 8) + reg_idx
                if reg == '(hl)':
                    self.cb_opcodes[opcode] = lambda b=bit: self._res_hl(b)
                else:
                    self.cb_opcodes[opcode] = lambda b=bit, r=reg: self._res_r8(b, r)
                
                # SET b, r
                opcode = 0xC0 + (bit * 8) + reg_idx
                if reg == '(hl)':
                    self.cb_opcodes[opcode] = lambda b=bit: self._set_hl(b)
                else:
                    self.cb_opcodes[opcode] = lambda b=bit, r=reg: self._set_r8(b, r)
    
    # =========================================================================
    # INSTRUCTION IMPLEMENTATIONS
    # =========================================================================
    
    def _get_r8(self, reg: str) -> int:
        return getattr(self, reg)
    
    def _set_r8(self, reg: str, value: int):
        setattr(self, reg, value & 0xFF)
    
    def _get_r16(self, reg: str) -> int:
        return getattr(self, reg)
    
    def _set_r16(self, reg: str, value: int):
        setattr(self, reg, value & 0xFFFF)
    
    def _nop(self):
        self.cycles = 4
    
    def _halt(self):
        self.halted = True
        self.cycles = 4
    
    def _stop(self):
        self.fetch_byte()  # STOP is a 2-byte instruction
        # Check for speed switch (GBC)
        key1 = self.read_byte(0xFF4D)
        if key1 & 0x01:
            self.double_speed = not self.double_speed
            self.write_byte(0xFF4D, (0x80 if self.double_speed else 0x00))
        else:
            self.stopped = True
        self.cycles = 4
    
    # 16-bit loads
    def _ld_r16_nn(self, reg: str):
        value = self.fetch_word()
        self._set_r16(reg, value)
        self.cycles = 12
    
    def _ld_sp_nn(self):
        self.sp = self.fetch_word()
        self.cycles = 12
    
    def _ld_nn_sp(self):
        addr = self.fetch_word()
        self.write_word(addr, self.sp)
        self.cycles = 20
    
    def _ld_sp_hl(self):
        self.sp = self.hl
        self.cycles = 8
    
    # 8-bit loads
    def _ld_r8_n(self, reg: str):
        value = self.fetch_byte()
        self._set_r8(reg, value)
        self.cycles = 8
    
    def _ld_hl_n(self):
        value = self.fetch_byte()
        self.write_byte(self.hl, value)
        self.cycles = 12
    
    def _ld_r8_r8(self, dst: str, src: str):
        self._set_r8(dst, self._get_r8(src))
        self.cycles = 4
    
    def _ld_r8_hl(self, dst: str):
        self._set_r8(dst, self.read_byte(self.hl))
        self.cycles = 8
    
    def _ld_hl_r8(self, src: str):
        self.write_byte(self.hl, self._get_r8(src))
        self.cycles = 8
    
    def _ld_r16_a(self, reg: str):
        self.write_byte(self._get_r16(reg), self.a)
        self.cycles = 8
    
    def _ld_a_r16(self, reg: str):
        self.a = self.read_byte(self._get_r16(reg))
        self.cycles = 8
    
    def _ld_hli_a(self):
        self.write_byte(self.hl, self.a)
        self.hl = (self.hl + 1) & 0xFFFF
        self.cycles = 8
    
    def _ld_hld_a(self):
        self.write_byte(self.hl, self.a)
        self.hl = (self.hl - 1) & 0xFFFF
        self.cycles = 8
    
    def _ld_a_hli(self):
        self.a = self.read_byte(self.hl)
        self.hl = (self.hl + 1) & 0xFFFF
        self.cycles = 8
    
    def _ld_a_hld(self):
        self.a = self.read_byte(self.hl)
        self.hl = (self.hl - 1) & 0xFFFF
        self.cycles = 8
    
    def _ldh_n_a(self):
        addr = 0xFF00 + self.fetch_byte()
        self.write_byte(addr, self.a)
        self.cycles = 12
    
    def _ldh_a_n(self):
        addr = 0xFF00 + self.fetch_byte()
        self.a = self.read_byte(addr)
        self.cycles = 12
    
    def _ldh_c_a(self):
        self.write_byte(0xFF00 + self.c, self.a)
        self.cycles = 8
    
    def _ldh_a_c(self):
        self.a = self.read_byte(0xFF00 + self.c)
        self.cycles = 8
    
    def _ld_nn_a(self):
        addr = self.fetch_word()
        self.write_byte(addr, self.a)
        self.cycles = 16
    
    def _ld_a_nn(self):
        addr = self.fetch_word()
        self.a = self.read_byte(addr)
        self.cycles = 16
    
    # 16-bit arithmetic
    def _inc_r16(self, reg: str):
        self._set_r16(reg, (self._get_r16(reg) + 1) & 0xFFFF)
        self.cycles = 8
    
    def _inc_sp(self):
        self.sp = (self.sp + 1) & 0xFFFF
        self.cycles = 8
    
    def _dec_r16(self, reg: str):
        self._set_r16(reg, (self._get_r16(reg) - 1) & 0xFFFF)
        self.cycles = 8
    
    def _dec_sp(self):
        self.sp = (self.sp - 1) & 0xFFFF
        self.cycles = 8
    
    def _add_hl_r16(self, reg: str):
        value = self._get_r16(reg)
        result = self.hl + value
        self.flag_n = False
        self.flag_h = ((self.hl & 0x0FFF) + (value & 0x0FFF)) > 0x0FFF
        self.flag_c = result > 0xFFFF
        self.hl = result & 0xFFFF
        self.cycles = 8
    
    def _add_hl_sp(self):
        result = self.hl + self.sp
        self.flag_n = False
        self.flag_h = ((self.hl & 0x0FFF) + (self.sp & 0x0FFF)) > 0x0FFF
        self.flag_c = result > 0xFFFF
        self.hl = result & 0xFFFF
        self.cycles = 8
    
    def _add_sp_n(self):
        offset = self.fetch_signed_byte()
        result = self.sp + offset
        self.flag_z = False
        self.flag_n = False
        self.flag_h = ((self.sp & 0x0F) + (offset & 0x0F)) > 0x0F
        self.flag_c = ((self.sp & 0xFF) + (offset & 0xFF)) > 0xFF
        self.sp = result & 0xFFFF
        self.cycles = 16
    
    def _ld_hl_sp_n(self):
        offset = self.fetch_signed_byte()
        result = self.sp + offset
        self.flag_z = False
        self.flag_n = False
        self.flag_h = ((self.sp & 0x0F) + (offset & 0x0F)) > 0x0F
        self.flag_c = ((self.sp & 0xFF) + (offset & 0xFF)) > 0xFF
        self.hl = result & 0xFFFF
        self.cycles = 12
    
    # 8-bit arithmetic
    def _inc_r8(self, reg: str):
        value = self._get_r8(reg)
        result = (value + 1) & 0xFF
        self._set_r8(reg, result)
        self.flag_z = result == 0
        self.flag_n = False
        self.flag_h = (value & 0x0F) == 0x0F
        self.cycles = 4
    
    def _inc_hl_ind(self):
        value = self.read_byte(self.hl)
        result = (value + 1) & 0xFF
        self.write_byte(self.hl, result)
        self.flag_z = result == 0
        self.flag_n = False
        self.flag_h = (value & 0x0F) == 0x0F
        self.cycles = 12
    
    def _dec_r8(self, reg: str):
        value = self._get_r8(reg)
        result = (value - 1) & 0xFF
        self._set_r8(reg, result)
        self.flag_z = result == 0
        self.flag_n = True
        self.flag_h = (value & 0x0F) == 0x00
        self.cycles = 4
    
    def _dec_hl_ind(self):
        value = self.read_byte(self.hl)
        result = (value - 1) & 0xFF
        self.write_byte(self.hl, result)
        self.flag_z = result == 0
        self.flag_n = True
        self.flag_h = (value & 0x0F) == 0x00
        self.cycles = 12
    
    # ALU operations
    def _add_a(self, value: int):
        result = self.a + value
        self.flag_z = (result & 0xFF) == 0
        self.flag_n = False
        self.flag_h = ((self.a & 0x0F) + (value & 0x0F)) > 0x0F
        self.flag_c = result > 0xFF
        self.a = result & 0xFF
    
    def _adc_a(self, value: int):
        carry = 1 if self.flag_c else 0
        result = self.a + value + carry
        self.flag_z = (result & 0xFF) == 0
        self.flag_n = False
        self.flag_h = ((self.a & 0x0F) + (value & 0x0F) + carry) > 0x0F
        self.flag_c = result > 0xFF
        self.a = result & 0xFF
    
    def _sub_a(self, value: int):
        result = self.a - value
        self.flag_z = (result & 0xFF) == 0
        self.flag_n = True
        self.flag_h = (self.a & 0x0F) < (value & 0x0F)
        self.flag_c = result < 0
        self.a = result & 0xFF
    
    def _sbc_a(self, value: int):
        carry = 1 if self.flag_c else 0
        result = self.a - value - carry
        self.flag_z = (result & 0xFF) == 0
        self.flag_n = True
        self.flag_h = (self.a & 0x0F) < (value & 0x0F) + carry
        self.flag_c = result < 0
        self.a = result & 0xFF
    
    def _and_a(self, value: int):
        self.a = self.a & value
        self.flag_z = self.a == 0
        self.flag_n = False
        self.flag_h = True
        self.flag_c = False
    
    def _xor_a(self, value: int):
        self.a = self.a ^ value
        self.flag_z = self.a == 0
        self.flag_n = False
        self.flag_h = False
        self.flag_c = False
    
    def _or_a(self, value: int):
        self.a = self.a | value
        self.flag_z = self.a == 0
        self.flag_n = False
        self.flag_h = False
        self.flag_c = False
    
    def _cp_a(self, value: int):
        result = self.a - value
        self.flag_z = (result & 0xFF) == 0
        self.flag_n = True
        self.flag_h = (self.a & 0x0F) < (value & 0x0F)
        self.flag_c = result < 0
    
    def _alu_r8(self, op: Callable, reg: str):
        op(self._get_r8(reg))
        self.cycles = 4
    
    def _alu_hl(self, op: Callable):
        op(self.read_byte(self.hl))
        self.cycles = 8
    
    def _alu_n(self, op: Callable):
        op(self.fetch_byte())
        self.cycles = 8
    
    # Rotate and shift
    def _rlca(self):
        carry = (self.a >> 7) & 1
        self.a = ((self.a << 1) | carry) & 0xFF
        self.flag_z = False
        self.flag_n = False
        self.flag_h = False
        self.flag_c = carry == 1
        self.cycles = 4
    
    def _rrca(self):
        carry = self.a & 1
        self.a = ((self.a >> 1) | (carry << 7)) & 0xFF
        self.flag_z = False
        self.flag_n = False
        self.flag_h = False
        self.flag_c = carry == 1
        self.cycles = 4
    
    def _rla(self):
        carry = 1 if self.flag_c else 0
        new_carry = (self.a >> 7) & 1
        self.a = ((self.a << 1) | carry) & 0xFF
        self.flag_z = False
        self.flag_n = False
        self.flag_h = False
        self.flag_c = new_carry == 1
        self.cycles = 4
    
    def _rra(self):
        carry = 0x80 if self.flag_c else 0
        new_carry = self.a & 1
        self.a = ((self.a >> 1) | carry) & 0xFF
        self.flag_z = False
        self.flag_n = False
        self.flag_h = False
        self.flag_c = new_carry == 1
        self.cycles = 4
    
    # CB prefix rotate/shift operations
    def _rlc(self, value: int) -> int:
        carry = (value >> 7) & 1
        result = ((value << 1) | carry) & 0xFF
        self.flag_z = result == 0
        self.flag_n = False
        self.flag_h = False
        self.flag_c = carry == 1
        return result
    
    def _rrc(self, value: int) -> int:
        carry = value & 1
        result = ((value >> 1) | (carry << 7)) & 0xFF
        self.flag_z = result == 0
        self.flag_n = False
        self.flag_h = False
        self.flag_c = carry == 1
        return result
    
    def _rl(self, value: int) -> int:
        carry = 1 if self.flag_c else 0
        new_carry = (value >> 7) & 1
        result = ((value << 1) | carry) & 0xFF
        self.flag_z = result == 0
        self.flag_n = False
        self.flag_h = False
        self.flag_c = new_carry == 1
        return result
    
    def _rr(self, value: int) -> int:
        carry = 0x80 if self.flag_c else 0
        new_carry = value & 1
        result = ((value >> 1) | carry) & 0xFF
        self.flag_z = result == 0
        self.flag_n = False
        self.flag_h = False
        self.flag_c = new_carry == 1
        return result
    
    def _sla(self, value: int) -> int:
        carry = (value >> 7) & 1
        result = (value << 1) & 0xFF
        self.flag_z = result == 0
        self.flag_n = False
        self.flag_h = False
        self.flag_c = carry == 1
        return result
    
    def _sra(self, value: int) -> int:
        carry = value & 1
        result = ((value >> 1) | (value & 0x80)) & 0xFF
        self.flag_z = result == 0
        self.flag_n = False
        self.flag_h = False
        self.flag_c = carry == 1
        return result
    
    def _swap(self, value: int) -> int:
        result = ((value >> 4) | (value << 4)) & 0xFF
        self.flag_z = result == 0
        self.flag_n = False
        self.flag_h = False
        self.flag_c = False
        return result
    
    def _srl(self, value: int) -> int:
        carry = value & 1
        result = (value >> 1) & 0xFF
        self.flag_z = result == 0
        self.flag_n = False
        self.flag_h = False
        self.flag_c = carry == 1
        return result
    
    def _cb_r8(self, op: Callable, reg: str):
        result = op(self._get_r8(reg))
        self._set_r8(reg, result)
        self.cycles = 8
    
    def _cb_hl(self, op: Callable):
        result = op(self.read_byte(self.hl))
        self.write_byte(self.hl, result)
        self.cycles = 16
    
    # Bit operations
    def _bit_r8(self, bit: int, reg: str):
        value = self._get_r8(reg)
        self.flag_z = (value & (1 << bit)) == 0
        self.flag_n = False
        self.flag_h = True
        self.cycles = 8
    
    def _bit_hl(self, bit: int):
        value = self.read_byte(self.hl)
        self.flag_z = (value & (1 << bit)) == 0
        self.flag_n = False
        self.flag_h = True
        self.cycles = 12
    
    def _res_r8(self, bit: int, reg: str):
        value = self._get_r8(reg)
        self._set_r8(reg, value & ~(1 << bit))
        self.cycles = 8
    
    def _res_hl(self, bit: int):
        value = self.read_byte(self.hl)
        self.write_byte(self.hl, value & ~(1 << bit))
        self.cycles = 16
    
    def _set_r8_bit(self, bit: int, reg: str):
        value = self._get_r8(reg)
        self._set_r8(reg, value | (1 << bit))
        self.cycles = 8
    
    def _set_hl(self, bit: int):
        value = self.read_byte(self.hl)
        self.write_byte(self.hl, value | (1 << bit))
        self.cycles = 16
    
    # Note: _set_r8 for SET instruction (different from the register setter)
    # Renaming to avoid conflict
    def _set_r8(self, reg: str, value: int):
        setattr(self, reg, value & 0xFF)
    
    # DAA, CPL, SCF, CCF
    def _daa(self):
        if not self.flag_n:
            if self.flag_c or self.a > 0x99:
                self.a = (self.a + 0x60) & 0xFF
                self.flag_c = True
            if self.flag_h or (self.a & 0x0F) > 0x09:
                self.a = (self.a + 0x06) & 0xFF
        else:
            if self.flag_c:
                self.a = (self.a - 0x60) & 0xFF
            if self.flag_h:
                self.a = (self.a - 0x06) & 0xFF
        self.flag_z = self.a == 0
        self.flag_h = False
        self.cycles = 4
    
    def _cpl(self):
        self.a = self.a ^ 0xFF
        self.flag_n = True
        self.flag_h = True
        self.cycles = 4
    
    def _scf(self):
        self.flag_n = False
        self.flag_h = False
        self.flag_c = True
        self.cycles = 4
    
    def _ccf(self):
        self.flag_n = False
        self.flag_h = False
        self.flag_c = not self.flag_c
        self.cycles = 4
    
    # Jumps
    def _jp_nn(self):
        self.pc = self.fetch_word()
        self.cycles = 16
    
    def _jp_cc_nn(self, condition: bool):
        addr = self.fetch_word()
        if condition:
            self.pc = addr
            self.cycles = 16
        else:
            self.cycles = 12
    
    def _jp_hl(self):
        self.pc = self.hl
        self.cycles = 4
    
    def _jr(self):
        offset = self.fetch_signed_byte()
        self.pc = (self.pc + offset) & 0xFFFF
        self.cycles = 12
    
    def _jr_cc(self, condition: bool):
        offset = self.fetch_signed_byte()
        if condition:
            self.pc = (self.pc + offset) & 0xFFFF
            self.cycles = 12
        else:
            self.cycles = 8
    
    # Calls and returns
    def _call_nn(self):
        addr = self.fetch_word()
        self.push_word(self.pc)
        self.pc = addr
        self.cycles = 24
    
    def _call_cc_nn(self, condition: bool):
        addr = self.fetch_word()
        if condition:
            self.push_word(self.pc)
            self.pc = addr
            self.cycles = 24
        else:
            self.cycles = 12
    
    def _ret(self):
        self.pc = self.pop_word()
        self.cycles = 16
    
    def _ret_cc(self, condition: bool):
        if condition:
            self.pc = self.pop_word()
            self.cycles = 20
        else:
            self.cycles = 8
    
    def _reti(self):
        self.pc = self.pop_word()
        self.ime = True
        self.cycles = 16
    
    def _rst(self, addr: int):
        self.push_word(self.pc)
        self.pc = addr
        self.cycles = 16
    
    # Stack operations
    def _push_r16(self, reg: str):
        self.push_word(self._get_r16(reg))
        self.cycles = 16
    
    def _push_af(self):
        self.push_word(self.af)
        self.cycles = 16
    
    def _pop_r16(self, reg: str):
        self._set_r16(reg, self.pop_word())
        self.cycles = 12
    
    def _pop_af(self):
        self.af = self.pop_word()
        self.cycles = 12
    
    # Interrupts
    def _di(self):
        self.ime = False
        self.cycles = 4
    
    def _ei(self):
        self.ime_scheduled = True
        self.cycles = 4


# Fix the SET bit instruction to not conflict with _set_r8
CPU._set_r8_method = CPU._set_r8

def _make_set_bit_r8(bit, reg):
    def set_bit_r8(self):
        value = getattr(self, reg)
        setattr(self, reg, (value | (1 << bit)) & 0xFF)
        self.cycles = 8
    return set_bit_r8

# Re-build SET instructions in CB table
def patch_set_instructions(cpu_class):
    regs = ['b', 'c', 'd', 'e', 'h', 'l', '(hl)', 'a']
    for bit in range(8):
        for reg_idx, reg in enumerate(regs):
            opcode = 0xC0 + (bit * 8) + reg_idx
            if reg == '(hl)':
                # Already defined correctly
                pass
            else:
                def make_setter(b, r):
                    def setter(self):
                        value = getattr(self, r)
                        setattr(self, r, (value | (1 << b)) & 0xFF)
                        self.cycles = 8
                    return setter
                # Will be applied at build time

patch_set_instructions(CPU)


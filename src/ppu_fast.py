"""
Optimized PPU using Numba JIT compilation for fast rendering.
"""

import numpy as np
from numba import njit, prange
from typing import Optional, Callable


@njit(cache=True)
def render_bg_line_fast(
    line_buffer: np.ndarray,
    line_priority: np.ndarray,
    vram_bank0: np.ndarray,
    vram_bank1: np.ndarray,
    ly: int,
    scy: int,
    scx: int,
    bg_tilemap: int,
    tile_data_addr: int,
    tile_data_signed: bool,
    cgb_mode: bool
):
    """Fast background line rendering with Numba."""
    y = (ly + scy) & 0xFF
    tile_y = y >> 3
    pixel_y = y & 7
    
    for screen_x in range(160):
        x = (screen_x + scx) & 0xFF
        tile_x = x >> 3
        pixel_x = x & 7
        
        # Get tile index from tilemap
        tilemap_offset = (bg_tilemap - 0x8000) + tile_y * 32 + tile_x
        tile_idx = vram_bank0[tilemap_offset]
        
        # Get attributes for GBC
        if cgb_mode:
            attr = vram_bank1[tilemap_offset]
            palette = attr & 0x07
            vram_bank = (attr >> 3) & 0x01
            flip_x = (attr & 0x20) != 0
            flip_y = (attr & 0x40) != 0
            priority = (attr & 0x80) != 0
        else:
            palette = 0
            vram_bank = 0
            flip_x = False
            flip_y = False
            priority = False
        
        # Apply flip
        px = 7 - pixel_x if flip_x else pixel_x
        py = 7 - pixel_y if flip_y else pixel_y
        
        # Calculate tile address
        if tile_data_signed:
            if tile_idx > 127:
                tile_idx = tile_idx - 256
            tile_offset = (0x9000 - 0x8000) + tile_idx * 16
        else:
            tile_offset = tile_idx * 16
        
        # Get pixel color
        row_offset = tile_offset + py * 2
        if vram_bank == 0:
            low_byte = vram_bank0[row_offset]
            high_byte = vram_bank0[row_offset + 1]
        else:
            low_byte = vram_bank1[row_offset]
            high_byte = vram_bank1[row_offset + 1]
        
        bit = 7 - px
        color_idx = ((high_byte >> bit) & 1) << 1 | ((low_byte >> bit) & 1)
        
        line_buffer[screen_x] = (palette << 8) | color_idx
        line_priority[screen_x] = 1 if priority else 0


@njit(cache=True)
def render_window_line_fast(
    line_buffer: np.ndarray,
    line_priority: np.ndarray,
    vram_bank0: np.ndarray,
    vram_bank1: np.ndarray,
    window_line: int,
    wx: int,
    window_tilemap: int,
    tile_data_addr: int,
    tile_data_signed: bool,
    cgb_mode: bool
) -> bool:
    """Fast window line rendering."""
    window_x_start = wx - 7
    if window_x_start >= 160:
        return False
    
    pixel_y = window_line & 7
    tile_y = window_line >> 3
    
    rendered = False
    start_x = max(0, window_x_start)
    
    for screen_x in range(start_x, 160):
        x = screen_x - window_x_start
        tile_x = x >> 3
        pixel_x = x & 7
        
        tilemap_offset = (window_tilemap - 0x8000) + tile_y * 32 + tile_x
        tile_idx = vram_bank0[tilemap_offset]
        
        if cgb_mode:
            attr = vram_bank1[tilemap_offset]
            palette = attr & 0x07
            vram_bank = (attr >> 3) & 0x01
            flip_x = (attr & 0x20) != 0
            flip_y = (attr & 0x40) != 0
            priority = (attr & 0x80) != 0
        else:
            palette = 0
            vram_bank = 0
            flip_x = False
            flip_y = False
            priority = False
        
        px = 7 - pixel_x if flip_x else pixel_x
        py = 7 - pixel_y if flip_y else pixel_y
        
        if tile_data_signed:
            if tile_idx > 127:
                tile_idx = tile_idx - 256
            tile_offset = (0x9000 - 0x8000) + tile_idx * 16
        else:
            tile_offset = tile_idx * 16
        
        row_offset = tile_offset + py * 2
        if vram_bank == 0:
            low_byte = vram_bank0[row_offset]
            high_byte = vram_bank0[row_offset + 1]
        else:
            low_byte = vram_bank1[row_offset]
            high_byte = vram_bank1[row_offset + 1]
        
        bit = 7 - px
        color_idx = ((high_byte >> bit) & 1) << 1 | ((low_byte >> bit) & 1)
        
        line_buffer[screen_x] = (palette << 8) | color_idx
        line_priority[screen_x] = 1 if priority else 0
        rendered = True
    
    return rendered


@njit(cache=True)
def finalize_scanline_fast(
    framebuffer: np.ndarray,
    line_buffer: np.ndarray,
    ly: int,
    cgb_mode: bool,
    bg_palette_ram: np.ndarray,
    obj_palette_ram: np.ndarray,
    bgp: int,
    obp0: int,
    obp1: int
):
    """Fast scanline finalization - convert to RGB."""
    # DMG colors
    dmg_colors = np.array([
        [224, 248, 208],
        [136, 192, 112],
        [52, 104, 86],
        [8, 24, 32]
    ], dtype=np.uint8)
    
    for x in range(160):
        value = line_buffer[x]
        is_sprite = (value & 0x8000) != 0
        palette = (value >> 8) & 0x07
        color_idx = value & 0x03
        
        if cgb_mode:
            # GBC color
            if is_sprite:
                offset = palette * 8 + color_idx * 2
                lo = obj_palette_ram[offset]
                hi = obj_palette_ram[offset + 1]
            else:
                offset = palette * 8 + color_idx * 2
                lo = bg_palette_ram[offset]
                hi = bg_palette_ram[offset + 1]
            
            rgb15 = (hi << 8) | lo
            r = (rgb15 & 0x1F) << 3
            g = ((rgb15 >> 5) & 0x1F) << 3
            b = ((rgb15 >> 10) & 0x1F) << 3
            
            framebuffer[ly, x, 0] = r
            framebuffer[ly, x, 1] = g
            framebuffer[ly, x, 2] = b
        else:
            # DMG color
            if is_sprite:
                pal = obp1 if palette else obp0
            else:
                pal = bgp
            
            mapped_color = (pal >> (color_idx * 2)) & 0x03
            framebuffer[ly, x, 0] = dmg_colors[mapped_color, 0]
            framebuffer[ly, x, 1] = dmg_colors[mapped_color, 1]
            framebuffer[ly, x, 2] = dmg_colors[mapped_color, 2]


@njit(cache=True)
def render_tilemap_fast(
    image: np.ndarray,
    vram_bank0: np.ndarray,
    vram_bank1: np.ndarray,
    tilemap_addr: int,
    tile_data_signed: bool,
    cgb_mode: bool,
    bg_palette_ram: np.ndarray,
    bgp: int
):
    """Fast tilemap rendering for debug view."""
    dmg_colors = np.array([
        [224, 248, 208],
        [136, 192, 112],
        [52, 104, 86],
        [8, 24, 32]
    ], dtype=np.uint8)
    
    for ty in range(32):
        for tx in range(32):
            tilemap_offset = (tilemap_addr - 0x8000) + ty * 32 + tx
            tile_idx = vram_bank0[tilemap_offset]
            
            if cgb_mode:
                attr = vram_bank1[tilemap_offset]
                palette = attr & 0x07
                vram_bank = (attr >> 3) & 0x01
                flip_x = (attr & 0x20) != 0
                flip_y = (attr & 0x40) != 0
            else:
                palette = 0
                vram_bank = 0
                flip_x = False
                flip_y = False
            
            # Calculate tile address
            if tile_data_signed:
                if tile_idx > 127:
                    tile_idx = tile_idx - 256
                tile_offset = (0x9000 - 0x8000) + tile_idx * 16
            else:
                tile_offset = tile_idx * 16
            
            for py in range(8):
                for px in range(8):
                    rpx = 7 - px if flip_x else px
                    rpy = 7 - py if flip_y else py
                    
                    row_offset = tile_offset + rpy * 2
                    if vram_bank == 0:
                        low_byte = vram_bank0[row_offset]
                        high_byte = vram_bank0[row_offset + 1]
                    else:
                        low_byte = vram_bank1[row_offset]
                        high_byte = vram_bank1[row_offset + 1]
                    
                    bit = 7 - rpx
                    color_idx = ((high_byte >> bit) & 1) << 1 | ((low_byte >> bit) & 1)
                    
                    img_y = ty * 8 + py
                    img_x = tx * 8 + px
                    
                    if cgb_mode:
                        offset = palette * 8 + color_idx * 2
                        lo = bg_palette_ram[offset]
                        hi = bg_palette_ram[offset + 1]
                        rgb15 = (hi << 8) | lo
                        image[img_y, img_x, 0] = (rgb15 & 0x1F) << 3
                        image[img_y, img_x, 1] = ((rgb15 >> 5) & 0x1F) << 3
                        image[img_y, img_x, 2] = ((rgb15 >> 10) & 0x1F) << 3
                    else:
                        mapped = (bgp >> (color_idx * 2)) & 0x03
                        image[img_y, img_x, 0] = dmg_colors[mapped, 0]
                        image[img_y, img_x, 1] = dmg_colors[mapped, 1]
                        image[img_y, img_x, 2] = dmg_colors[mapped, 2]


class PPUFast:
    """
    Optimized PPU using Numba JIT compilation.
    """
    
    MODE_HBLANK = 0
    MODE_VBLANK = 1
    MODE_OAM = 2
    MODE_TRANSFER = 3
    
    SCREEN_WIDTH = 160
    SCREEN_HEIGHT = 144
    
    CYCLES_OAM = 80
    CYCLES_TRANSFER = 172
    CYCLES_HBLANK = 204
    CYCLES_SCANLINE = 456
    SCANLINES_VISIBLE = 144
    SCANLINES_TOTAL = 154
    
    def __init__(self, memory):
        self.memory = memory
        
        # Frame buffer (RGB)
        self.framebuffer = np.zeros((self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        
        # Line buffers
        self.line_buffer = np.zeros(self.SCREEN_WIDTH, dtype=np.int32)
        self.line_priority = np.zeros(self.SCREEN_WIDTH, dtype=np.uint8)
        
        # State
        self.mode = self.MODE_OAM
        self.cycles = 0
        self.ly = 0
        self.window_line = 0
        
        # Sprite cache
        self.sprite_buffer = []
        
        # Callbacks
        self.on_vblank: Optional[Callable] = None
        self.on_hblank: Optional[Callable] = None
        self.on_frame_complete: Optional[Callable] = None
        
        self.frame_count = 0
    
    @property
    def lcdc(self) -> int:
        return int(self.memory.io[0x40])
    
    @property
    def stat(self) -> int:
        return int(self.memory.io[0x41])
    
    @property
    def scy(self) -> int:
        return int(self.memory.io[0x42])
    
    @property
    def scx(self) -> int:
        return int(self.memory.io[0x43])
    
    @property
    def lyc(self) -> int:
        return int(self.memory.io[0x45])
    
    @property
    def bgp(self) -> int:
        return int(self.memory.io[0x47])
    
    @property
    def obp0(self) -> int:
        return int(self.memory.io[0x48])
    
    @property
    def obp1(self) -> int:
        return int(self.memory.io[0x49])
    
    @property
    def wy(self) -> int:
        return int(self.memory.io[0x4A])
    
    @property
    def wx(self) -> int:
        return int(self.memory.io[0x4B])
    
    @property
    def lcd_enabled(self) -> bool:
        return (self.lcdc & 0x80) != 0
    
    @property
    def window_tilemap(self) -> int:
        return 0x9C00 if (self.lcdc & 0x40) else 0x9800
    
    @property
    def window_enabled(self) -> bool:
        return (self.lcdc & 0x20) != 0
    
    @property
    def tile_data_addr(self) -> int:
        return 0x8000 if (self.lcdc & 0x10) else 0x8800
    
    @property
    def tile_data_signed(self) -> bool:
        return (self.lcdc & 0x10) == 0
    
    @property
    def bg_tilemap(self) -> int:
        return 0x9C00 if (self.lcdc & 0x08) else 0x9800
    
    @property
    def sprite_height(self) -> int:
        return 16 if (self.lcdc & 0x04) else 8
    
    @property
    def sprites_enabled(self) -> bool:
        return (self.lcdc & 0x02) != 0
    
    @property
    def bg_enabled(self) -> bool:
        return (self.lcdc & 0x01) != 0
    
    def step(self, cycles: int) -> int:
        if not self.lcd_enabled:
            return 0
        
        interrupts = 0
        self.cycles += cycles
        
        while self.cycles >= self._mode_cycles():
            self.cycles -= self._mode_cycles()
            interrupts |= self._advance_mode()
        
        return interrupts
    
    def _mode_cycles(self) -> int:
        if self.mode == self.MODE_OAM:
            return self.CYCLES_OAM
        elif self.mode == self.MODE_TRANSFER:
            return self.CYCLES_TRANSFER
        elif self.mode == self.MODE_HBLANK:
            return self.CYCLES_HBLANK
        else:
            return self.CYCLES_SCANLINE
    
    def _advance_mode(self) -> int:
        interrupts = 0
        
        if self.mode == self.MODE_OAM:
            self._scan_oam()
            self.mode = self.MODE_TRANSFER
            self._update_stat()
            
        elif self.mode == self.MODE_TRANSFER:
            self._render_scanline()
            self.mode = self.MODE_HBLANK
            self._update_stat()
            
            if self.stat & 0x08:
                interrupts |= 0x02
            
            if self.on_hblank:
                self.on_hblank()
            
            if self.memory.cgb_mode:
                self.memory.do_hdma_transfer()
            
        elif self.mode == self.MODE_HBLANK:
            self.ly += 1
            self.memory.io[0x44] = self.ly
            
            if self.ly == self.SCANLINES_VISIBLE:
                self.mode = self.MODE_VBLANK
                self._update_stat()
                interrupts |= 0x01
                
                if self.stat & 0x10:
                    interrupts |= 0x02
                
                if self.on_vblank:
                    self.on_vblank()
                
                self.frame_count += 1
                if self.on_frame_complete:
                    self.on_frame_complete(self.framebuffer)
            else:
                self.mode = self.MODE_OAM
                self._update_stat()
                
                if self.stat & 0x20:
                    interrupts |= 0x02
            
            interrupts |= self._check_lyc()
            
        else:  # VBLANK
            self.ly += 1
            self.memory.io[0x44] = self.ly
            
            if self.ly >= self.SCANLINES_TOTAL:
                self.ly = 0
                self.window_line = 0
                self.memory.io[0x44] = 0
                self.mode = self.MODE_OAM
                self._update_stat()
                
                if self.stat & 0x20:
                    interrupts |= 0x02
            
            interrupts |= self._check_lyc()
        
        return interrupts
    
    def _update_stat(self):
        stat = int(self.memory.io[0x41])
        stat = (stat & 0xFC) | self.mode
        self.memory.io[0x41] = stat
    
    def _check_lyc(self) -> int:
        if self.ly == self.lyc:
            self.memory.io[0x41] = int(self.memory.io[0x41]) | 0x04
            if self.stat & 0x40:
                return 0x02
        else:
            self.memory.io[0x41] = int(self.memory.io[0x41]) & 0xFB
        return 0
    
    def _scan_oam(self):
        self.sprite_buffer = []
        sprite_height = self.sprite_height
        
        for i in range(40):
            y = int(self.memory.oam[i * 4]) - 16
            x = int(self.memory.oam[i * 4 + 1]) - 8
            tile = int(self.memory.oam[i * 4 + 2])
            attr = int(self.memory.oam[i * 4 + 3])
            
            if y <= self.ly < y + sprite_height:
                self.sprite_buffer.append({
                    'y': y, 'x': x, 'tile': tile, 'attr': attr, 'oam_idx': i
                })
            
            if len(self.sprite_buffer) >= 10:
                break
        
        if self.memory.cgb_mode:
            self.sprite_buffer.sort(key=lambda s: s['oam_idx'])
        else:
            self.sprite_buffer.sort(key=lambda s: (s['x'], s['oam_idx']))
    
    def _render_scanline(self):
        if not self.lcd_enabled:
            self.framebuffer[self.ly] = 255
            return
        
        self.line_buffer.fill(0)
        self.line_priority.fill(0)
        
        # Use JIT-compiled functions
        if self.bg_enabled or self.memory.cgb_mode:
            render_bg_line_fast(
                self.line_buffer,
                self.line_priority,
                self.memory.vram[0],
                self.memory.vram[1],
                self.ly,
                self.scy,
                self.scx,
                self.bg_tilemap,
                self.tile_data_addr,
                self.tile_data_signed,
                self.memory.cgb_mode
            )
            
            if self.window_enabled and self.ly >= self.wy:
                rendered = render_window_line_fast(
                    self.line_buffer,
                    self.line_priority,
                    self.memory.vram[0],
                    self.memory.vram[1],
                    self.window_line,
                    self.wx,
                    self.window_tilemap,
                    self.tile_data_addr,
                    self.tile_data_signed,
                    self.memory.cgb_mode
                )
                if rendered:
                    self.window_line += 1
        
        if self.sprites_enabled:
            self._render_sprites()
        
        # Fast finalization
        finalize_scanline_fast(
            self.framebuffer,
            self.line_buffer,
            self.ly,
            self.memory.cgb_mode,
            self.memory.bg_palette_ram,
            self.memory.obj_palette_ram,
            self.bgp,
            self.obp0,
            self.obp1
        )
    
    def _render_sprites(self):
        sprite_height = self.sprite_height
        
        for sprite in reversed(self.sprite_buffer):
            x = sprite['x']
            y = sprite['y']
            tile = sprite['tile']
            attr = sprite['attr']
            
            if x <= -8 or x >= 160:
                continue
            
            if self.memory.cgb_mode:
                palette = attr & 0x07
                vram_bank = (attr >> 3) & 0x01
            else:
                palette = (attr >> 4) & 0x01
                vram_bank = 0
            
            flip_x = (attr & 0x20) != 0
            flip_y = (attr & 0x40) != 0
            bg_priority = (attr & 0x80) != 0
            
            row = self.ly - y
            if flip_y:
                row = sprite_height - 1 - row
            
            if sprite_height == 16:
                if row < 8:
                    tile = tile & 0xFE
                else:
                    tile = tile | 0x01
                    row -= 8
            
            tile_addr = 0x8000 + tile * 16
            row_addr = tile_addr + row * 2 - 0x8000
            
            if vram_bank == 0:
                low_byte = int(self.memory.vram[0][row_addr])
                high_byte = int(self.memory.vram[0][row_addr + 1])
            else:
                low_byte = int(self.memory.vram[1][row_addr])
                high_byte = int(self.memory.vram[1][row_addr + 1])
            
            for px in range(8):
                screen_x = x + px
                if screen_x < 0 or screen_x >= 160:
                    continue
                
                col = 7 - px if flip_x else px
                bit = 7 - col
                color_idx = ((high_byte >> bit) & 1) << 1 | ((low_byte >> bit) & 1)
                
                if color_idx == 0:
                    continue
                
                bg_color = self.line_buffer[screen_x] & 0xFF
                bg_has_priority = self.line_priority[screen_x] != 0
                
                if self.memory.cgb_mode:
                    if bg_has_priority and bg_color != 0:
                        continue
                    if bg_priority and bg_color != 0:
                        continue
                else:
                    if bg_priority and bg_color != 0:
                        continue
                
                self.line_buffer[screen_x] = 0x8000 | (palette << 8) | color_idx
    
    def get_tilemap_image(self, tilemap_addr: int = None) -> np.ndarray:
        if tilemap_addr is None:
            tilemap_addr = self.bg_tilemap
        
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        
        render_tilemap_fast(
            image,
            self.memory.vram[0],
            self.memory.vram[1],
            tilemap_addr,
            self.tile_data_signed,
            self.memory.cgb_mode,
            self.memory.bg_palette_ram,
            self.bgp
        )
        
        return image
    
    def get_tiles_image(self, bank: int = 0) -> np.ndarray:
        image = np.zeros((192, 128, 3), dtype=np.uint8)
        
        for tile_idx in range(384):
            tx = (tile_idx % 16) * 8
            ty = (tile_idx // 16) * 8
            
            tile_offset = tile_idx * 16
            
            for py in range(8):
                for px in range(8):
                    row_offset = tile_offset + py * 2
                    low_byte = int(self.memory.vram[bank][row_offset])
                    high_byte = int(self.memory.vram[bank][row_offset + 1])
                    
                    bit = 7 - px
                    color_idx = ((high_byte >> bit) & 1) << 1 | ((low_byte >> bit) & 1)
                    
                    gray = 255 - (color_idx * 85)
                    image[ty + py, tx + px] = [gray, gray, gray]
        
        return image


"""
Game Boy Color Pixel Processing Unit (PPU)
Handles tile rendering, sprites, backgrounds, and color palettes.
"""

import numpy as np
from typing import Optional, Callable, Tuple


class PPU:
    """
    GBC PPU - Renders graphics to a 160x144 pixel display.
    
    The PPU has several rendering modes:
    - Mode 0: H-Blank (204 cycles)
    - Mode 1: V-Blank (4560 cycles total, 10 scanlines)
    - Mode 2: OAM Search (80 cycles)
    - Mode 3: Pixel Transfer (172-289 cycles, variable)
    
    Each scanline takes 456 cycles.
    """
    
    # LCD modes
    MODE_HBLANK = 0
    MODE_VBLANK = 1
    MODE_OAM = 2
    MODE_TRANSFER = 3
    
    # Screen dimensions
    SCREEN_WIDTH = 160
    SCREEN_HEIGHT = 144
    
    # Timing constants
    CYCLES_OAM = 80
    CYCLES_TRANSFER = 172
    CYCLES_HBLANK = 204
    CYCLES_SCANLINE = 456
    SCANLINES_VISIBLE = 144
    SCANLINES_TOTAL = 154
    
    # DMG palette colors (grayscale)
    DMG_COLORS = np.array([
        [224, 248, 208],  # White
        [136, 192, 112],  # Light gray
        [52, 104, 86],    # Dark gray
        [8, 24, 32],      # Black
    ], dtype=np.uint8)
    
    def __init__(self, memory):
        self.memory = memory
        
        # Frame buffer (RGB)
        self.framebuffer = np.zeros((self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        
        # Internal line buffer for priority handling (use int32 to avoid overflow)
        self.line_buffer = np.zeros(self.SCREEN_WIDTH, dtype=np.int32)
        self.line_priority = np.zeros(self.SCREEN_WIDTH, dtype=np.uint8)
        
        # PPU state
        self.mode = self.MODE_OAM
        self.cycles = 0
        self.ly = 0  # Current scanline
        self.window_line = 0  # Window internal line counter
        
        # Cached palette data for GBC
        self.bg_palettes = np.zeros((8, 4, 3), dtype=np.uint8)
        self.obj_palettes = np.zeros((8, 4, 3), dtype=np.uint8)
        
        # Sprite cache for current scanline
        self.sprite_buffer = []
        
        # Callbacks
        self.on_vblank: Optional[Callable] = None
        self.on_hblank: Optional[Callable] = None
        self.on_frame_complete: Optional[Callable] = None
        
        # Statistics
        self.frame_count = 0
        
        # Tilemap viewer cache
        self.tilemap_cache = None
        self.tiles_cache = None
    
    @property
    def lcdc(self) -> int:
        """LCD Control register."""
        return int(self.memory.io[0x40])
    
    @property
    def stat(self) -> int:
        """LCD Status register."""
        return int(self.memory.io[0x41])
    
    @stat.setter
    def stat(self, value: int):
        # Only bits 3-6 are writable
        current = int(self.memory.io[0x41])
        self.memory.io[0x41] = (value & 0x78) | (current & 0x07)
    
    @property
    def scy(self) -> int:
        """Scroll Y."""
        return int(self.memory.io[0x42])
    
    @property
    def scx(self) -> int:
        """Scroll X."""
        return int(self.memory.io[0x43])
    
    @property
    def lyc(self) -> int:
        """LY Compare."""
        return int(self.memory.io[0x45])
    
    @property
    def bgp(self) -> int:
        """BG Palette (DMG)."""
        return int(self.memory.io[0x47])
    
    @property
    def obp0(self) -> int:
        """OBJ Palette 0 (DMG)."""
        return int(self.memory.io[0x48])
    
    @property
    def obp1(self) -> int:
        """OBJ Palette 1 (DMG)."""
        return int(self.memory.io[0x49])
    
    @property
    def wy(self) -> int:
        """Window Y position."""
        return int(self.memory.io[0x4A])
    
    @property
    def wx(self) -> int:
        """Window X position."""
        return int(self.memory.io[0x4B])
    
    @property
    def lcd_enabled(self) -> bool:
        return (self.lcdc & 0x80) != 0
    
    @property
    def window_tilemap(self) -> int:
        """Window tilemap address: 0x9800 or 0x9C00."""
        return 0x9C00 if (self.lcdc & 0x40) else 0x9800
    
    @property
    def window_enabled(self) -> bool:
        return (self.lcdc & 0x20) != 0
    
    @property
    def tile_data_addr(self) -> int:
        """Tile data address: 0x8000 (unsigned) or 0x8800 (signed)."""
        return 0x8000 if (self.lcdc & 0x10) else 0x8800
    
    @property
    def tile_data_signed(self) -> bool:
        """Whether tile indices are signed (-128 to 127)."""
        return (self.lcdc & 0x10) == 0
    
    @property
    def bg_tilemap(self) -> int:
        """BG tilemap address: 0x9800 or 0x9C00."""
        return 0x9C00 if (self.lcdc & 0x08) else 0x9800
    
    @property
    def sprite_height(self) -> int:
        """Sprite height: 8 or 16 pixels."""
        return 16 if (self.lcdc & 0x04) else 8
    
    @property
    def sprites_enabled(self) -> bool:
        return (self.lcdc & 0x02) != 0
    
    @property
    def bg_enabled(self) -> bool:
        """BG/Window enable (or priority on GBC)."""
        return (self.lcdc & 0x01) != 0
    
    def step(self, cycles: int) -> int:
        """Advance PPU by given cycles. Returns interrupt flags to set."""
        if not self.lcd_enabled:
            return 0
        
        interrupts = 0
        self.cycles += cycles
        
        while self.cycles >= self._mode_cycles():
            self.cycles -= self._mode_cycles()
            interrupts |= self._advance_mode()
        
        return interrupts
    
    def _mode_cycles(self) -> int:
        """Get cycles for current mode."""
        if self.mode == self.MODE_OAM:
            return self.CYCLES_OAM
        elif self.mode == self.MODE_TRANSFER:
            return self.CYCLES_TRANSFER
        elif self.mode == self.MODE_HBLANK:
            return self.CYCLES_HBLANK
        else:  # VBLANK
            return self.CYCLES_SCANLINE
    
    def _advance_mode(self) -> int:
        """Advance to next mode. Returns interrupt flags."""
        interrupts = 0
        
        if self.mode == self.MODE_OAM:
            # OAM -> Transfer
            self._scan_oam()
            self.mode = self.MODE_TRANSFER
            self._update_stat()
            
        elif self.mode == self.MODE_TRANSFER:
            # Transfer -> H-Blank
            self._render_scanline()
            self.mode = self.MODE_HBLANK
            self._update_stat()
            
            if self.stat & 0x08:  # H-Blank interrupt
                interrupts |= 0x02  # STAT interrupt
            
            if self.on_hblank:
                self.on_hblank()
            
            # HDMA transfer during H-Blank
            if self.memory.cgb_mode:
                self.memory.do_hdma_transfer()
            
        elif self.mode == self.MODE_HBLANK:
            # H-Blank -> next line
            self.ly += 1
            self.memory.io[0x44] = self.ly
            
            if self.ly == self.SCANLINES_VISIBLE:
                # Enter V-Blank
                self.mode = self.MODE_VBLANK
                self._update_stat()
                interrupts |= 0x01  # V-Blank interrupt
                
                if self.stat & 0x10:  # V-Blank STAT interrupt
                    interrupts |= 0x02
                
                if self.on_vblank:
                    self.on_vblank()
                
                self.frame_count += 1
                if self.on_frame_complete:
                    self.on_frame_complete(self.framebuffer)
            else:
                # Next visible line
                self.mode = self.MODE_OAM
                self._update_stat()
                
                if self.stat & 0x20:  # OAM interrupt
                    interrupts |= 0x02
            
            # LYC check
            interrupts |= self._check_lyc()
            
        else:  # VBLANK
            self.ly += 1
            self.memory.io[0x44] = self.ly
            
            if self.ly >= self.SCANLINES_TOTAL:
                # End of V-Blank, start new frame
                self.ly = 0
                self.window_line = 0
                self.memory.io[0x44] = 0
                self.mode = self.MODE_OAM
                self._update_stat()
                
                if self.stat & 0x20:  # OAM interrupt
                    interrupts |= 0x02
            
            # LYC check
            interrupts |= self._check_lyc()
        
        return interrupts
    
    def _update_stat(self):
        """Update STAT register mode bits."""
        stat = int(self.memory.io[0x41])
        stat = (stat & 0xFC) | self.mode
        self.memory.io[0x41] = stat
    
    def _check_lyc(self) -> int:
        """Check LY=LYC and return interrupt if needed."""
        if self.ly == self.lyc:
            self.memory.io[0x41] = int(self.memory.io[0x41]) | 0x04  # Set coincidence flag
            if self.stat & 0x40:  # LYC interrupt enabled
                return 0x02
        else:
            self.memory.io[0x41] = int(self.memory.io[0x41]) & 0xFB  # Clear bit 2
        return 0
    
    def _scan_oam(self):
        """Scan OAM for sprites on current scanline."""
        self.sprite_buffer = []
        sprite_height = self.sprite_height
        
        for i in range(40):
            # OAM entry: Y, X, Tile, Attributes
            y = int(self.memory.oam[i * 4]) - 16
            x = int(self.memory.oam[i * 4 + 1]) - 8
            tile = int(self.memory.oam[i * 4 + 2])
            attr = int(self.memory.oam[i * 4 + 3])
            
            # Check if sprite is on current scanline
            if y <= self.ly < y + sprite_height:
                self.sprite_buffer.append({
                    'y': y,
                    'x': x,
                    'tile': tile,
                    'attr': attr,
                    'oam_idx': i
                })
            
            # Max 10 sprites per line
            if len(self.sprite_buffer) >= 10:
                break
        
        # Sort by X coordinate (lower X = higher priority)
        # On DMG, OAM index is secondary; on GBC, only OAM index matters
        if self.memory.cgb_mode:
            self.sprite_buffer.sort(key=lambda s: s['oam_idx'])
        else:
            self.sprite_buffer.sort(key=lambda s: (s['x'], s['oam_idx']))
    
    def _render_scanline(self):
        """Render current scanline to framebuffer."""
        if not self.lcd_enabled:
            self.framebuffer[self.ly] = 255  # White
            return
        
        # Clear line buffer
        self.line_buffer.fill(0)
        self.line_priority.fill(0)
        
        # Render layers
        if self.bg_enabled or self.memory.cgb_mode:
            self._render_bg()
            
            if self.window_enabled and self.ly >= self.wy:
                self._render_window()
        
        if self.sprites_enabled:
            self._render_sprites()
        
        # Convert to RGB
        self._finalize_scanline()
    
    def _render_bg(self):
        """Render background layer."""
        y = (int(self.ly) + int(self.scy)) & 0xFF
        tile_y = y >> 3
        pixel_y = y & 7
        
        for screen_x in range(self.SCREEN_WIDTH):
            x = (screen_x + int(self.scx)) & 0xFF
            tile_x = x >> 3
            pixel_x = x & 7
            
            # Get tile index from tilemap
            tilemap_addr = int(self.bg_tilemap) + tile_y * 32 + tile_x
            tile_idx = int(self._read_vram(tilemap_addr, 0))
            
            # GBC: Get tile attributes from bank 1
            if self.memory.cgb_mode:
                attr = int(self._read_vram(tilemap_addr, 1))
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
            
            # Get tile data
            color_idx = self._get_tile_pixel(tile_idx, pixel_x, pixel_y, 
                                              vram_bank, flip_x, flip_y)
            
            self.line_buffer[screen_x] = int((palette << 8) | color_idx)
            self.line_priority[screen_x] = 1 if priority else 0
    
    def _render_window(self):
        """Render window layer."""
        if self.wx > 166 or self.wy > 143:
            return
        
        window_x_start = self.wx - 7
        if window_x_start >= self.SCREEN_WIDTH:
            return
        
        pixel_y = self.window_line & 7
        tile_y = self.window_line >> 3
        
        rendered = False
        
        for screen_x in range(max(0, window_x_start), self.SCREEN_WIDTH):
            x = screen_x - window_x_start
            tile_x = x >> 3
            pixel_x = x & 7
            
            tilemap_addr = self.window_tilemap + tile_y * 32 + tile_x
            tile_idx = self._read_vram(tilemap_addr, 0)
            
            if self.memory.cgb_mode:
                attr = self._read_vram(tilemap_addr, 1)
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
            
            color_idx = self._get_tile_pixel(tile_idx, pixel_x, pixel_y,
                                              vram_bank, flip_x, flip_y)
            
            self.line_buffer[screen_x] = (palette << 8) | color_idx
            self.line_priority[screen_x] = 1 if priority else 0
            rendered = True
        
        if rendered:
            self.window_line += 1
    
    def _render_sprites(self):
        """Render sprite layer."""
        sprite_height = self.sprite_height
        
        # Render sprites in reverse order (later sprites have lower priority)
        for sprite in reversed(self.sprite_buffer):
            x = sprite['x']
            y = sprite['y']
            tile = sprite['tile']
            attr = sprite['attr']
            
            # Skip offscreen sprites
            if x <= -8 or x >= self.SCREEN_WIDTH:
                continue
            
            # Sprite attributes
            if self.memory.cgb_mode:
                palette = attr & 0x07
                vram_bank = (attr >> 3) & 0x01
            else:
                palette = (attr >> 4) & 0x01
                vram_bank = 0
            
            flip_x = (attr & 0x20) != 0
            flip_y = (attr & 0x40) != 0
            bg_priority = (attr & 0x80) != 0
            
            # Calculate tile row
            row = self.ly - y
            if flip_y:
                row = sprite_height - 1 - row
            
            # For 8x16 sprites, select correct tile
            if sprite_height == 16:
                if row < 8:
                    tile = tile & 0xFE
                else:
                    tile = tile | 0x01
                    row -= 8
            
            # Render sprite pixels
            for px in range(8):
                screen_x = x + px
                if screen_x < 0 or screen_x >= self.SCREEN_WIDTH:
                    continue
                
                col = 7 - px if flip_x else px
                color_idx = self._get_tile_pixel_raw(tile, col, row, vram_bank)
                
                # Color 0 is transparent
                if color_idx == 0:
                    continue
                
                # Check priority
                bg_color = self.line_buffer[screen_x] & 0xFF
                bg_has_priority = self.line_priority[screen_x] != 0
                
                # GBC priority rules
                if self.memory.cgb_mode:
                    if bg_has_priority and bg_color != 0:
                        continue
                    if bg_priority and bg_color != 0:
                        continue
                else:
                    # DMG priority
                    if bg_priority and bg_color != 0:
                        continue
                
                # Set sprite pixel (mark as sprite with bit 15)
                self.line_buffer[screen_x] = 0x8000 | (palette << 8) | color_idx
    
    def _get_tile_pixel(self, tile_idx: int, x: int, y: int, 
                        vram_bank: int, flip_x: bool, flip_y: bool) -> int:
        """Get pixel color index from tile."""
        if flip_x:
            x = 7 - x
        if flip_y:
            y = 7 - y
        
        # Calculate tile address
        if self.tile_data_signed and tile_idx > 127:
            tile_idx = tile_idx - 256
        
        if self.tile_data_signed:
            tile_addr = 0x9000 + tile_idx * 16
        else:
            tile_addr = 0x8000 + tile_idx * 16
        
        return self._get_pixel_from_addr(tile_addr, x, y, vram_bank)
    
    def _get_tile_pixel_raw(self, tile_idx: int, x: int, y: int, vram_bank: int) -> int:
        """Get pixel from tile at 0x8000 base (for sprites)."""
        tile_addr = 0x8000 + tile_idx * 16
        return self._get_pixel_from_addr(tile_addr, x, y, vram_bank)
    
    def _get_pixel_from_addr(self, tile_addr: int, x: int, y: int, vram_bank: int) -> int:
        """Get pixel color index from tile address."""
        row_addr = int(tile_addr) + int(y) * 2
        
        low_byte = int(self._read_vram(row_addr, vram_bank))
        high_byte = int(self._read_vram(row_addr + 1, vram_bank))
        
        bit = 7 - int(x)
        color = ((high_byte >> bit) & 1) << 1 | ((low_byte >> bit) & 1)
        
        return color
    
    def _read_vram(self, addr: int, bank: int) -> int:
        """Read from VRAM with bank selection."""
        offset = int(addr) - 0x8000
        if 0 <= offset < 0x2000:
            return int(self.memory.vram[bank][offset])
        return 0
    
    def _finalize_scanline(self):
        """Convert line buffer to RGB framebuffer."""
        for x in range(self.SCREEN_WIDTH):
            value = int(self.line_buffer[x])
            is_sprite = (value & 0x8000) != 0
            palette = (value >> 8) & 0x07
            color_idx = value & 0x03
            
            if self.memory.cgb_mode:
                # GBC: Use palette RAM
                if is_sprite:
                    color = self._get_cgb_color(self.memory.obj_palette_ram, palette, color_idx)
                else:
                    color = self._get_cgb_color(self.memory.bg_palette_ram, palette, color_idx)
            else:
                # DMG: Use palette registers
                if is_sprite:
                    pal = self.obp1 if palette else self.obp0
                else:
                    pal = self.bgp
                
                mapped_color = (pal >> (color_idx * 2)) & 0x03
                color = self.DMG_COLORS[mapped_color]
            
            self.framebuffer[self.ly, x] = color
    
    def _get_cgb_color(self, palette_ram: np.ndarray, palette: int, color_idx: int) -> np.ndarray:
        """Get RGB color from GBC palette RAM."""
        offset = int(palette) * 8 + int(color_idx) * 2
        lo = int(palette_ram[offset])
        hi = int(palette_ram[offset + 1])
        
        # GBC color format: GGGRRRRR XBBBBBGG
        rgb15 = (hi << 8) | lo
        r = (rgb15 & 0x1F) << 3
        g = ((rgb15 >> 5) & 0x1F) << 3
        b = ((rgb15 >> 10) & 0x1F) << 3
        
        return np.array([r, g, b], dtype=np.uint8)
    
    def get_tilemap_image(self, tilemap_addr: int = None) -> np.ndarray:
        """Generate a visual representation of the tilemap for debugging."""
        if tilemap_addr is None:
            tilemap_addr = self.bg_tilemap
        
        # Tilemap is 32x32 tiles = 256x256 pixels
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        
        for ty in range(32):
            for tx in range(32):
                tile_idx = self._read_vram(tilemap_addr + ty * 32 + tx, 0)
                
                if self.memory.cgb_mode:
                    attr = self._read_vram(tilemap_addr + ty * 32 + tx, 1)
                    palette = attr & 0x07
                    vram_bank = (attr >> 3) & 0x01
                    flip_x = (attr & 0x20) != 0
                    flip_y = (attr & 0x40) != 0
                else:
                    palette = 0
                    vram_bank = 0
                    flip_x = False
                    flip_y = False
                
                # Render tile
                for py in range(8):
                    for px in range(8):
                        color_idx = self._get_tile_pixel(tile_idx, px, py,
                                                         vram_bank, flip_x, flip_y)
                        
                        if self.memory.cgb_mode:
                            color = self._get_cgb_color(self.memory.bg_palette_ram, palette, color_idx)
                        else:
                            mapped = (self.bgp >> (color_idx * 2)) & 0x03
                            color = self.DMG_COLORS[mapped]
                        
                        image[ty * 8 + py, tx * 8 + px] = color
        
        return image
    
    def get_tiles_image(self, bank: int = 0) -> np.ndarray:
        """Generate visual representation of all tiles in VRAM."""
        # 384 tiles total (0x8000-0x97FF), arranged 16x24
        image = np.zeros((192, 128, 3), dtype=np.uint8)
        
        for tile_idx in range(384):
            tx = (tile_idx % 16) * 8
            ty = (tile_idx // 16) * 8
            
            tile_addr = 0x8000 + tile_idx * 16
            
            for py in range(8):
                for px in range(8):
                    color_idx = self._get_pixel_from_addr(tile_addr, px, py, bank)
                    
                    # Use grayscale for tile viewer
                    gray = 255 - (color_idx * 85)
                    image[ty + py, tx + px] = [gray, gray, gray]
        
        return image
    
    def get_sprites_info(self) -> list:
        """Get information about all sprites in OAM."""
        sprites = []
        for i in range(40):
            y = self.memory.oam[i * 4] - 16
            x = self.memory.oam[i * 4 + 1] - 8
            tile = self.memory.oam[i * 4 + 2]
            attr = self.memory.oam[i * 4 + 3]
            
            sprites.append({
                'index': i,
                'x': x,
                'y': y,
                'tile': tile,
                'attr': attr,
                'flip_x': (attr & 0x20) != 0,
                'flip_y': (attr & 0x40) != 0,
                'priority': (attr & 0x80) != 0,
                'palette': attr & 0x07 if self.memory.cgb_mode else (attr >> 4) & 0x01
            })
        
        return sprites


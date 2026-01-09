"""
Screen Reader for GBC Emulator - Pokemon Crystal

This module reads ACTUAL screen content from VRAM - text, menus, dialogs.
The AI uses this to UNDERSTAND what's on screen, not guess.

Pokemon Crystal Character Encoding (from decompilation):
- Text is rendered as tiles in VRAM
- Each character has a specific tile ID
- We decode the tilemap to read what's displayed

VRAM Layout:
- 0x8000-0x97FF: Tile data (384 tiles, 16 bytes each)
- 0x9800-0x9BFF: Background tilemap (32x32 tiles)
- 0x9C00-0x9FFF: Window tilemap (32x32 tiles)

Window layer usually contains menus/dialogs (higher priority).
Background layer shows the game world.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np


# Pokemon Crystal character encoding (from pokecrystal decompilation)
# charmap.asm defines the exact mappings
POKEMON_CRYSTAL_CHARS = {
    # Terminator
    0x50: '@',  # String terminator - marks end of text
    
    # Space and basic punctuation
    0x7F: ' ',
    
    # Numbers (0xF6-0xFF)
    0xF6: '0', 0xF7: '1', 0xF8: '2', 0xF9: '3', 0xFA: '4',
    0xFB: '5', 0xFC: '6', 0xFD: '7', 0xFE: '8', 0xFF: '9',
    
    # Punctuation and symbols
    0xE0: "'",  # Apostrophe
    0xE1: "PK", # PK symbol
    0xE2: "MN", # MN symbol
    0xE3: "-",
    0xE4: "?", 0xE5: "?",
    0xE6: "!",
    0xE7: ".",
    0xE8: "&",
    0xE9: "e",  # lowercase e for "the"
    0xEA: "->", # Arrow
    0xEB: ">",
    0xEC: "<",
    0xED: '"',  # Quote open
    0xEE: '"',  # Quote close
    0xEF: " ",  # Male symbol (show as space)
    0xF0: "$",
    0xF1: "x",  # Multiplication
    0xF2: ".",  # Dot
    0xF3: "/",
    0xF4: ",",
    0xF5: " ",  # Female symbol
    
    # Control codes (usually mark formatting)
    0x00: '',
    0x4E: '\n',  # Newline
    0x4F: '\n',  # Paragraph end
    0x55: '+',   # Plus sign for cont'd text
    0x56: '',    # End line
    0x57: '',    # End
}

# Add uppercase A-Z (0x80-0x99)
for i in range(26):
    POKEMON_CRYSTAL_CHARS[0x80 + i] = chr(ord('A') + i)

# Add lowercase a-z (0xA0-0xB9)
for i in range(26):
    POKEMON_CRYSTAL_CHARS[0xA0 + i] = chr(ord('a') + i)


@dataclass
class TextRegion:
    """Extracted text from a screen region."""
    raw_text: str = ""
    lines: List[str] = field(default_factory=list)
    has_content: bool = False
    region_type: str = ""  # "dialog", "menu", "overworld", "title"
    
    def clean(self) -> str:
        """Get cleaned, readable text."""
        # Remove terminators and excess whitespace
        text = self.raw_text.replace('@', '').replace('\n\n', '\n')
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        return '\n'.join(lines)


@dataclass
class MenuOption:
    """A selectable menu option."""
    text: str
    index: int
    is_selected: bool = False


@dataclass  
class ScreenContent:
    """Complete parsed screen content for AI understanding."""
    
    # Raw extracted text
    window_text: TextRegion = field(default_factory=TextRegion)
    background_text: TextRegion = field(default_factory=TextRegion)
    
    # Interpreted content
    dialog: str = ""                    # Dialog/message text
    question: str = ""                  # Question being asked
    menu_options: List[MenuOption] = field(default_factory=list)
    selected_option: int = -1           # Currently selected menu option
    
    # Screen type detection
    is_title_screen: bool = False
    is_dialog_active: bool = False
    is_menu_active: bool = False
    is_name_entry: bool = False
    is_battle: bool = False
    
    # What the AI should know
    screen_summary: str = ""
    visible_choices: List[str] = field(default_factory=list)
    suggested_action: str = ""
    
    def to_llm_context(self) -> str:
        """Format for LLM - this is what the AI 'sees'."""
        lines = []
        
        lines.append("=== WHAT I SEE ON SCREEN ===")
        
        if self.is_title_screen:
            lines.append("SCREEN TYPE: Title Screen")
            lines.append("ACTION NEEDED: Press START to begin")
        
        elif self.is_name_entry:
            lines.append("SCREEN TYPE: Name Entry")
            lines.append("There is a character grid to type a name.")
            lines.append("Navigate with arrows, select characters with A.")
            lines.append("Select END when done, or press START for preset names.")
        
        elif self.is_battle:
            lines.append("SCREEN TYPE: Battle")
            if self.menu_options:
                lines.append("Battle menu options:")
                for opt in self.menu_options:
                    marker = ">" if opt.is_selected else " "
                    lines.append(f"  {marker} {opt.text}")
        
        elif self.is_dialog_active:
            lines.append("SCREEN TYPE: Dialog Box")
            if self.dialog:
                lines.append(f"TEXT: {self.dialog}")
            if self.question:
                lines.append(f"QUESTION: {self.question}")
            if self.menu_options:
                lines.append("CHOICES:")
                for opt in self.menu_options:
                    marker = ">" if opt.is_selected else " "
                    lines.append(f"  {marker} {opt.text}")
            if not self.menu_options:
                lines.append("ACTION: Press A to continue reading")
        
        elif self.is_menu_active:
            lines.append("SCREEN TYPE: Menu/Selection")
            if self.menu_options:
                lines.append("OPTIONS:")
                for opt in self.menu_options:
                    marker = ">" if opt.is_selected else " "
                    lines.append(f"  {marker} {opt.text}")
            if self.dialog:
                lines.append(f"CONTEXT: {self.dialog}")
        
        else:
            lines.append("SCREEN TYPE: Overworld/Gameplay")
            if self.background_text.has_content:
                text = self.background_text.clean()
                if text:
                    lines.append(f"VISIBLE TEXT: {text[:100]}")
        
        if self.screen_summary:
            lines.append(f"SUMMARY: {self.screen_summary}")
        
        return '\n'.join(lines)


class ScreenReader:
    """
    Reads actual screen content from GBC VRAM.
    
    This lets the AI UNDERSTAND what's displayed - text, menus, choices.
    Not guessing, but actually reading the game's output.
    """
    
    def __init__(self, memory):
        self.memory = memory
        
        # Screen dimensions
        self.SCREEN_WIDTH = 20   # Visible tiles wide
        self.SCREEN_HEIGHT = 18  # Visible tiles tall
        self.TILEMAP_WIDTH = 32  # Full tilemap width
        
        # VRAM addresses
        self.BG_MAP_ADDR = 0x9800   # Background tilemap
        self.WIN_MAP_ADDR = 0x9C00  # Window tilemap
        
        # Scroll registers for proper reading
        self.SCX = 0xFF43  # Scroll X
        self.SCY = 0xFF42  # Scroll Y
        self.WX = 0xFF4B   # Window X (+7)
        self.WY = 0xFF4A   # Window Y
        
        # Last read cache to avoid redundant reads
        self._last_content = None
        self._cache_frame = 0
    
    def _read_vram_byte(self, addr: int) -> int:
        """Read a byte from VRAM."""
        if 0x8000 <= addr < 0xA000:
            offset = addr - 0x8000
            bank = self.memory.vram_bank
            if offset < len(self.memory.vram[bank]):
                return int(self.memory.vram[bank][offset])
        return 0
    
    def _read_io(self, addr: int) -> int:
        """Read an I/O register."""
        if 0xFF00 <= addr < 0xFF80:
            offset = addr - 0xFF00
            if offset < len(self.memory.io):
                return int(self.memory.io[offset])
        return 0
    
    def _tile_to_char(self, tile_id: int) -> str:
        """Convert tile ID to character using Pokemon Crystal encoding."""
        return POKEMON_CRYSTAL_CHARS.get(tile_id, '')
    
    def _read_tilemap_line(self, base_addr: int, y: int) -> List[int]:
        """Read a single line of tiles from a tilemap."""
        tiles = []
        for x in range(self.SCREEN_WIDTH):
            addr = base_addr + (y * self.TILEMAP_WIDTH) + x
            tiles.append(self._read_vram_byte(addr))
        return tiles
    
    def _tiles_to_text(self, tile_rows: List[List[int]]) -> str:
        """Convert tile grid to text string."""
        lines = []
        for row in tile_rows:
            line = ''.join(self._tile_to_char(t) for t in row)
            # Clean up line - trim and skip empty
            line = line.rstrip()
            if line:
                lines.append(line)
        return '\n'.join(lines)
    
    def _read_window_layer(self) -> TextRegion:
        """Read the window layer - usually menus and dialogs."""
        region = TextRegion(region_type="window")
        
        # Read window position
        wy = self._read_io(self.WY)
        wx = self._read_io(self.WX)
        
        # Window is active if WY is on screen
        if wy >= 144:
            return region
        
        # Read visible window tiles
        tile_rows = []
        start_y = 0  # Window Y position in tilemap
        
        for y in range(self.SCREEN_HEIGHT):
            row = self._read_tilemap_line(self.WIN_MAP_ADDR, start_y + y)
            tile_rows.append(row)
        
        region.raw_text = self._tiles_to_text(tile_rows)
        region.lines = region.raw_text.split('\n')
        region.has_content = bool(region.raw_text.strip())
        
        return region
    
    def _read_background_layer(self) -> TextRegion:
        """Read the background layer."""
        region = TextRegion(region_type="background")
        
        # Get scroll position
        scx = self._read_io(self.SCX)
        scy = self._read_io(self.SCY)
        
        # Read visible background tiles (accounting for scroll)
        tile_rows = []
        start_y = scy // 8  # Convert pixel scroll to tile position
        
        for y in range(self.SCREEN_HEIGHT):
            row = self._read_tilemap_line(self.BG_MAP_ADDR, (start_y + y) % 32)
            tile_rows.append(row)
        
        region.raw_text = self._tiles_to_text(tile_rows)
        region.lines = region.raw_text.split('\n')
        region.has_content = bool(region.raw_text.strip())
        
        return region
    
    def _detect_menu_options(self, text: str) -> Tuple[List[MenuOption], int]:
        """Extract menu options from text."""
        options = []
        selected = -1
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check for selection indicator
            is_sel = False
            if line.startswith('>') or line.startswith('-'):
                is_sel = True
                selected = len(options)
                line = line[1:].strip()
            
            # Common menu keywords
            menu_keywords = [
                "YES", "NO", "BOY", "GIRL", "NEW GAME", "CONTINUE",
                "OPTIONS", "FIGHT", "ITEM", "POKEMON", "PKMN", "RUN",
                "MORNING", "DAY", "NIGHT", "END", "DEL",
            ]
            
            # Check if this looks like a menu option
            upper = line.upper()
            if any(kw in upper for kw in menu_keywords):
                options.append(MenuOption(text=line, index=len(options), is_selected=is_sel))
            elif len(line) < 15 and not any(c in line for c in '.?!'):
                # Short text without punctuation - likely an option
                options.append(MenuOption(text=line, index=len(options), is_selected=is_sel))
        
        return options, selected
    
    def _detect_question(self, text: str) -> str:
        """Find question in text."""
        # Direct question mark
        if '?' in text:
            for sentence in text.split('\n'):
                if '?' in sentence:
                    return sentence.strip()
        
        # Question keywords
        q_keywords = ["What", "Which", "Who", "Are you", "Is it", "Do you"]
        lower = text.lower()
        for kw in q_keywords:
            if kw.lower() in lower:
                # Extract context around keyword
                idx = lower.find(kw.lower())
                start = max(0, text.rfind('\n', 0, idx) + 1)
                end = text.find('\n', idx)
                if end == -1:
                    end = len(text)
                return text[start:end].strip()
        
        return ""
    
    def _analyze_content(self, window: TextRegion, bg: TextRegion) -> ScreenContent:
        """Analyze extracted text to understand the screen."""
        content = ScreenContent(window_text=window, background_text=bg)
        
        # Combine texts for analysis
        all_text = window.raw_text + '\n' + bg.raw_text
        all_text_upper = all_text.upper()
        
        # === Detect screen type ===
        
        # Title screen
        if "NEW GAME" in all_text_upper or "CONTINUE" in all_text_upper:
            content.is_title_screen = True
            content.screen_summary = "Title screen - start or continue game"
            content.menu_options, content.selected_option = self._detect_menu_options(all_text)
        
        # Name entry
        elif "YOUR NAME" in all_text_upper or "NAME?" in all_text_upper:
            content.is_name_entry = True
            content.screen_summary = "Name entry screen"
        
        # Check for character grid (name entry without explicit text)
        elif self._looks_like_name_grid(window.raw_text):
            content.is_name_entry = True
            content.screen_summary = "Character selection grid"
        
        # Battle
        elif "FIGHT" in all_text_upper and ("PKMN" in all_text_upper or "POKEMON" in all_text_upper):
            content.is_battle = True
            content.screen_summary = "Pokemon battle"
            content.menu_options, content.selected_option = self._detect_menu_options(all_text)
        
        # Dialog with choices (YES/NO, BOY/GIRL, etc)
        elif "YES" in all_text_upper and "NO" in all_text_upper:
            content.is_dialog_active = True
            content.is_menu_active = True
            content.menu_options, content.selected_option = self._detect_menu_options(all_text)
            content.question = self._detect_question(all_text)
            content.dialog = window.clean() if window.has_content else bg.clean()
            content.screen_summary = "Question with YES/NO choice"
        
        elif "BOY" in all_text_upper and "GIRL" in all_text_upper:
            content.is_dialog_active = True
            content.is_menu_active = True
            content.menu_options, content.selected_option = self._detect_menu_options(all_text)
            content.screen_summary = "Gender selection"
        
        # Time selection
        elif "MORNING" in all_text_upper or "DAY" in all_text_upper or "NIGHT" in all_text_upper:
            content.is_dialog_active = True
            content.is_menu_active = True  
            content.menu_options, content.selected_option = self._detect_menu_options(all_text)
            content.question = self._detect_question(all_text)
            content.screen_summary = "Time selection"
        
        # General dialog
        elif window.has_content:
            content.is_dialog_active = True
            content.dialog = window.clean()
            content.question = self._detect_question(all_text)
            
            # Check if this dialog has menu options
            options, sel = self._detect_menu_options(all_text)
            if options:
                content.is_menu_active = True
                content.menu_options = options
                content.selected_option = sel
                content.screen_summary = f"Dialog with {len(options)} choices"
            else:
                content.screen_summary = "Dialog text"
        
        # No clear type - probably overworld
        else:
            content.screen_summary = "Game world / exploration"
        
        return content
    
    def _looks_like_name_grid(self, text: str) -> bool:
        """Check if text looks like the name entry character grid."""
        # Name entry has rows of single letters
        upper = text.upper()
        
        # Check for letter sequences
        if 'ABCDE' in upper or 'KLMNO' in upper or 'UVWXY' in upper:
            return True
        
        # Check for END keyword (name entry confirmation)
        if 'END' in upper and 'DEL' in upper:
            return True
        
        return False
    
    def read_screen(self) -> ScreenContent:
        """Main method - read and interpret the entire screen."""
        # Read both layers
        window = self._read_window_layer()
        bg = self._read_background_layer()
        
        # Analyze and interpret
        content = self._analyze_content(window, bg)
        
        return content
    
    def get_screen_for_llm(self) -> str:
        """Get formatted screen content for LLM prompt."""
        content = self.read_screen()
        return content.to_llm_context()
    
    def get_raw_text(self) -> str:
        """Get raw extracted text for debugging."""
        window = self._read_window_layer()
        bg = self._read_background_layer()
        
        lines = ["=== WINDOW LAYER ==="]
        lines.append(window.raw_text if window.has_content else "(empty)")
        lines.append("")
        lines.append("=== BACKGROUND LAYER ===")
        lines.append(bg.raw_text if bg.has_content else "(empty)")
        
        return '\n'.join(lines)

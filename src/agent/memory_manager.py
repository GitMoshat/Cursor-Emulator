"""
Memory Manager for GBC Emulator
Provides structured access to game memory for AI agents.

Pokemon Red/Blue/Yellow and similar games have well-documented memory layouts.
This manager extracts meaningful game state from raw memory.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum, auto
import json


class GameType(Enum):
    """Supported game types with known memory layouts."""
    UNKNOWN = auto()
    POKEMON_RED_BLUE = auto()
    POKEMON_YELLOW = auto()
    POKEMON_GOLD_SILVER = auto()
    POKEMON_CRYSTAL = auto()
    GENERIC_GBC = auto()


@dataclass
class Position:
    """Player/entity position."""
    x: int = 0
    y: int = 0
    map_id: int = 0
    map_name: str = ""
    facing: str = "down"  # up, down, left, right


@dataclass  
class PokemonData:
    """Data about a Pokemon in party/box."""
    species_id: int = 0
    species_name: str = ""
    level: int = 0
    current_hp: int = 0
    max_hp: int = 0
    status: int = 0  # 0=healthy, various bits for status conditions
    exp: int = 0
    moves: List[int] = field(default_factory=list)
    move_names: List[str] = field(default_factory=list)


@dataclass
class BattleState:
    """Current battle state."""
    in_battle: bool = False
    is_wild: bool = False
    is_trainer: bool = False
    enemy_species: int = 0
    enemy_name: str = ""
    enemy_level: int = 0
    enemy_hp: int = 0
    enemy_max_hp: int = 0
    player_pokemon_idx: int = 0
    turn_count: int = 0
    menu_state: int = 0  # 0=main, 1=fight, 2=pokemon, 3=item, 4=run


@dataclass
class MenuState:
    """Current menu/dialog state."""
    in_menu: bool = False
    menu_type: str = ""  # "main", "pokemon", "items", "save", "dialog", etc.
    cursor_position: int = 0
    text_active: bool = False
    options_count: int = 0


@dataclass
class GameState:
    """Complete parsed game state from memory."""
    # Core state
    game_type: GameType = GameType.UNKNOWN
    frame_count: int = 0
    
    # Player
    player_name: str = ""
    player_position: Position = field(default_factory=Position)
    money: int = 0
    badges: int = 0
    play_time_hours: int = 0
    play_time_minutes: int = 0
    
    # Party
    party_count: int = 0
    party: List[PokemonData] = field(default_factory=list)
    
    # Battle
    battle: BattleState = field(default_factory=BattleState)
    
    # Menu/UI
    menu: MenuState = field(default_factory=MenuState)
    
    # Items (simplified)
    item_count: int = 0
    has_pokeballs: bool = False
    has_potions: bool = False
    
    # Flags
    has_starter: bool = False
    has_pokedex: bool = False
    game_started: bool = False
    
    # Raw values for debugging
    raw_values: Dict[str, int] = field(default_factory=dict)
    
    def to_prompt(self) -> str:
        """Convert state to text for LLM prompt."""
        lines = []
        
        # Location
        pos = self.player_position
        lines.append(f"Location: {pos.map_name or f'Map {pos.map_id}'} at ({pos.x}, {pos.y}), facing {pos.facing}")
        
        # Battle state
        if self.battle.in_battle:
            b = self.battle
            battle_type = "wild" if b.is_wild else "trainer"
            lines.append(f"IN BATTLE ({battle_type}): vs {b.enemy_name} Lv{b.enemy_level}")
            lines.append(f"  Enemy HP: {b.enemy_hp}/{b.enemy_max_hp}")
            if self.party:
                p = self.party[b.player_pokemon_idx] if b.player_pokemon_idx < len(self.party) else self.party[0]
                lines.append(f"  Your {p.species_name} Lv{p.level}: {p.current_hp}/{p.max_hp} HP")
            menu_names = ["MAIN", "FIGHT", "POKEMON", "ITEM", "RUN"]
            lines.append(f"  Menu: {menu_names[b.menu_state] if b.menu_state < len(menu_names) else 'UNKNOWN'}")
        
        # Menu state
        elif self.menu.in_menu:
            lines.append(f"MENU: {self.menu.menu_type}")
            lines.append(f"  Cursor at option {self.menu.cursor_position}")
            if self.menu.text_active:
                lines.append("  Text/dialog is showing (press A to continue)")
        
        # Party summary
        if self.party:
            lines.append(f"Party ({self.party_count} Pokemon):")
            for i, p in enumerate(self.party[:6]):
                hp_pct = (p.current_hp * 100 // p.max_hp) if p.max_hp > 0 else 0
                status = "OK" if p.status == 0 else f"STATUS:{p.status}"
                lines.append(f"  {i+1}. {p.species_name} Lv{p.level} - {hp_pct}% HP [{status}]")
        
        # Progress
        lines.append(f"Badges: {self.badges} | Money: ${self.money}")
        
        # Flags
        flags = []
        if not self.game_started:
            flags.append("GAME NOT STARTED")
        if not self.has_starter:
            flags.append("NO STARTER YET")
        if self.has_pokedex:
            flags.append("HAS POKEDEX")
        if flags:
            lines.append(f"Status: {', '.join(flags)}")
        
        return '\n'.join(lines)
    
    def get_recommended_action(self) -> Tuple[List[str], str]:
        """Get recommended buttons based on current state."""
        # In battle
        if self.battle.in_battle:
            if self.battle.menu_state == 0:  # Main battle menu
                return ["A"], "Select FIGHT to attack"
            elif self.battle.menu_state == 1:  # Fight menu
                return ["A"], "Select first move"
            else:
                return ["A"], "Confirm selection"
        
        # Text/dialog showing
        if self.menu.text_active:
            return ["A"], "Advance dialog"
        
        # In menu
        if self.menu.in_menu:
            if self.menu.menu_type == "pokemon_select":
                return ["A"], "Select Pokemon"
            return ["B"], "Exit menu or go back"
        
        # Not started
        if not self.game_started:
            return ["START"], "Start the game"
        
        # No starter yet - need to explore
        if not self.has_starter:
            return ["UP", "A"], "Look for professor/Pokeballs"
        
        # Default - explore
        return ["UP"], "Continue moving/exploring"


# =============================================================================
# Memory Address Maps for Different Games
# =============================================================================

# Pokemon Red/Blue (US) memory addresses
POKEMON_RB_ADDRESSES = {
    # Game state
    'game_state': 0xD057,  # 0=title, various values for in-game
    'in_battle': 0xD057,
    
    # Player info
    'player_name': 0xD158,  # 11 bytes
    'player_x': 0xD362,
    'player_y': 0xD361,
    'player_map': 0xD35E,
    'player_facing': 0xC109,  # 0=down, 4=up, 8=left, 0xC=right
    'money': 0xD347,  # 3 bytes BCD
    'badges': 0xD356,
    
    # Party
    'party_count': 0xD163,
    'party_species': 0xD164,  # 6 bytes
    'party_data': 0xD16B,  # 44 bytes per Pokemon, 6 Pokemon
    
    # Battle
    'battle_type': 0xD057,
    'enemy_species': 0xCFE5,
    'enemy_level': 0xCFF3,
    'enemy_hp': 0xCFE6,  # 2 bytes
    'enemy_max_hp': 0xCFF4,  # 2 bytes
    'battle_menu': 0xCC2B,
    
    # Menu/UI
    'menu_open': 0xCF94,
    'cursor_pos': 0xCC26,
    'text_progress': 0xC4F2,
    
    # Flags
    'has_pokedex': 0xD5B3,
    'starter_flag': 0xD72E,
    
    # Audio (for detecting events)
    'current_music': 0xC0EE,
}

# Pokemon Yellow addresses (mostly same as RB with some differences)
POKEMON_YELLOW_ADDRESSES = {
    **POKEMON_RB_ADDRESSES,
    # Yellow-specific overrides
    'pikachu_happiness': 0xD46F,
}

# Pokemon Gold/Silver/Crystal addresses (from pokecrystal disassembly)
# These are for Crystal specifically but work for G/S variants too
POKEMON_GSC_ADDRESSES = {
    # Game state
    'game_state': 0xD4B3,      # wGameState - 0=title, other=in-game
    'in_overworld': 0xD4B4,    # Overworld state
    
    # Player info  
    'player_name': 0xD47D,     # wPlayerName - 11 bytes
    'player_id': 0xD47B,       # wPlayerID - 2 bytes
    'player_x': 0xD4E6,        # wXCoord
    'player_y': 0xD4E5,        # wYCoord  
    'player_map_group': 0xDCB5, # wMapGroup
    'player_map': 0xDCB6,      # wMapNumber
    'player_facing': 0xD4DE,   # wPlayerDirection - 0=down, 1=up, 2=left, 3=right
    'player_moving': 0xD4E1,   # wPlayerMoving
    
    # Money (3 bytes BCD)
    'money': 0xD573,           # wMoney
    
    # Badges
    'badges': 0xD857,          # wJohtoBadges (8 bits for 8 badges)
    'kanto_badges': 0xD858,    # wKantoBadges
    
    # Party
    'party_count': 0xDCD7,     # wPartyCount
    'party_species': 0xDCD8,   # wPartySpecies - 6 bytes (species IDs)
    'party_data': 0xDCDF,      # wPartyMon1 - 48 bytes per Pokemon, 6 Pokemon
    'party_nicknames': 0xDE41, # wPartyMonNicknames
    
    # Battle
    'in_battle': 0xD22D,       # wBattleMode - 0=no, 1=wild, 2=trainer
    'battle_type': 0xD230,     # wBattleType
    'enemy_species': 0xD206,   # wEnemyMonSpecies  
    'enemy_level': 0xD213,     # wEnemyMonLevel
    'enemy_hp': 0xD214,        # wEnemyMonHP - 2 bytes
    'enemy_max_hp': 0xD216,    # wEnemyMonMaxHP - 2 bytes
    'enemy_status': 0xD218,    # wEnemyMonStatus
    'battle_menu': 0xD0D4,     # wBattleMenuCursorPosition
    'battle_turn': 0xD264,     # Turn counter
    'player_mon_hp': 0xD218,   # wBattleMonHP - 2 bytes
    
    # Menu/UI
    'menu_open': 0xCF63,       # wMenuCursorY
    'cursor_pos': 0xCF64,      # wMenuCursorX
    'text_delay': 0xCFC9,      # wTextDelayFrames
    'joy_pressed': 0xCFA5,     # wJoyPressed - button state
    
    # Flags
    'has_pokedex': 0xD84C,     # wPokedexCaught flags start
    'starter_flag': 0xD84F,    # Event flag for starter
    'game_time_hours': 0xD4C4, # wGameTimeHours
    'game_time_mins': 0xD4C6,  # wGameTimeMinutes
    
    # Items
    'num_items': 0xD892,       # wNumItems
    'items': 0xD893,           # wItems
    'num_balls': 0xD8D7,       # wNumBalls (PC)
    
    # Audio
    'current_music': 0xC2A4,   # wCurMusic
    
    # Map/warp related
    'last_spawn_map': 0xD6FA,  # wLastSpawnMapGroup/Number
}

# Pokemon Crystal specific (some addresses differ slightly)
POKEMON_CRYSTAL_ADDRESSES = {
    **POKEMON_GSC_ADDRESSES,
    # Crystal-specific overrides if needed
    'mobile_adapter': 0xC2D5,  # Crystal had mobile features
}

# Pokemon species names (Gen 2 / Crystal order - index = national dex - 1)
# Gen 2 uses national dex order unlike Gen 1's internal order
POKEMON_NAMES_GEN2 = {
    0: "???",
    1: "BULBASAUR", 2: "IVYSAUR", 3: "VENUSAUR", 4: "CHARMANDER", 5: "CHARMELEON",
    6: "CHARIZARD", 7: "SQUIRTLE", 8: "WARTORTLE", 9: "BLASTOISE", 10: "CATERPIE",
    11: "METAPOD", 12: "BUTTERFREE", 13: "WEEDLE", 14: "KAKUNA", 15: "BEEDRILL",
    16: "PIDGEY", 17: "PIDGEOTTO", 18: "PIDGEOT", 19: "RATTATA", 20: "RATICATE",
    21: "SPEAROW", 22: "FEAROW", 23: "EKANS", 24: "ARBOK", 25: "PIKACHU",
    26: "RAICHU", 27: "SANDSHREW", 28: "SANDSLASH", 29: "NIDORAN♀", 30: "NIDORINA",
    31: "NIDOQUEEN", 32: "NIDORAN♂", 33: "NIDORINO", 34: "NIDOKING", 35: "CLEFAIRY",
    36: "CLEFABLE", 37: "VULPIX", 38: "NINETALES", 39: "JIGGLYPUFF", 40: "WIGGLYTUFF",
    41: "ZUBAT", 42: "GOLBAT", 43: "ODDISH", 44: "GLOOM", 45: "VILEPLUME",
    46: "PARAS", 47: "PARASECT", 48: "VENONAT", 49: "VENOMOTH", 50: "DIGLETT",
    51: "DUGTRIO", 52: "MEOWTH", 53: "PERSIAN", 54: "PSYDUCK", 55: "GOLDUCK",
    56: "MANKEY", 57: "PRIMEAPE", 58: "GROWLITHE", 59: "ARCANINE", 60: "POLIWAG",
    61: "POLIWHIRL", 62: "POLIWRATH", 63: "ABRA", 64: "KADABRA", 65: "ALAKAZAM",
    66: "MACHOP", 67: "MACHOKE", 68: "MACHAMP", 69: "BELLSPROUT", 70: "WEEPINBELL",
    71: "VICTREEBEL", 72: "TENTACOOL", 73: "TENTACRUEL", 74: "GEODUDE", 75: "GRAVELER",
    76: "GOLEM", 77: "PONYTA", 78: "RAPIDASH", 79: "SLOWPOKE", 80: "SLOWBRO",
    81: "MAGNEMITE", 82: "MAGNETON", 83: "FARFETCHD", 84: "DODUO", 85: "DODRIO",
    86: "SEEL", 87: "DEWGONG", 88: "GRIMER", 89: "MUK", 90: "SHELLDER",
    91: "CLOYSTER", 92: "GASTLY", 93: "HAUNTER", 94: "GENGAR", 95: "ONIX",
    96: "DROWZEE", 97: "HYPNO", 98: "KRABBY", 99: "KINGLER", 100: "VOLTORB",
    101: "ELECTRODE", 102: "EXEGGCUTE", 103: "EXEGGUTOR", 104: "CUBONE", 105: "MAROWAK",
    106: "HITMONLEE", 107: "HITMONCHAN", 108: "LICKITUNG", 109: "KOFFING", 110: "WEEZING",
    111: "RHYHORN", 112: "RHYDON", 113: "CHANSEY", 114: "TANGELA", 115: "KANGASKHAN",
    116: "HORSEA", 117: "SEADRA", 118: "GOLDEEN", 119: "SEAKING", 120: "STARYU",
    121: "STARMIE", 122: "MR.MIME", 123: "SCYTHER", 124: "JYNX", 125: "ELECTABUZZ",
    126: "MAGMAR", 127: "PINSIR", 128: "TAUROS", 129: "MAGIKARP", 130: "GYARADOS",
    131: "LAPRAS", 132: "DITTO", 133: "EEVEE", 134: "VAPOREON", 135: "JOLTEON",
    136: "FLAREON", 137: "PORYGON", 138: "OMANYTE", 139: "OMASTAR", 140: "KABUTO",
    141: "KABUTOPS", 142: "AERODACTYL", 143: "SNORLAX", 144: "ARTICUNO", 145: "ZAPDOS",
    146: "MOLTRES", 147: "DRATINI", 148: "DRAGONAIR", 149: "DRAGONITE", 150: "MEWTWO",
    151: "MEW",
    # Gen 2 Pokemon (152-251)
    152: "CHIKORITA", 153: "BAYLEEF", 154: "MEGANIUM", 155: "CYNDAQUIL", 156: "QUILAVA",
    157: "TYPHLOSION", 158: "TOTODILE", 159: "CROCONAW", 160: "FERALIGATR",
    161: "SENTRET", 162: "FURRET", 163: "HOOTHOOT", 164: "NOCTOWL", 165: "LEDYBA",
    166: "LEDIAN", 167: "SPINARAK", 168: "ARIADOS", 169: "CROBAT", 170: "CHINCHOU",
    171: "LANTURN", 172: "PICHU", 173: "CLEFFA", 174: "IGGLYBUFF", 175: "TOGEPI",
    176: "TOGETIC", 177: "NATU", 178: "XATU", 179: "MAREEP", 180: "FLAAFFY",
    181: "AMPHAROS", 182: "BELLOSSOM", 183: "MARILL", 184: "AZUMARILL", 185: "SUDOWOODO",
    186: "POLITOED", 187: "HOPPIP", 188: "SKIPLOOM", 189: "JUMPLUFF", 190: "AIPOM",
    191: "SUNKERN", 192: "SUNFLORA", 193: "YANMA", 194: "WOOPER", 195: "QUAGSIRE",
    196: "ESPEON", 197: "UMBREON", 198: "MURKROW", 199: "SLOWKING", 200: "MISDREAVUS",
    201: "UNOWN", 202: "WOBBUFFET", 203: "GIRAFARIG", 204: "PINECO", 205: "FORRETRESS",
    206: "DUNSPARCE", 207: "GLIGAR", 208: "STEELIX", 209: "SNUBBULL", 210: "GRANBULL",
    211: "QWILFISH", 212: "SCIZOR", 213: "SHUCKLE", 214: "HERACROSS", 215: "SNEASEL",
    216: "TEDDIURSA", 217: "URSARING", 218: "SLUGMA", 219: "MAGCARGO", 220: "SWINUB",
    221: "PILOSWINE", 222: "CORSOLA", 223: "REMORAID", 224: "OCTILLERY", 225: "DELIBIRD",
    226: "MANTINE", 227: "SKARMORY", 228: "HOUNDOUR", 229: "HOUNDOOM", 230: "KINGDRA",
    231: "PHANPY", 232: "DONPHAN", 233: "PORYGON2", 234: "STANTLER", 235: "SMEARGLE",
    236: "TYROGUE", 237: "HITMONTOP", 238: "SMOOCHUM", 239: "ELEKID", 240: "MAGBY",
    241: "MILTANK", 242: "BLISSEY", 243: "RAIKOU", 244: "ENTEI", 245: "SUICUNE",
    246: "LARVITAR", 247: "PUPITAR", 248: "TYRANITAR", 249: "LUGIA", 250: "HO-OH",
    251: "CELEBI",
}

# Gen 1 internal index (different from national dex)
POKEMON_NAMES_GEN1 = {
    0: "???", 1: "RHYDON", 2: "KANGASKHAN", 3: "NIDORAN♂", 4: "CLEFAIRY",
    5: "SPEAROW", 6: "VOLTORB", 7: "NIDOKING", 8: "SLOWBRO", 9: "IVYSAUR",
    # ... simplified, Gen 2 uses national dex order
}

# Map names for Gen 2 (Johto)
# Format: (group, number) for GSC's two-byte map system
MAP_NAMES_GSC = {
    # New Bark Town area (Group 1)
    (1, 1): "New Bark Town",
    (1, 2): "Player's House 1F",
    (1, 3): "Player's House 2F", 
    (1, 4): "Elm's Lab",
    (1, 5): "Elm's House",
    # Cherrygrove area (Group 2)
    (2, 1): "Cherrygrove City",
    (2, 2): "Pokemon Center",
    (2, 3): "Mart",
    (2, 4): "Guide Gent's House",
    # Violet City area (Group 3)
    (3, 1): "Violet City",
    (3, 2): "Sprout Tower 1F",
    (3, 3): "Sprout Tower 2F",
    (3, 4): "Sprout Tower 3F",
    (3, 5): "Violet Gym",
    # Routes
    (24, 1): "Route 29",
    (24, 2): "Route 30",
    (24, 3): "Route 31",
    (24, 4): "Route 32",
    (24, 5): "Route 33",
    (24, 6): "Route 34",
    (24, 7): "Route 35",
    (24, 8): "Route 36",
    (24, 9): "Route 37",
    (24, 10): "Route 38",
    (24, 11): "Route 39",
    (24, 12): "Route 40",
    (24, 13): "Route 41",
}

# Simple map lookup by single number (for compatibility)
MAP_NAMES_GEN1 = {
    0: "Pallet Town", 1: "Viridian City", 2: "Pewter City", 3: "Cerulean City",
    4: "Lavender Town", 5: "Vermilion City", 6: "Celadon City", 7: "Fuchsia City",
    8: "Cinnabar Island", 9: "Indigo Plateau", 10: "Saffron City",
}


class MemoryManager:
    """
    Manages reading and interpreting game memory.
    Provides structured game state to AI agents.
    """
    
    def __init__(self, emulator):
        self.emulator = emulator
        self.memory = emulator.memory
        
        # Detect game type
        self.game_type = GameType.UNKNOWN
        self.addresses = {}
        self.pokemon_names = POKEMON_NAMES_GEN2  # Default to Gen 2 for Dokemon
        self.map_names = MAP_NAMES_GEN1
        self.map_names_gsc = MAP_NAMES_GSC
        
        # Cache for expensive reads
        self._state_cache: Optional[GameState] = None
        self._cache_frame = -1
        
        # History for change detection
        self.position_history: List[Position] = []
        self.max_history = 100
        
        # Custom watchers
        self.watchers: Dict[str, Tuple[int, str]] = {}  # name -> (address, type)
        
        # Debug: log all memory reads
        self.debug_reads = False
    
    def detect_game(self):
        """Detect game type from ROM header."""
        try:
            # Read ROM title (0x134-0x143)
            title_bytes = bytes(self.memory.rom[0x134:0x144])
            title = title_bytes.decode('ascii', errors='ignore').strip('\x00')
            
            title_lower = title.lower()
            
            print(f"[MemoryManager] ROM Title: '{title}'")
            
            # Check for Crystal variants first (Dokemon is Crystal-based)
            if 'dokemon' in title_lower or 'crystal' in title_lower:
                self.game_type = GameType.POKEMON_CRYSTAL
                self.addresses = POKEMON_CRYSTAL_ADDRESSES
                self.pokemon_names = POKEMON_NAMES_GEN2
                print(f"[MemoryManager] Detected as Pokemon Crystal variant")
            elif 'pokemon red' in title_lower or 'pokemon blue' in title_lower:
                self.game_type = GameType.POKEMON_RED_BLUE
                self.addresses = POKEMON_RB_ADDRESSES
                self.pokemon_names = POKEMON_NAMES_GEN1
            elif 'pokemon yellow' in title_lower or 'pokemon pikachu' in title_lower:
                self.game_type = GameType.POKEMON_YELLOW
                self.addresses = POKEMON_YELLOW_ADDRESSES
                self.pokemon_names = POKEMON_NAMES_GEN1
            elif 'pokemon gold' in title_lower or 'pokemon silver' in title_lower:
                self.game_type = GameType.POKEMON_GOLD_SILVER
                self.addresses = POKEMON_GSC_ADDRESSES
            elif 'pokemon crystal' in title_lower:
                self.game_type = GameType.POKEMON_CRYSTAL
                self.addresses = POKEMON_GSC_ADDRESSES
            else:
                # Generic - use RB addresses as fallback
                self.game_type = GameType.GENERIC_GBC
                self.addresses = POKEMON_RB_ADDRESSES
            
            print(f"[MemoryManager] Detected game: {title} -> {self.game_type.name}")
            
        except Exception as e:
            print(f"[MemoryManager] Game detection failed: {e}")
            self.game_type = GameType.GENERIC_GBC
            self.addresses = POKEMON_RB_ADDRESSES
    
    def read_byte(self, address: int) -> int:
        """Read a single byte from memory."""
        return int(self.memory.read(address))
    
    def read_word(self, address: int) -> int:
        """Read a 16-bit word (little endian)."""
        lo = self.read_byte(address)
        hi = self.read_byte(address + 1)
        return lo | (hi << 8)
    
    def read_bytes(self, address: int, length: int) -> bytes:
        """Read multiple bytes."""
        return bytes(self.memory.read(address + i) for i in range(length))
    
    def read_bcd(self, address: int, length: int) -> int:
        """Read BCD-encoded number (used for money)."""
        result = 0
        for i in range(length):
            byte = self.read_byte(address + i)
            result = result * 100 + ((byte >> 4) * 10) + (byte & 0x0F)
        return result
    
    def read_string(self, address: int, max_length: int = 11) -> str:
        """Read a Pokemon-style string (0x50 terminated)."""
        chars = []
        for i in range(max_length):
            byte = self.read_byte(address + i)
            if byte == 0x50:  # Terminator
                break
            # Pokemon charset conversion (simplified)
            if 0x80 <= byte <= 0x99:  # A-Z
                chars.append(chr(ord('A') + byte - 0x80))
            elif 0xA0 <= byte <= 0xB9:  # a-z
                chars.append(chr(ord('a') + byte - 0xA0))
            elif 0xF6 <= byte <= 0xFF:  # 0-9
                chars.append(chr(ord('0') + byte - 0xF6))
            elif byte == 0x7F:
                chars.append(' ')
            else:
                chars.append('?')
        return ''.join(chars)
    
    def get_state(self, force_refresh: bool = False) -> GameState:
        """Get current game state, with caching."""
        current_frame = self.emulator.total_frames
        
        if not force_refresh and self._state_cache and self._cache_frame == current_frame:
            return self._state_cache
        
        state = self._read_game_state()
        self._state_cache = state
        self._cache_frame = current_frame
        
        # Track position history
        if state.player_position.x != 0 or state.player_position.y != 0:
            self.position_history.append(state.player_position)
            if len(self.position_history) > self.max_history:
                self.position_history.pop(0)
        
        return state
    
    def _read_game_state(self) -> GameState:
        """Read and parse complete game state from memory."""
        state = GameState(game_type=self.game_type, frame_count=self.emulator.total_frames)
        
        if not self.addresses:
            return state
        
        is_gsc = self.game_type in [GameType.POKEMON_GOLD_SILVER, GameType.POKEMON_CRYSTAL]
        
        try:
            # Player position
            if 'player_x' in self.addresses:
                state.player_position.x = self.read_byte(self.addresses['player_x'])
                state.player_position.y = self.read_byte(self.addresses['player_y'])
                
                # GSC uses two-byte map system (group, number)
                if is_gsc and 'player_map_group' in self.addresses:
                    map_group = self.read_byte(self.addresses['player_map_group'])
                    map_num = self.read_byte(self.addresses['player_map'])
                    state.player_position.map_id = (map_group << 8) | map_num
                    
                    # Try GSC map names first
                    map_key = (map_group, map_num)
                    if map_key in self.map_names_gsc:
                        state.player_position.map_name = self.map_names_gsc[map_key]
                    else:
                        state.player_position.map_name = f"Map {map_group}:{map_num}"
                else:
                    state.player_position.map_id = self.read_byte(self.addresses['player_map'])
                    state.player_position.map_name = self.map_names.get(
                        state.player_position.map_id, f"Map {state.player_position.map_id}"
                    )
                
                # Facing direction (GSC uses 0-3, Gen 1 uses 0/4/8/C)
                facing = self.read_byte(self.addresses['player_facing'])
                if is_gsc:
                    # GSC: 0=down, 1=up, 2=left, 3=right
                    facing_map = {0: "down", 1: "up", 2: "left", 3: "right"}
                    state.player_position.facing = facing_map.get(facing & 0x3, "down")
                else:
                    # Gen 1: 0=down, 4=up, 8=left, C=right
                    facing_map = {0: "down", 4: "up", 8: "left", 0xC: "right"}
                    state.player_position.facing = facing_map.get(facing & 0xC, "down")
            
            # Player name
            if 'player_name' in self.addresses:
                state.player_name = self.read_string(self.addresses['player_name'])
            
            # Money
            if 'money' in self.addresses:
                state.money = self.read_bcd(self.addresses['money'], 3)
            
            # Badges (GSC has both Johto and Kanto)
            if 'badges' in self.addresses:
                johto_badges = bin(self.read_byte(self.addresses['badges'])).count('1')
                kanto_badges = 0
                if is_gsc and 'kanto_badges' in self.addresses:
                    kanto_badges = bin(self.read_byte(self.addresses['kanto_badges'])).count('1')
                state.badges = johto_badges + kanto_badges
            
            # Party - GSC uses 48 bytes per Pokemon, Gen 1 uses 44
            party_mon_size = 48 if is_gsc else 44
            
            if 'party_count' in self.addresses:
                state.party_count = min(self.read_byte(self.addresses['party_count']), 6)
                
                if 'party_species' in self.addresses and state.party_count > 0:
                    for i in range(state.party_count):
                        species = self.read_byte(self.addresses['party_species'] + i)
                        
                        pokemon = PokemonData(
                            species_id=species,
                            species_name=self.pokemon_names.get(species, f"Pokemon#{species}")
                        )
                        
                        # Read detailed party data if available
                        if 'party_data' in self.addresses:
                            base = self.addresses['party_data'] + i * party_mon_size
                            
                            if is_gsc:
                                # GSC party_struct offsets:
                                # 0: species, 1: item, 2-3: moves[0-1], 4-5: moves[2-3]
                                # 6-7: OT ID, 8-10: exp, 11-12: HP EV, etc.
                                # 31: level, 32-33: status, 34-35: current HP
                                # 36-37: max HP, 38-39: attack, etc.
                                pokemon.level = self.read_byte(base + 31)
                                pokemon.current_hp = self.read_word(base + 34)
                                pokemon.max_hp = self.read_word(base + 36)
                                pokemon.status = self.read_byte(base + 32)
                            else:
                                # Gen 1 offsets
                                pokemon.level = self.read_byte(base + 33)
                                pokemon.current_hp = self.read_word(base + 1)
                                pokemon.max_hp = self.read_word(base + 34)
                                pokemon.status = self.read_byte(base + 4)
                        
                        state.party.append(pokemon)
            
            # Battle state
            if 'in_battle' in self.addresses:
                battle_val = self.read_byte(self.addresses['in_battle'])
                state.battle.in_battle = battle_val != 0
                
                if state.battle.in_battle:
                    if 'battle_type' in self.addresses:
                        bt = self.read_byte(self.addresses['battle_type'])
                        state.battle.is_wild = (bt == 1)
                        state.battle.is_trainer = (bt == 2)
                    
                    if 'enemy_species' in self.addresses:
                        species = self.read_byte(self.addresses['enemy_species'])
                        state.battle.enemy_species = species
                        state.battle.enemy_name = self.pokemon_names.get(species, f"Enemy#{species}")
                    
                    if 'enemy_level' in self.addresses:
                        state.battle.enemy_level = self.read_byte(self.addresses['enemy_level'])
                    
                    if 'enemy_hp' in self.addresses:
                        state.battle.enemy_hp = self.read_word(self.addresses['enemy_hp'])
                        state.battle.enemy_max_hp = self.read_word(self.addresses['enemy_max_hp'])
                    
                    if 'battle_menu' in self.addresses:
                        state.battle.menu_state = self.read_byte(self.addresses['battle_menu'])
            
            # Menu state
            if 'menu_open' in self.addresses:
                menu_val = self.read_byte(self.addresses['menu_open'])
                state.menu.in_menu = menu_val != 0
                
                if 'cursor_pos' in self.addresses:
                    state.menu.cursor_position = self.read_byte(self.addresses['cursor_pos'])
                
                if 'text_progress' in self.addresses:
                    text_val = self.read_byte(self.addresses['text_progress'])
                    state.menu.text_active = text_val != 0
            
            # Game progress flags
            if 'has_pokedex' in self.addresses:
                state.has_pokedex = self.read_byte(self.addresses['has_pokedex']) != 0
            
            if 'starter_flag' in self.addresses:
                state.has_starter = self.read_byte(self.addresses['starter_flag']) != 0
            
            if 'game_state' in self.addresses:
                gs = self.read_byte(self.addresses['game_state'])
                state.game_started = gs not in [0, 0xFF]  # 0 usually means title screen
            
            # Store some raw values for debugging
            state.raw_values = {
                'game_state': self.read_byte(self.addresses.get('game_state', 0)),
                'menu_open': self.read_byte(self.addresses.get('menu_open', 0)),
            }
            
        except Exception as e:
            print(f"[MemoryManager] Error reading state: {e}")
        
        return state
    
    def add_watcher(self, name: str, address: int, value_type: str = "byte"):
        """Add a custom memory watcher."""
        self.watchers[name] = (address, value_type)
    
    def get_watched_values(self) -> Dict[str, int]:
        """Get all watched memory values."""
        values = {}
        for name, (address, vtype) in self.watchers.items():
            if vtype == "word":
                values[name] = self.read_word(address)
            elif vtype == "bcd3":
                values[name] = self.read_bcd(address, 3)
            else:
                values[name] = self.read_byte(address)
        return values
    
    def is_player_stuck(self, frames: int = 60) -> bool:
        """Check if player hasn't moved in N frames."""
        if len(self.position_history) < 2:
            return False
        
        recent = self.position_history[-frames:] if len(self.position_history) >= frames else self.position_history
        if not recent:
            return False
        
        first = recent[0]
        return all(p.x == first.x and p.y == first.y for p in recent)
    
    def get_movement_direction_to(self, target_x: int, target_y: int) -> Optional[str]:
        """Get direction to move towards target."""
        state = self.get_state()
        dx = target_x - state.player_position.x
        dy = target_y - state.player_position.y
        
        if abs(dx) > abs(dy):
            return "RIGHT" if dx > 0 else "LEFT"
        elif dy != 0:
            return "DOWN" if dy > 0 else "UP"
        return None
    
    def export_state_json(self) -> str:
        """Export current state as JSON."""
        state = self.get_state()
        return json.dumps({
            'game_type': state.game_type.name,
            'frame': state.frame_count,
            'player': {
                'name': state.player_name,
                'position': {
                    'x': state.player_position.x,
                    'y': state.player_position.y,
                    'map': state.player_position.map_name,
                    'facing': state.player_position.facing,
                },
                'money': state.money,
                'badges': state.badges,
            },
            'party': [
                {
                    'species': p.species_name,
                    'level': p.level,
                    'hp': f"{p.current_hp}/{p.max_hp}",
                }
                for p in state.party
            ],
            'in_battle': state.battle.in_battle,
            'in_menu': state.menu.in_menu,
            'has_starter': state.has_starter,
        }, indent=2)
    
    def scan_for_value(self, value: int, start: int = 0xC000, end: int = 0xE000) -> List[int]:
        """
        Scan memory range for a specific byte value.
        Useful for finding addresses when values are known.
        """
        matches = []
        for addr in range(start, end):
            try:
                if self.read_byte(addr) == value:
                    matches.append(addr)
            except:
                pass
        return matches
    
    def scan_for_change(self, start: int = 0xC000, end: int = 0xE000) -> Dict[int, Tuple[int, int]]:
        """
        Compare current memory to cached state and find changes.
        Returns dict of address -> (old_value, new_value)
        """
        if not hasattr(self, '_scan_cache'):
            # First call - just cache current state
            self._scan_cache = {}
            for addr in range(start, end):
                try:
                    self._scan_cache[addr] = self.read_byte(addr)
                except:
                    pass
            return {}
        
        changes = {}
        for addr in range(start, end):
            try:
                new_val = self.read_byte(addr)
                old_val = self._scan_cache.get(addr, 0)
                if new_val != old_val:
                    changes[addr] = (old_val, new_val)
                self._scan_cache[addr] = new_val
            except:
                pass
        return changes
    
    def dump_memory_region(self, start: int, length: int = 64) -> str:
        """Dump a region of memory as hex for debugging."""
        lines = []
        for offset in range(0, length, 16):
            addr = start + offset
            hex_bytes = ' '.join(f'{self.read_byte(addr + i):02X}' for i in range(min(16, length - offset)))
            ascii_bytes = ''.join(
                chr(self.read_byte(addr + i)) if 32 <= self.read_byte(addr + i) < 127 else '.'
                for i in range(min(16, length - offset))
            )
            lines.append(f'{addr:04X}: {hex_bytes:<48} {ascii_bytes}')
        return '\n'.join(lines)
    
    def debug_addresses(self) -> str:
        """Print all configured addresses and their current values."""
        lines = [f"=== Memory Debug for {self.game_type.name} ==="]
        for name, addr in sorted(self.addresses.items()):
            try:
                val = self.read_byte(addr)
                lines.append(f"  {name}: 0x{addr:04X} = {val} (0x{val:02X})")
            except Exception as e:
                lines.append(f"  {name}: 0x{addr:04X} = ERROR: {e}")
        return '\n'.join(lines)


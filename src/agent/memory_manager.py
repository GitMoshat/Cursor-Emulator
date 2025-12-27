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

# Pokemon Gold/Silver/Crystal addresses
POKEMON_GSC_ADDRESSES = {
    'game_state': 0xD4B4,
    'player_name': 0xD47D,
    'player_x': 0xD4E3,
    'player_y': 0xD4E4,
    'player_map': 0xDCB5,
    'player_facing': 0xD4DE,
    'money': 0xD573,
    'badges': 0xD857,  # Johto badges
    'kanto_badges': 0xD858,
    'party_count': 0xDCD7,
    'party_species': 0xDCD8,
    'party_data': 0xDCDF,
    'in_battle': 0xD22D,
    'battle_type': 0xD230,
    'enemy_species': 0xD206,
    'enemy_level': 0xD213,
}

# Pokemon species names (Gen 1)
POKEMON_NAMES_GEN1 = {
    0: "???", 1: "RHYDON", 2: "KANGASKHAN", 3: "NIDORAN♂", 4: "CLEFAIRY",
    5: "SPEAROW", 6: "VOLTORB", 7: "NIDOKING", 8: "SLOWBRO", 9: "IVYSAUR",
    10: "EXEGGUTOR", 21: "MEW", 27: "GROWLITHE", 33: "ONIX", 36: "TANGELA",
    49: "PONYTA", 52: "MAROWAK", 55: "ABRA", 58: "ALAKAZAM", 64: "PIDGEOTTO",
    65: "PIDGEOT", 66: "STARMIE", 70: "JYNX", 71: "SNORLAX", 72: "POLIWHIRL",
    76: "GEODUDE", 78: "CUBONE", 82: "DODRIO", 83: "PRIMEAPE", 84: "DUGTRIO",
    85: "VENOMOTH", 88: "DEWGONG", 96: "CATERPIE", 97: "METAPOD", 98: "BUTTERFREE",
    99: "MACHAMP", 102: "GOLDUCK", 103: "HYPNO", 106: "KADABRA", 107: "GRAVELER",
    108: "CHANSEY", 109: "MACHOKE", 110: "MR. MIME", 112: "SCYTHER", 113: "STARYU",
    115: "JIGGLYPUFF", 116: "WIGGLYTUFF", 117: "EEVEE", 118: "FLAREON",
    119: "JOLTEON", 120: "VAPOREON", 121: "MACHOP", 122: "ZUBAT", 123: "EKANS",
    124: "PARAS", 125: "POLIWAG", 126: "POLIWRATH", 127: "WEEDLE", 128: "KAKUNA",
    129: "BEEDRILL", 133: "DODUO", 134: "PRIMEAPE", 136: "MAGNEMITE", 138: "CHARMANDER",
    139: "SQUIRTLE", 140: "CHARMELEON", 141: "WARTORTLE", 142: "CHARIZARD",
    147: "ODDISH", 148: "GLOOM", 149: "VILEPLUME", 150: "BELLSPROUT",
    151: "WEEPINBELL", 152: "VICTREEBEL", 
    # Starters and common Pokemon
    153: "BULBASAUR", 154: "IVYSAUR", 155: "VENUSAUR",
    176: "PIKACHU", 177: "RAICHU",
    # More will be added as needed
}

# Map names for Gen 1
MAP_NAMES_GEN1 = {
    0: "Pallet Town", 1: "Viridian City", 2: "Pewter City", 3: "Cerulean City",
    4: "Lavender Town", 5: "Vermilion City", 6: "Celadon City", 7: "Fuchsia City",
    8: "Cinnabar Island", 9: "Indigo Plateau", 10: "Saffron City",
    12: "Route 1", 13: "Route 2", 14: "Route 3", 15: "Route 4",
    33: "Player's House 1F", 34: "Player's House 2F",
    37: "Prof Oak's Lab", 38: "Viridian Pokemon Center",
    40: "Viridian Mart",
    # Routes
    # ... more maps
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
        self.pokemon_names = POKEMON_NAMES_GEN1
        self.map_names = MAP_NAMES_GEN1
        
        # Cache for expensive reads
        self._state_cache: Optional[GameState] = None
        self._cache_frame = -1
        
        # History for change detection
        self.position_history: List[Position] = []
        self.max_history = 100
        
        # Custom watchers
        self.watchers: Dict[str, Tuple[int, str]] = {}  # name -> (address, type)
    
    def detect_game(self):
        """Detect game type from ROM header."""
        try:
            # Read ROM title (0x134-0x143)
            title_bytes = bytes(self.memory.rom[0x134:0x144])
            title = title_bytes.decode('ascii', errors='ignore').strip('\x00')
            
            title_lower = title.lower()
            
            if 'pokemon red' in title_lower or 'pokemon blue' in title_lower:
                self.game_type = GameType.POKEMON_RED_BLUE
                self.addresses = POKEMON_RB_ADDRESSES
            elif 'pokemon yellow' in title_lower or 'pokemon pikachu' in title_lower:
                self.game_type = GameType.POKEMON_YELLOW
                self.addresses = POKEMON_YELLOW_ADDRESSES
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
        
        try:
            # Player position
            if 'player_x' in self.addresses:
                state.player_position.x = self.read_byte(self.addresses['player_x'])
                state.player_position.y = self.read_byte(self.addresses['player_y'])
                state.player_position.map_id = self.read_byte(self.addresses['player_map'])
                state.player_position.map_name = self.map_names.get(
                    state.player_position.map_id, f"Map {state.player_position.map_id}"
                )
                
                # Facing direction
                facing = self.read_byte(self.addresses['player_facing'])
                facing_map = {0: "down", 4: "up", 8: "left", 0xC: "right"}
                state.player_position.facing = facing_map.get(facing & 0xC, "down")
            
            # Player name
            if 'player_name' in self.addresses:
                state.player_name = self.read_string(self.addresses['player_name'])
            
            # Money
            if 'money' in self.addresses:
                state.money = self.read_bcd(self.addresses['money'], 3)
            
            # Badges
            if 'badges' in self.addresses:
                state.badges = bin(self.read_byte(self.addresses['badges'])).count('1')
            
            # Party
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
                            base = self.addresses['party_data'] + i * 44
                            pokemon.level = self.read_byte(base + 33)  # Level offset
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


GM_PROGRAMS = {
    0: "Acoustic Grand Piano", 
    1: "Bright Acoustic Piano", 
    2: "Electric Grand Piano",
    3: "Honky-tonk Piano", 
    4: "Electric Piano 1", 
    5: "Electric Piano 2",
    6: "Harpsichord", 
    7: "Clavi",

    8: "Celesta", 
    9: "Glockenspiel", 
    10: "Music Box", 
    11: "Vibraphone",
    12: "Marimba", 
    13: "Xylophone", 
    14: "Tubular Bells", 
    15: "Dulcimer",

    16: "Drawbar Organ", 
    17: "Percussive Organ", 
    18: "Rock Organ", 
    19: "Church Organ",
    20: "Reed Organ", 
    21: "Accordion", 
    22: "Harmonica", 
    23: "Tango Accordion",

    24: "Acoustic Guitar (nylon)", 
    25: "Acoustic Guitar (steel)", 
    26: "Electric Guitar (jazz)",
    27: "Electric Guitar (clean)", 
    28: "Electric Guitar (muted)", 
    29: "Overdriven Guitar",
    30: "Distortion Guitar", 
    31: "Guitar Harmonics",

    32: "Acoustic Bass", 
    33: "Electric Bass (finger)", 
    34: "Electric Bass (pick)",
    35: "Fretless Bass", 
    36: "Slap Bass 1", 
    37: "Slap Bass 2", 
    38: "Synth Bass 1",
    39: "Synth Bass 2",

    40: "Violin", 
    41: "Viola", 
    42: "Cello", 
    43: "Contrabass",
    44: "Tremolo Strings", 
    45: "Pizzicato Strings", 
    46: "Orchestral Harp",
    47: "Timpani",

    48: "String Ensemble 1", 
    49: "String Ensemble 2", 
    50: "Synth Strings 1",
    51: "Synth Strings 2", 
    52: "Choir Aahs", 
    53: "Voice Oohs", 
    54: "Synth Choir",
    55: "Orchestra Hit",

    56: "Trumpet", 
    57: "Trombone", 
    58: "Tuba", 
    59: "Muted Trumpet",
    60: "French Horn", 
    61: "Brass Section", 
    62: "Synth Brass 1", 
    63: "Synth Brass 2",

    64: "Soprano Sax", 
    65: "Alto Sax", 
    66: "Tenor Sax", 
    67: "Baritone Sax",
    68: "Oboe", 
    69: "English Horn", 
    70: "Bassoon", 
    71: "Clarinet",

    72: "Piccolo", 
    73: "Flute", 
    74: "Recorder", 
    75: "Pan Flute",
    76: "Blown Bottle", 
    77: "Shakuhachi", 
    78: "Whistle", 
    79: "Ocarina",

    80: "Lead 1 (square)", 
    81: "Lead 2 (sawtooth)", 
    82: "Lead 3 (calliope)",
    83: "Lead 4 (chiff)", 
    84: "Lead 5 (charang)", 
    85: "Lead 6 (voice)",
    86: "Lead 7 (fifths)", 
    87: "Lead 8 (bass + lead)",

    88: "Pad 1 (new age)", 
    89: "Pad 2 (warm)", 
    90: "Pad 3 (polysynth)",
    91: "Pad 4 (choir)", 
    92: "Pad 5 (bowed)", 
    93: "Pad 6 (metallic)",
    94: "Pad 7 (halo)", 
    95: "Pad 8 (sweep)",

    96: "FX 1 (rain)", 
    97: "FX 2 (soundtrack)", 
    98: "FX 3 (crystal)",
    99: "FX 4 (atmosphere)", 
    100: "FX 5 (brightness)", 
    101: "FX 6 (goblins)",
    102: "FX 7 (echoes)", 
    103: "FX 8 (sci-fi)",

    104: "Sitar", 
    105: "Banjo", 
    106: "Shamisen", 
    107: "Koto",
    108: "Kalimba", 
    109: "Bagpipe", 
    110: "Fiddle", 
    111: "Shanai",

    112: "Tinkle Bell", 
    113: "Agogo", 
    114: "Steel Drums", 
    115: "Woodblock",
    116: "Taiko Drum", 
    117: "Melodic Tom", 
    118: "Synth Drum", 
    119: "Reverse Cymbal",

    120: "Guitar Fret Noise", 
    121: "Breath Noise", 
    122: "Seashore", 
    123: "Bird Tweet",
    124: "Telephone Ring", 
    125: "Helicopter", 
    126: "Applause", 
    127: "Gunshot",
}

from music21 import instrument
# -----------------------------
# Enhanced instrument mapping
# -----------------------------
INSTRUMENT_MAPPING = {
    # Keyboards
    "piano": instrument.Piano,
    "fortepiano": instrument.Piano,
    "harpsichord": instrument.Harpsichord,
    "celesta": instrument.Celesta,
    "organ": instrument.Organ,
    "hammond": instrument.ElectricOrgan,
    "accordion": instrument.Accordion,
    
    # Strings
    "violin": instrument.Violin,
    "viola": instrument.Viola,
    "cello": instrument.Violoncello,
    "contrabass": instrument.Contrabass,
    "bass": instrument.ElectricBass,
    "guitar": instrument.ElectricGuitar,
    "harp": instrument.Harp,
    "strings": instrument.StringInstrument,
    
    # Brass
    "trumpet": instrument.Trumpet,
    "trombone": instrument.Trombone,
    "horn": instrument.Horn,
    "french horn": instrument.Horn,
    "tuba": instrument.Tuba,
    "brass": instrument.BrassInstrument,
    
    # Woodwinds
    "flute": instrument.Flute,
    "piccolo": instrument.Piccolo,
    "oboe": instrument.Oboe,
    "clarinet": instrument.Clarinet,
    "bassoon": instrument.Bassoon,
    "saxophone": instrument.Saxophone,
    "alto sax": instrument.AltoSaxophone,
    "tenor sax": instrument.TenorSaxophone,
    "soprano sax": instrument.SopranoSaxophone,
    "baritone sax": instrument.BaritoneSaxophone,
    
    # Percussion
    "drums": instrument.Percussion,
    "timpani": instrument.Timpani,
    "xylophone": instrument.Xylophone,
    "marimba": instrument.Marimba,
    "vibraphone": instrument.Vibraphone,
    
    # Voices
    "choir": instrument.Choir,
    "soprano": instrument.Soprano,
    "alto": instrument.Alto,
    "tenor": instrument.Tenor,
    "bass voice": instrument.Bass,
    
    # Other
    "banjo": instrument.Banjo,
    "mandolin": instrument.Mandolin,
    "sitar": instrument.Sitar,
}

# MIDI Program number mapping
MIDI_PROGRAM_MAPPING = {
    0: instrument.Piano,           # Acoustic Grand Piano
    1: instrument.Piano,           # Bright Acoustic Piano
    4: instrument.ElectricPiano,   # Electric Piano 1
    6: instrument.Harpsichord,     # Harpsichord
    16: instrument.ElectricOrgan,  # Hammond Organ
    19: instrument.Organ,          # Church Organ
    24: instrument.Guitar,         # Acoustic Guitar (nylon)
    25: instrument.ElectricGuitar, # Acoustic Guitar (steel)
    32: instrument.ElectricBass,   # Acoustic Bass
    40: instrument.Violin,         # Violin
    41: instrument.Viola,          # Viola
    42: instrument.Violoncello,    # Cello
    43: instrument.Contrabass,     # Contrabass
    46: instrument.Harp,           # Harp
    56: instrument.Trumpet,        # Trumpet
    57: instrument.Trombone,       # Trombone
    58: instrument.Tuba,           # Tuba
    60: instrument.Horn,           # French Horn
    64: instrument.SopranoSaxophone, # Soprano Sax
    65: instrument.AltoSaxophone,    # Alto Sax
    66: instrument.TenorSaxophone,   # Tenor Sax
    67: instrument.BaritoneSaxophone, # Baritone Sax
    68: instrument.Oboe,           # Oboe
    70: instrument.Bassoon,        # Bassoon
    71: instrument.Clarinet,       # Clarinet
    72: instrument.Piccolo,        # Piccolo
    73: instrument.Flute,          # Flute
}
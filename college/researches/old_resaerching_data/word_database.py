import sqlite3

DB_PATH1 = "nuance_words.db"
DB_PATH2 = "rules.db"

conn = sqlite3.connect(DB_PATH1)
c = conn.cursor()
c.execute("""
    CREATE TABLE IF NOT EXISTS phoneme_words (
        type TEXT CHECK(type IN ('consonant', 'vowel')),
        phoneme TEXT,
        nuance TEXT CHECK(nuance IN ('neutral', 'soft', 'formal', 'netslang', 'retro', 'stylish')),
        tension_score INTEGER,
        emotion_score INTEGER,
        context_score INTEGER
    )
""")
# Create initial data
phoneme_data = [
    ("vowel", "a", "neutral", 0, 0, 1),
    ("consonant", "b", "neutral", 0, 0, 1),
    ("consonant", "c", "neutral", 0, 0, 1),
    ("consonant", "d", "neutral", 0, 0, 1),
    ("vowel", "e", "neutral", 0, 0, 1),
    ("consonant", "f", "neutral", 0, 0, 1),
    ("consonant", "g", "neutral", 0, 0, 1),
    ("consonant", "h", "neutral", 0, 0, 1),
    ("vowel", "i", "neutral", 0, 0, 1),
    ("consonant", "j", "neutral", 0, 0, 1),
    ("consonant", "k", "neutral", 0, 0, 1),
    ("consonant", "l", "neutral", 0, 0, 1),
    ("consonant", "m", "neutral", 0, 0, 1),
    ("consonant", "n", "neutral", 0, 0, 1),
    ("vowel", "o", "neutral", 0, 0, 1),
    ("consonant", "p", "neutral", 0, 0, 1),
    ("consonant", "q", "neutral", 0, 0, 1),
    ("consonant", "r", "neutral", 0, 0, 1),
    ("consonant", "s", "neutral", 0, 0, 1),
    ("consonant", "ss", "neutral", 0, 0, 1),
    ("consonant", "t", "neutral", 0, 0, 1),
    ("vowel", "u", "neutral", 0, 0, 1),
    ("consonant", "v", "neutral", 0, 0, 1),
    ("consonant", "w", "neutral", 0, 0, 1),
    ("consonant", "x", "neutral", 0, 0, 1),
    ("consonant", "y", "neutral", 0, 0, 1),
    ("consonant", "z", "neutral", 0, 0, 1),
    ("vowel", "ar", "neutral", 0, 0, 1),
    ("vowel", "ee", "neutral", 0, 0, 1),
    ("vowel", "ea", "neutral", 0, 0, 1),
    ("vowel", "oo", "neutral", 0, 0, 1),
    ("consonant", "dg", "neutral", 0, 0, 1),
    ("consonant", "ch", "neutral", 0, 0, 1),
    ("consonant", "tch", "neutral", 0, 0, 1),
    ("consonant", "th", "neutral", 0, 0, 1),
    ("consonant", "ts", "neutral", 0, 0, 1),
    ("consonant", "sh", "neutral", 0, 0, 1),
    ("consonant", "zh", "neutral", 0, 0, 1),
    ("consonant", "si", "neutral", 0, 0, 1),
    ("consonant", "ng", "neutral", 0, 0, 1),
    ("vowel", "aw", "neutral", 0, 0, 1),
    ("vowel", "or", "neutral", 0, 0, 1),
    ("vowel", "ay", "neutral", 0, 0, 1),
    ("vowel", "ei", "neutral", 0, 0, 1),
    ("vowel", "igh", "neutral", 0, 0, 1),
    ("vowel", "ow", "neutral", 0, 0, 1),
    ("vowel", "ou", "neutral", 0, 0, 1),
    ("vowel", "oi", "neutral", 0, 0, 1),
    ("vowel", "oy", "neutral", 0, 0, 1)
]
c.executemany("INSERT INTO phoneme_words (type, phoneme, nuance, tension_score, emotion_score, context_score) VALUES (?, ?, ?, ?, ?, ?)", phoneme_data)
conn.commit()

# Create rules table
conn = sqlite3.connect(DB_PATH2)
c = conn.cursor()
c.execute("""
    CREATE TABLE IF NOT EXISTS rules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        romaji TEXT,
        replacement TEXT
    )
""")
rules = {
    "neutral": {
        ("a", "a"), ("b", "b"), ("c", "si"), ("d", "d"), ("e", "ii"), ("f", "ef"),
        ("g", "g"), ("h", "h"), ("i", "i"), ("j", "j"), ("k", "k"), ("l", "l"),
        ("m", "m"), ("n", "n"), ("o", "o"), ("p", "p"), ("q", "k"), ("r", "l"),
        ("s", "s"), ("t", "t"), ("u", "u"), ("v", "b"), ("w", "h"), ("x", "ks"),
        ("y", "y"), ("z", "z"), ("sh", "sh"), ("ch", "ch"), ("ts", "ts"),
        ("th", "th"), ("dg", "d"), ("ng", "ng"), ("wh", "f"), ("si", "si"),
        ("ss", "s"), ("tch", "ch"), ("zh", "j"), ("ee", "i"), ("ea", "i"),
        ("oo", "u"), ("ar", "a"), ("aw", "o"), ("or", "o")
    },
    "soft": {
        ("a", "a"), ("b", "b"), ("c", "k"), ("d", "d"), ("e", "e"), ("f", "f"),
        ("g", "g"), ("h", "h"), ("i", "i"), ("j", "j"), ("k", "k"), ("l", "l"),
        ("m", "m"), ("n", "n"), ("o", "o"), ("p", "p"), ("q", "k"), ("r", "l"),
        ("s", "s"), ("t", "t"), ("u", "u"), ("v", "b"), ("w", "h"), ("x", "ks"),
        ("y", "y"), ("z", "z"), ("sh", "sh"), ("ch", "ch"), ("ts", "ts"),
        ("th", "th"), ("dg", "d"), ("ng", "ng"), ("wh", "f"), ("si", "si"),
        ("ss", "s"), ("tch", "ch"), ("zh", "j"), ("ee", "i"), ("ea", "i"),
        ("oo", "u"), ("ar", "a"), ("aw", "o"), ("or", "o")
    },
    "formal": {
        ("a", "a"), ("b", "b"), ("c", "si"), ("d", "d"), ("e", "e"), ("f", "f"),
        ("g", "g"), ("h", "h"), ("i", "i"), ("j", "j"), ("k", "k"), ("l", "l"),
        ("m", "m"), ("n", "n"), ("o", "o"), ("p", "p"), ("q", "k"), ("r", "l"),
        ("s", "s"), ("t", "t"), ("u", "u"), ("v", "b"), ("w", "h"), ("x", "ks"),
        ("y", "y"), ("z", "z"), ("sh", "sh"), ("ch", "ch"), ("ts", "ts"),
        ("th", "th"), ("dg", "d"), ("ng", "ng"), ("wh", "f"), ("si", "si"),
        ("ss", "s"), ("tch", "ch"), ("zh", "j"), ("ee", "i"), ("ea", "i"),
        ("oo", "u"), ("ar", "a"), ("aw", "o"), ("or", "o")
    },
    "netslang": {
        ("a", "a"), ("b", "b"), ("c", "si"), ("d", "d"), ("e", "e"), ("f", "f"),
        ("g", "g"), ("h", "h"), ("i", "i"), ("j", "j"), ("k", "k"), ("l", "l"),
        ("m", "m"), ("n", "n"), ("o", "o"), ("p", "p"), ("q", "k"), ("r", "l"),
        ("s", "s"), ("t", "t"), ("u", "u"), ("v", "b"), ("w", "h"), ("x", "ks"),
        ("y", "y"), ("z", "z"), ("sh", "sh"), ("ch", "ch"), ("ts", "ts"),
        ("th", "th"), ("dg", "d"), ("ng", "ng"), ("wh", "f"), ("si", "si"),
        ("ss", "s"), ("tch", "ch"), ("zh", "j"), ("ee", "i"), ("ea", "i"),
        ("oo", "u"), ("ar", "a"), ("aw", "o"), ("or", "o")
    },
    "retro": {
        ("a", "a"), ("b", "b"), ("c", "si"), ("d", "d"), ("e", "e"), ("f", "f"),
        ("g", "g"), ("h", "h"), ("i", "i"), ("j", "j"), ("k", "k"), ("l", "l"),
        ("m", "m"), ("n", "n"), ("o", "o"), ("p", "p"), ("q", "k"), ("r", "l"),
        ("s", "s"), ("t", "t"), ("u", "u"), ("v", "b"), ("w", "h"), ("x", "ks"),
        ("y", "y"), ("z", "z"), ("sh", "sh"), ("ch", "ch"), ("ts", "ts"),
        ("th", "th"), ("dg", "d"), ("ng", "ng"), ("wh", "f"), ("si", "si"),
        ("ss", "s"), ("tch", "ch"), ("zh", "j"), ("ee", "i"), ("ea", "i"),
        ("oo", "u"), ("ar", "a"), ("aw", "o"), ("or", "o")
    },
    "stylish": {
        ("a", "a"), ("b", "b"), ("c", "si"), ("d", "d"), ("e", "e"), ("f", "f"),
        ("g", "g"), ("h", "h"), ("i", "i"), ("j", "j"), ("k", "k"), ("l", "l"),
        ("m", "m"), ("n", "n"), ("o", "o"), ("p", "p"), ("q", "k"), ("r", "l"),
        ("s", "s"), ("t", "t"), ("u", "u"), ("v", "b"), ("w", "h"), ("x", "ks"),
        ("y", "y"), ("z", "z"), ("sh", "sh"), ("ch", "ch"), ("ts", "ts"),
        ("th", "th"), ("dg", "d"), ("ng", "ng"), ("wh", "f"), ("si", "si"),
        ("ss", "s"), ("tch", "ch"), ("zh", "j"), ("ee", "i"), ("ea", "i"),
        ("oo", "u"), ("ar", "a"), ("aw", "o"), ("or", "o")
    }
}
for nuance, ruleset in rules.items():
    for src, dst in ruleset:
        c.execute("INSERT INTO rules (romaji, replacement) VALUES (?, ?)", (src, dst))
conn.commit()
conn.close()
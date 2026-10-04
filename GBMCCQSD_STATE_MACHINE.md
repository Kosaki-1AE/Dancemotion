# GBMCCQSD 8-bit State Machine (prototype)

This directory adds a minimal simulator for the current GBMCCQSD state-transition hypothesis.

## Current encoding

```text
8 bit = [ RRRR ][ CC ][ OO ]
          Room    |     |
                  |     +-- local operation
                  +-------- responsibility
```

- `RRRR` (upper 4 bits): current Room / WHERE
- `CC` (next 2 bits): responsibility / WHAT TO CARE ABOUT
- `OO` (lower 2 bits): local operation / HOW

Local operation labels currently follow the design notes:

| OO | Operation |
|---|---|
| `00` | 打つ |
| `01` | 弾く |
| `10` | 外す |
| `11` | 置く |

## O/S/J/T coarse routing

The lower nibble is coarse-grained into four routing modes:

| CC | OO | Mode |
|---|---|---|
| `00/01` | `00/01` | Observe |
| `00/01` | `10/11` | Straight |
| `10/11` | `00/01` | Jab |
| `10/11` | `10/11` | Twist |

This is implemented by `classify_mode()`.

## One simulator cycle

```text
current Room
    ↓
construct lower 4 bits (CCOO)
    ↓
Phi_R(CC, OO)       # executor
    ↓
classify O/S/J/T
    ↓
Gamma(Room, result, mode)  # router
    ↓
next Room
```

The simulator deliberately does **not** hard-code the still-unfixed semantics of `Gamma`.
`TableRouter` lets a caller define `(Room, Mode) -> Next Room` explicitly. Missing rules conservatively keep the current Room.

That means the prototype can run today without pretending that the theory's Room-routing law is already settled.

## Run

From the repository root:

```bash
python -m gbmccqsd.cli --room 0 0x0 0x6 0xA 0xC
```

Example with routing rules:

```bash
python -m gbmccqsd.cli \
  --room 0 \
  --rule 0x0:Straight:0x1 \
  --rule 0x1:Twist:0x4 \
  0x2 0xA
```

JSON trace:

```bash
python -m gbmccqsd.cli --json --room 0 0x0 0x6 0xA 0xC
```

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover:

- 8-bit encode/decode
- `00/01/10/11` local-operation mapping
- all 16 lower-nibble -> Observe/Straight/Jab/Twist classifications
- conservative fallback routing
- explicit table routing

## Important: intentionally unresolved

The following are left configurable rather than guessed:

1. The semantic names of responsibility states `CC=00/01/10/11`.
2. The true `Gamma` law for selecting the next Room.
3. Whether entering the next Room should automatically reset the lower nibble to a canonical Observe state.
4. Domain-specific `Phi_R` behavior (dance, dialogue, code generation, etc.).

Those can be added without changing the 8-bit representation or the simulator interface.

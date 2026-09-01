# Usage

```bash
./gbmc_map Groove
./gbmc_map Attention
./gbmc_map Pipe
./gbmc_map Translator
./gbmc_map --nodes
```

## Expected idea

`Groove` should not only return:

```text
Groove -> Resonance
```

It should also show where `Resonance` sits in the local GBMC structure.

```text
Resonance
├── Port
├── Flow
└── ...
```

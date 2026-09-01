# GBMC Map

This is a minimal CLI prototype for showing the position of an external concept inside GBMC.

It is not just a dictionary.

It answers:

```text
Where is this concept located in GBMC?
What nodes are nearby?
Why is it there?
```

## Build

```bash
mkdir build
cd build
cmake ..
cmake --build .
./gbmc_map Groove
```

Direct build:

```bash
g++ -std=c++17 -Iinclude src/main.cpp -o gbmc_map
./gbmc_map Groove
```

## Examples

```bash
./gbmc_map Groove
./gbmc_map Attention
./gbmc_map Pipe
./gbmc_map Observer
./gbmc_map --nodes
```

## Difference from gbmc-concept-translator

`gbmc-concept-translator`:

```text
Attention -> Port
```

`gbmc-map`:

```text
Attention -> Port
Port is a mediation surface.
Nearby nodes:
  Port -> Module
  Port -> Translator
  Port -> Resonance
```

So this prototype gives position, not only translation.

## Current GBMC Nodes

```text
Module
Port
Translator
ResponsibilityRange
Resonance
ReferenceFrame
Flow
MeaningWall
ModuleState
```

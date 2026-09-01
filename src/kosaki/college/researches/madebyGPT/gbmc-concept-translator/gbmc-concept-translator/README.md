# GBMC Concept Translator

This is a minimal CLI prototype for translating external concepts into GBMC concepts.

It is intentionally different from the control-flow prototypes.

## Purpose

The goal is not:

```text
angle -> pwm
```

The goal is:

```text
Attention -> Port
Groove -> Resonance
Pipe -> Port
Observer -> Translator
```

## Core Idea

```text
external concept
-> candidate GBMC concept
-> confidence
-> reason
```

## Build

```bash
mkdir build
cd build
cmake ..
cmake --build .
./gbmc_translate Attention
```

Direct build:

```bash
g++ -std=c++17 -Iinclude src/main.cpp -o gbmc_translate
./gbmc_translate Attention
```

## Examples

```bash
./gbmc_translate Attention
./gbmc_translate Groove
./gbmc_translate Pipe
./gbmc_translate Observer
./gbmc_translate --all
```

## Current GBMC Vocabulary

```text
Module
Port
Translator
ResponsibilityRange
Resonance
ReferenceFrame
ModuleState
```

## Why this matters

This prototype is the first step toward a web app that can align terms from papers, systems, dance, control theory, and GBMC.

The current version uses a hard-coded dictionary.

The future version should:

```text
paper text
-> concept extraction
-> GBMC mapping candidates
-> human correction
-> growing translation dictionary
```

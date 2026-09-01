# GBMC Core Translator

This version adds the **Translator** layer.

## Version History

```text
v0: Module / Port / ResponsibilityRange / Connection / Resonance
v1: Flow
v2: Signal / Transition / Responsibility clamp
v3: Candidate / Planner / Responsibility Arrow
v4: Translator / Concept Mapping
```

## Core Definition

GBMC is a framework for:

```text
isolating modules,
defining ports,
determining responsibility ranges,
testing resonance,
running state transitions,
selecting executable candidates,
and translating between incompatible types.
```

## New Concept: Translator

A translator converts one type into another.

```text
angle -> pwm
Attention -> Port
Groove -> Resonance
Observer -> Translator
```

## Human-Powered Aircraft Demo

The demo now uses:

```text
Joystick
-> Pico2W
-> AngleToPWM
-> Servo
```

This fixes the previous issue where:

```text
angle -> pwm
```

was blocked by the meaning wall.

## Build

```bash
mkdir build
cd build
cmake ..
cmake --build .
./gbmc_translator_demo
```

Direct build:

```bash
g++ -std=c++17 -Iinclude src/main.cpp -o gbmc_translator_demo
./gbmc_translator_demo
```

## Why this matters

This version makes GBMC less like a simple connection model and more like a translation engine.

It can now express:

```text
Module
-> Port
-> Translator
-> ResponsibilityRange
-> Resonance
```

This is closer to the idea of connecting papers, systems, dance concepts, and aircraft control through a common GBMC vocabulary.

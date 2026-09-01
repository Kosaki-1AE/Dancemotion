# GBMC Core Final Prototype

This is the fourth GBMC prototype.

It adds the **Policy / Planner** layer.

## Version History

```text
v0: Module / Port / ResponsibilityRange / Connection / Resonance
v1: Flow
v2: Signal / Transition / Responsibility clamp
v3: Candidate / Planner / Responsibility Arrow
```

## Core Definition

GBMC is a framework for:

```text
isolating modules,
defining ports,
determining responsibility ranges,
testing resonance,
running state transitions,
and selecting the safest executable candidate.
```

## Current Flow

```text
Joystick
-> Pico2W
-> ESP32
-> Servo
```

## Current Planner

The planner receives candidate pilot intentions:

```text
soft-left
center
soft-right
hard-right
panic-right
```

Each candidate is passed through the whole flow.

The planner chooses the candidate with the smallest responsibility violation.

## Build

```bash
mkdir build
cd build
cmake ..
cmake --build .
./gbmc_final_demo
```

Direct build:

```bash
g++ -std=c++17 -Iinclude src/main.cpp -o gbmc_final_demo
./gbmc_final_demo
```

## Why this matters

This version finally reaches:

```text
candidate distribution
-> meaning wall / responsibility range
-> selection
-> responsibility arrow
-> action
```

So it is the first prototype that feels close to the GBMC transition model.

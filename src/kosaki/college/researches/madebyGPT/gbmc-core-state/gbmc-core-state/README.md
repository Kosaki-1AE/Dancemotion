# GBMC Core State

This is the third GBMC prototype.

It adds the **State / Transition** layer.

## Versions

v0:

```text
Module
Port
ResponsibilityRange
Connection
Resonance
```

v1:

```text
Flow
```

v2:

```text
Signal
Transition
Responsibility clamp
```

## Core Definition

A module receives a signal, transforms it, and only outputs the part that stays inside its responsibility range.

```text
intention
-> candidate
-> responsibility range
-> safe transition
```

## Build

```bash
mkdir build
cd build
cmake ..
cmake --build .
./gbmc_state_demo
```

Direct build:

```bash
g++ -std=c++17 -Iinclude src/main.cpp -o gbmc_state_demo
./gbmc_state_demo
```

## Demo

The demo runs two cases:

```text
safe input      : 20 degrees
excessive input : 80 degrees
```

The excessive input is clipped by the module responsibility range.

This is the first prototype where GBMC behaves like a control system.

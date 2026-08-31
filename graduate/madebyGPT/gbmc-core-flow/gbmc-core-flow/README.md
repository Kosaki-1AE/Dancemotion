# GBMC Core Flow

This is the second minimal GBMC prototype.

It adds the **Flow** layer.

## What changed from v0?

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
Module
Port
ResponsibilityRange
Connection
Resonance
Flow
```

## Definition

A flow is a sequence of connections.

A flow is complete only when every connection resonates.

## Build

```bash
mkdir build
cd build
cmake ..
cmake --build .
./gbmc_flow_demo
```

Direct build:

```bash
g++ -std=c++17 -Iinclude src/main.cpp -o gbmc_flow_demo
./gbmc_flow_demo
```

## Current Demo

The demo checks two flows.

### Safe flow

```text
Joystick -> Pico2W -> ESP32 -> Servo
```

### Broken flow

```text
Pico2W -> Servo
```

The broken flow fails because the port types do not match.
That failure is treated as a meaning wall.

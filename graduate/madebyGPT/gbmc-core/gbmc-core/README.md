# GBMC Core

GBMC Core is a minimal C++ prototype for expressing GBMC as a modular connection system.

## Core Definition

GBMC is a general transition framework for:

```text
isolating modules,
defining ports,
determining responsibility ranges,
and constructing resonant connections.
```

## Concepts

### Module

A module is an isolated part. It does not directly depend on other modules. It only connects through ports.

### Port

A port is not just a connector. In this prototype, a port is the mediation surface that allows one module to interpret another module.

### Responsibility Range

A responsibility range is the region in which a module can guarantee a predictable and repeatable transition.

```text
ResponsibilityRange = [min_value, max_value]
```

### Resonance

A connection is resonant when:

```text
port types are compatible
and
responsibility ranges overlap
```

If either condition fails, the system reports a meaning wall.

## Build

### WSL / Linux / macOS

```bash
mkdir build
cd build
cmake ..
cmake --build .
./gbmc_demo
```

### Direct g++ build

```bash
g++ -std=c++17 -Iinclude src/main.cpp -o gbmc_demo
./gbmc_demo
```

### Direct clang++ build

```bash
clang++ -std=c++17 -Iinclude src/main.cpp -o gbmc_demo
./gbmc_demo
```

## Current Prototype

The demo creates:

```text
Pico2W -> ESP32 -> Servo
```

and checks whether each connection can resonate.

It also intentionally tries:

```text
Pico2W.angle -> Servo.pwm
```

which fails because the ports do not match. That failure represents a meaning wall.

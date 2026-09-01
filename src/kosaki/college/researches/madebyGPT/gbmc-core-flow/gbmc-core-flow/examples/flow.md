# Flow Layer

The flow layer checks whether a sequence of module connections can continue.

## Core Idea

```text
Module
-> Connection
-> Resonance check
-> Flow
```

A flow is complete only when every connection resonates.

## Meaning Wall

A meaning wall appears when:

- port types do not match,
- responsibility ranges do not overlap.

## Human-Powered Aircraft Example

Safe flow:

```text
Joystick.angle
-> Pico2W.angle
-> ESP32.pwm
-> Servo.pwm
```

Broken flow:

```text
Pico2W.angle
-> Servo.pwm
```

This is blocked because `angle` and `pwm` are different port types.

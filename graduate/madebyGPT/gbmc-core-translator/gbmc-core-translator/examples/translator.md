# Translator Layer

This version adds the Translator concept.

## Why Translator?

Previous versions could connect only when port types matched directly.

But real systems often require type conversion:

```text
angle -> pwm
concept A -> GBMC concept
dance feeling -> engineering term
```

## Core Definition

A Translator is a module that converts one signal type into another.

```text
input type
-> Translator
-> output type
```

## Human-Powered Aircraft Example

```text
Joystick.angle
-> Pico2W.angle
-> AngleToPWM
-> Servo.pwm
```

## Concept Translation Example

```text
Transformer.Attention -> GBMC.Port
StreetDance.Groove -> GBMC.Resonance
ControlTheory.Observer -> GBMC.Translator
```

This is the first prototype that can connect physical control and paper/concept mapping.

# Human-Powered Aircraft Example

This example treats each control component as a GBMC module.

## Modules

- Pico2W: receives pilot intention and creates control candidates.
- ESP32: filters executable commands.
- Servo: produces physical motion.
- Battery: constrains power responsibility.

## GBMC Interpretation

```text
Pilot intention
-> candidate distribution
-> port compatibility
-> responsibility range check
-> resonance
-> command output
```

## Meaning Wall

A meaning wall appears when:

- port types do not match,
- responsibility ranges do not overlap,
- the requested transition exceeds the module guarantee range.

Example:

```text
Pico2W.angle -> Servo.pwm
```

is blocked because the port types are different. The control candidate must pass through ESP32, which translates the candidate into an executable PWM command.

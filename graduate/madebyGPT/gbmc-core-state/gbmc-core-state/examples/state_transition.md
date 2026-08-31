# State / Transition Layer

This version adds values.

Previous versions checked only whether modules can connect.

This version checks what happens when a signal actually flows.

## Core Idea

```text
Signal
-> Module transition
-> ResponsibilityRange clamp
-> next Module
```

## Meaning

If a module receives a value outside its responsibility range, the output is clipped.

This means:

```text
the module accepts the intention,
but only keeps the part it can guarantee.
```

## Human-Powered Aircraft Example

Input:

```text
pilot angle = 80 degrees
```

The Pico2W responsibility range is:

```text
[-45, 45]
```

So the signal is clipped before it becomes an executable command.

This is the GBMC idea applied to control:

```text
do not pass intention directly;
translate it into a responsibility-safe transition.
```

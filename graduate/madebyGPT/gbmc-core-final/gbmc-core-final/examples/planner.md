# Planner Layer

This version adds candidate selection.

Previous versions:

```text
v0: Module / Port / ResponsibilityRange / Connection
v1: Flow
v2: Signal / Transition
```

This version:

```text
v3: Candidate / Planner / Responsibility Arrow
```

## Core Idea

```text
candidate distribution
-> run each candidate through the flow
-> measure responsibility violation
-> choose the safest executable candidate
-> responsibility arrow
```

## Score

The demo uses a simple score:

```text
score = total_violation + clipped_count * 10 + blocked * 1000
```

The lower score is better.

## Meaning

A candidate is not chosen because it is closest to intention.

A candidate is chosen because it can pass through the modular system with the least responsibility violation.

This is GBMC as a control-selection system.

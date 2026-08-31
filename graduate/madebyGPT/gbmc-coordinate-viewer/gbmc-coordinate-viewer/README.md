# GBMC Coordinate Viewer

This is the final prototype for this thread.

It converts GBMC from a graph into a coordinate system.

## Axes

```text
x = G/D : Interest ↔ Attractor
y = B/S : Candidate Distribution ↔ Resonance
z = M/Q : Credit/Predictability ↔ Exploration
c = C/C : Trust/Recursion ↔ Freedom
```

Internally, each concept has 4 values.

The viewer displays 3 axes and uses color for the 4th axis.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python gbmc_coordinate.py Groove
python gbmc_coordinate.py Attention
python gbmc_coordinate.py Research
python gbmc_coordinate.py Job
```

## Draw

```bash
python gbmc_coordinate.py Dance --draw
```

This saves:

```text
gbmc_coordinates.png
```

## Meaning

This prototype is meant to answer:

```text
Where am I / where is this concept inside GBMC?
How far is it from Represent:Dance?
What concepts are nearby?
```

## Direction

This can become:

```text
GBMC Coordinate Viewer
-> editable coordinates
-> paper concept auto-placement
-> web/VR visualization
```

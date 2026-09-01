# GBMC Graph Viewer

This is a Python prototype for visualizing where a concept is located inside the GBMC structure.

It answers:

```text
Where is this concept?
What GBMC node is it near?
What nodes are connected around it?
How far is it from the GBMC core?
```

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python gbmc_graph.py Groove
python gbmc_graph.py Attention
python gbmc_graph.py Pipe
```

## Draw

```bash
python gbmc_graph.py Groove --draw
```

This saves:

```text
gbmc_graph.png
```

## Current Concept Groups

```text
GBMC
StreetDance
Transformer
Unix
ControlTheory
HumanPoweredAircraft
```

## Direction

```text
Python graph viewer
-> web app
-> paper concept extraction
-> editable GBMC map
```

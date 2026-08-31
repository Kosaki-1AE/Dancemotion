# Usage

## Install

```bash
pip install -r requirements.txt
```

## Show current position

```bash
python gbmc_graph.py Groove
python gbmc_graph.py Attention
python gbmc_graph.py Pipe
python gbmc_graph.py ServoLimit
```

## Draw graph

```bash
python gbmc_graph.py Groove --draw
```

This creates:

```text
gbmc_graph.png
```

## Use custom JSON

```bash
python gbmc_graph.py Groove --data data/gbmc_graph.json --draw
```

from __future__ import annotations

import argparse
import json
from typing import List

from .core import GBMCCQSDMachine, RoutingMode, TableRouter


def _parse_int(value: str) -> int:
    return int(value, 0)


def _parse_rule(value: str):
    # Format: ROOM:MODE:NEXT_ROOM, e.g. 0x0:Straight:0x1
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("rule must be ROOM:MODE:NEXT_ROOM")
    room_s, mode_s, next_s = parts
    try:
        room = int(room_s, 0)
        next_room = int(next_s, 0)
        mode = RoutingMode(mode_s)
    except (ValueError, KeyError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return room, mode, next_room


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GBMCCQSD 8-bit state-machine simulator")
    parser.add_argument("inputs", nargs="*", type=_parse_int, help="lower nibbles, e.g. 0x0 0x6 0xA 0xC")
    parser.add_argument("--room", type=_parse_int, default=0, help="initial room (0..15)")
    parser.add_argument(
        "--rule",
        action="append",
        default=[],
        type=_parse_rule,
        help="routing rule ROOM:MODE:NEXT_ROOM; may be repeated",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON trace")
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    router = TableRouter()
    for room, mode, next_room in args.rule:
        router.set_rule(room, mode, next_room)

    machine = GBMCCQSDMachine(initial_room=args.room, router=router)
    records = machine.run(args.inputs)

    if args.json:
        print(json.dumps([r.to_dict() for r in records], ensure_ascii=False, indent=2))
        return 0

    if not records:
        print(f"state={machine.state.hex} bits={machine.state.bits}")
        return 0

    for r in records:
        print(
            f"#{r.index:02d} "
            f"room={r.working.room:X} "
            f"lower=0x{r.working.lower_nibble:X} "
            f"resp={r.working.responsibility:02b} "
            f"op={r.working.operation:02b}({r.working.operation_label}) "
            f"mode={r.mode.value:<8} "
            f"-> room={r.after.room:X} "
            f"rule={'hit' if r.routing_rule_hit else 'fallback'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

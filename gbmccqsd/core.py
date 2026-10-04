from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Protocol, Tuple


class RoutingMode(str, Enum):
    """Coarse routing mode derived from the lower 4 bits."""

    OBSERVE = "Observe"
    STRAIGHT = "Straight"
    JAB = "Jab"
    TWIST = "Twist"


OPERATION_LABELS: Dict[int, str] = {
    0b00: "打つ",
    0b01: "弾く",
    0b10: "外す",
    0b11: "置く",
}


@dataclass(frozen=True)
class GBMCCQSDState:
    """8-bit state: [room:4][responsibility:2][operation:2]."""

    room: int
    responsibility: int
    operation: int

    def __post_init__(self) -> None:
        if not 0 <= self.room <= 0xF:
            raise ValueError("room must fit in 4 bits (0..15)")
        if not 0 <= self.responsibility <= 0b11:
            raise ValueError("responsibility must fit in 2 bits (0..3)")
        if not 0 <= self.operation <= 0b11:
            raise ValueError("operation must fit in 2 bits (0..3)")

    @property
    def lower_nibble(self) -> int:
        return (self.responsibility << 2) | self.operation

    @property
    def byte(self) -> int:
        return (self.room << 4) | self.lower_nibble

    @property
    def bits(self) -> str:
        return f"{self.byte:08b}"

    @property
    def hex(self) -> str:
        return f"0x{self.byte:02X}"

    @property
    def operation_label(self) -> str:
        return OPERATION_LABELS[self.operation]

    @classmethod
    def from_byte(cls, value: int) -> "GBMCCQSDState":
        if not 0 <= value <= 0xFF:
            raise ValueError("value must fit in 8 bits (0..255)")
        return cls(
            room=(value >> 4) & 0xF,
            responsibility=(value >> 2) & 0b11,
            operation=value & 0b11,
        )

    @classmethod
    def from_parts(cls, room: int, lower_nibble: int) -> "GBMCCQSDState":
        if not 0 <= lower_nibble <= 0xF:
            raise ValueError("lower_nibble must fit in 4 bits (0..15)")
        return cls(
            room=room,
            responsibility=(lower_nibble >> 2) & 0b11,
            operation=lower_nibble & 0b11,
        )


def classify_mode(responsibility: int, operation: int) -> RoutingMode:
    """Map the 4x4 lower-bit space into Observe/Straight/Jab/Twist.

    Current hypothesis from the design notes:
      responsibility 00/01 + operation 00/01 -> Observe
      responsibility 00/01 + operation 10/11 -> Straight
      responsibility 10/11 + operation 00/01 -> Jab
      responsibility 10/11 + operation 10/11 -> Twist
    """

    if responsibility not in range(4) or operation not in range(4):
        raise ValueError("responsibility and operation must be 2-bit values")

    responsibility_high = responsibility >= 0b10
    operation_high = operation >= 0b10

    if not responsibility_high and not operation_high:
        return RoutingMode.OBSERVE
    if not responsibility_high and operation_high:
        return RoutingMode.STRAIGHT
    if responsibility_high and not operation_high:
        return RoutingMode.JAB
    return RoutingMode.TWIST


@dataclass(frozen=True)
class ExecutionResult:
    """Result of Phi_R(state). Payload stays intentionally open-ended."""

    payload: Any
    note: str = ""


class Executor(Protocol):
    def __call__(self, state: GBMCCQSDState) -> ExecutionResult: ...


class Router(Protocol):
    def resolve(
        self,
        room: int,
        mode: RoutingMode,
        result: ExecutionResult,
    ) -> Tuple[int, bool]: ...


class SymbolicExecutor:
    """Default Phi: preserve theory structure without inventing domain behavior."""

    def __call__(self, state: GBMCCQSDState) -> ExecutionResult:
        return ExecutionResult(
            payload={
                "room": state.room,
                "responsibility": state.responsibility,
                "operation": state.operation,
                "operation_label": state.operation_label,
            },
            note="symbolic execution only",
        )


class TableRouter:
    """Configurable Gamma routing table.

    Keys are (room, RoutingMode). Missing rules keep the current room by default.
    This deliberately avoids hard-coding an unconfirmed theory for Gamma.
    """

    def __init__(
        self,
        rules: Optional[Mapping[Tuple[int, RoutingMode], int]] = None,
        *,
        fallback_to_same_room: bool = True,
    ) -> None:
        self._rules: Dict[Tuple[int, RoutingMode], int] = dict(rules or {})
        self.fallback_to_same_room = fallback_to_same_room
        for (_, _), next_room in self._rules.items():
            if not 0 <= next_room <= 0xF:
                raise ValueError("next_room must fit in 4 bits (0..15)")

    def set_rule(self, room: int, mode: RoutingMode, next_room: int) -> None:
        if not 0 <= room <= 0xF or not 0 <= next_room <= 0xF:
            raise ValueError("room values must fit in 4 bits (0..15)")
        self._rules[(room, mode)] = next_room

    def resolve(
        self,
        room: int,
        mode: RoutingMode,
        result: ExecutionResult,
    ) -> Tuple[int, bool]:
        key = (room, mode)
        if key in self._rules:
            return self._rules[key], True
        if self.fallback_to_same_room:
            return room, False
        raise KeyError(f"No routing rule for room={room:X}, mode={mode.value}")


@dataclass(frozen=True)
class TransitionRecord:
    index: int
    before: GBMCCQSDState
    working: GBMCCQSDState
    mode: RoutingMode
    result: ExecutionResult
    after: GBMCCQSDState
    routing_rule_hit: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "before": {
                **asdict(self.before),
                "byte": self.before.byte,
                "hex": self.before.hex,
                "bits": self.before.bits,
            },
            "working": {
                **asdict(self.working),
                "byte": self.working.byte,
                "hex": self.working.hex,
                "bits": self.working.bits,
            },
            "mode": self.mode.value,
            "result": {
                "payload": self.result.payload,
                "note": self.result.note,
            },
            "after": {
                **asdict(self.after),
                "byte": self.after.byte,
                "hex": self.after.hex,
                "bits": self.after.bits,
            },
            "routing_rule_hit": self.routing_rule_hit,
        }


class GBMCCQSDMachine:
    """Minimal simulator for Room -> lower-4-bit execution -> routing -> next Room."""

    def __init__(
        self,
        *,
        initial_room: int = 0,
        executor: Optional[Executor] = None,
        router: Optional[Router] = None,
    ) -> None:
        self.state = GBMCCQSDState(initial_room, 0, 0)
        self.executor: Executor = executor or SymbolicExecutor()
        self.router: Router = router or TableRouter()
        self.history: List[TransitionRecord] = []

    def step(self, lower_nibble: int) -> TransitionRecord:
        """Run one cycle with the supplied lower 4 bits.

        The current Room is preserved while Phi_R operates. Gamma then resolves
        the next Room. The lower 4 bits are retained in the resulting 8-bit
        state so the trace remains inspectable; a subsequent step supplies the
        next observed lower nibble.
        """

        before = self.state
        working = GBMCCQSDState.from_parts(before.room, lower_nibble)
        mode = classify_mode(working.responsibility, working.operation)
        result = self.executor(working)
        next_room, hit = self.router.resolve(working.room, mode, result)
        after = GBMCCQSDState(
            room=next_room,
            responsibility=working.responsibility,
            operation=working.operation,
        )

        record = TransitionRecord(
            index=len(self.history),
            before=before,
            working=working,
            mode=mode,
            result=result,
            after=after,
            routing_rule_hit=hit,
        )
        self.history.append(record)
        self.state = after
        return record

    def reset(self, room: int = 0) -> None:
        self.state = GBMCCQSDState(room, 0, 0)
        self.history.clear()

    def run(self, lower_nibbles: List[int]) -> List[TransitionRecord]:
        return [self.step(value) for value in lower_nibbles]

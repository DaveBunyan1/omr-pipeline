from dataclasses import dataclass

STAFF_LINES = 5


@dataclass(frozen=True, slots=True)
class Staff:
    line_positions: tuple[float, float, float, float, float]

    def __post_init__(self) -> None:
        if len(self.line_positions) != STAFF_LINES:
            raise ValueError(f"Staff must have exactly {STAFF_LINES} lines.")

        if tuple(sorted(self.line_positions)) != self.line_positions:
            raise ValueError("Staff lines must be ordered from top to bottom.")

        if len(set(self.line_positions)) != 5:
            raise ValueError("Staff lines must have distinct positions.")

    @property
    def top(self) -> float:
        return self.line_positions[0]

    @property
    def bottom(self) -> float:
        return self.line_positions[-1]

    @property
    def spacing(self) -> float:
        return (self.line_positions[-1] - self.line_positions[0]) / (STAFF_LINES - 1)

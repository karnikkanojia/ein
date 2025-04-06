from typing import List, Union, Any, TypeVar, Optional
from ein.errors.exceptions import ValidationError

NodeType = Union['AxisNode', 'MergeNode', 'SplitNode', 'EllipsisNode', 'AnonymousAxis']


class AxisNode:
    """Represents a single named axis."""

    def __init__(self, name: str) -> None:
        self.name: str = name

    def __repr__(self) -> str:
        return f"AxisNode({self.name})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, AxisNode):
            return False
        return self.name == other.name


class MergeNode:
    """Represents merging of multiple axes (e.g., (h w) -> hw)."""

    def __init__(self, axes: List[NodeType]) -> None:
        self.axes: List[NodeType] = axes

    def __repr__(self) -> str:
        return f"MergeNode({self.axes})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MergeNode):
            return False
        return self.axes == other.axes

    def __getitem__(self, idx: int) -> NodeType:
        """Make MergeNode subscriptable."""
        return self.axes[idx]

    def __len__(self) -> int:
        """Return the number of axes."""
        return len(self.axes)


class SplitNode:
    """Represents splitting into multiple axes (e.g., (h1 2) -> h1, 2)."""

    def __init__(self, axes: List[NodeType]) -> None:
        self.axes: List[NodeType] = axes  # Can be named axes or fixed integers

    def __repr__(self) -> str:
        return f"SplitNode({self.axes})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SplitNode):
            return False
        return self.axes == other.axes

    def __getitem__(self, idx: int) -> NodeType:
        """Make SplitNode subscriptable."""
        return self.axes[idx]

    def __len__(self) -> int:
        """Return the number of axes."""
        return len(self.axes)


class EllipsisNode:
    """Represents an ellipsis '...' in the pattern."""

    def __repr__(self) -> str:
        return "EllipsisNode()"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, EllipsisNode)


class AnonymousAxis:
    """Instances of this class are not equal to each other."""

    def __init__(self, value: str) -> None:
        self.value: int = int(value)
        if self.value <= 1:
            if self.value == 1:
                raise ValidationError(
                    "No need to create an anonymous axis of length 1."
                )
            else:
                raise ValidationError(
                    f"Anonymous axis should have positive length, not {self.value}"
                )

    def __repr__(self) -> str:
        return f"AnonymousAxis({self.value})"


class AnonymousAxisPlaceholder:
    def __init__(self, value: int) -> None:
        self.value: int = value
        assert isinstance(self.value, int)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, AnonymousAxis) and self.value == other.value

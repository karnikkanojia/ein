from typing import List, Union
from ein.errors.exceptions import ValidationError


class AxisNode:
    """Represents a single named axis."""

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"AxisNode({self.name})"

    def __eq__(self, other):
        if not isinstance(other, AxisNode):
            return False
        return self.name == other.name


class MergeNode:
    """Represents merging of multiple axes (e.g., (h w) -> hw)."""

    def __init__(self, axes: List[Union["AxisNode", "SplitNode"]]):
        self.axes = axes

    def __repr__(self):
        return f"MergeNode({self.axes})"

    def __eq__(self, other):
        if not isinstance(other, MergeNode):
            return False
        return self.axes == other.axes

    def __getitem__(self, idx):
        """Make MergeNode subscriptable."""
        return self.axes[idx]

    def __len__(self):
        """Return the number of axes."""
        return len(self.axes)


class SplitNode:
    """Represents splitting into multiple axes (e.g., (h1 2) -> h1, 2)."""

    def __init__(self, axes: List[Union[str, int]]):
        self.axes = axes  # Can be named axes or fixed integers

    def __repr__(self):
        return f"SplitNode({self.axes})"

    def __eq__(self, other):
        if not isinstance(other, SplitNode):
            return False
        return self.axes == other.axes

    def __getitem__(self, idx):
        """Make SplitNode subscriptable."""
        return self.axes[idx]

    def __len__(self):
        """Return the number of axes."""
        return len(self.axes)


class EllipsisNode:
    """Represents an ellipsis '...' in the pattern."""

    def __repr__(self):
        return "EllipsisNode()"

    def __eq__(self, other):
        return isinstance(other, EllipsisNode)


class AnonymousAxis(object):
    """Instances of this class are not equal to each other"""

    def __init__(self, value: str):
        self.value = int(value)
        if self.value <= 1:
            if self.value == 1:
                raise ValidationError(
                    "No need to create an anonymous axis of length 1."
                )
            else:
                raise ValidationError(
                    f"Anonymous axis should have positive length, not {self.value}"
                )

    def __repr__(self):
        return f"AnonymousAxis({self.value})"


class AnonymousAxisPlaceholder:
    def __init__(self, value: int):
        self.value = value
        assert isinstance(self.value, int)

    def __eq__(self, other):
        return isinstance(other, AnonymousAxis) and self.value == other.value

from typing import Set, List, Any, Union
from ein.nodes import AxisNode, MergeNode, SplitNode, AnonymousAxis, NodeType


def get_named_axes(nodes: List[NodeType]) -> Set[str]:
    """
    Get all named axes from an expression tree.

    Args:
        nodes: List of nodes to extract axis names from

    Returns:
        Set of all unique axis names found in the tree
    """
    result: Set[str] = set()

    def extract_names(node: NodeType) -> None:
        if isinstance(node, AxisNode):
            result.add(node.name)
        elif isinstance(node, (MergeNode, SplitNode)):
            for subnode in node.axes:
                extract_names(subnode)

    for node in nodes:
        extract_names(node)

    return result


def has_anonymous_axis(nodes: List[NodeType]) -> bool:
    """
    Check if the expression tree contains any anonymous axes.

    Args:
        nodes: List of nodes to check

    Returns:
        True if any anonymous axes are found, False otherwise
    """

    def _walk(node: NodeType) -> bool:
        if isinstance(node, AnonymousAxis):
            return True
        elif isinstance(node, (MergeNode, SplitNode)):
            return any(_walk(sub) for sub in node.axes)
        return False

    return any(_walk(n) for n in nodes)


def get_duplicate_axes(nodes: List[NodeType]) -> Set[str]:
    """
    Find duplicate axis names in the expression tree.

    Args:
        nodes: List of nodes to check for duplicates

    Returns:
        Set of axis names that appear multiple times
    """
    seen: Set[str] = set()
    duplicates: Set[str] = set()

    def _walk(node: NodeType) -> None:
        if isinstance(node, AxisNode):
            if node.name in seen:
                duplicates.add(node.name)
            seen.add(node.name)
        elif isinstance(node, (MergeNode, SplitNode)):
            for sub in node.axes:
                _walk(sub)

    for node in nodes:
        _walk(node)
    return duplicates

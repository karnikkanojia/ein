from ein.nodes import AxisNode, MergeNode, SplitNode, EllipsisNode, AnonymousAxis

def get_named_axes(nodes):
    axes = set()

    def _walk(node):
        if isinstance(node, AxisNode):
            axes.add(node.name)
        elif isinstance(node, (MergeNode, SplitNode)):
            for sub in node.axes:
                _walk(sub)

    for node in nodes:
        _walk(node)
    return axes


def has_anonymous_axis(nodes):
    def _walk(node):
        if isinstance(node, AnonymousAxis):
            return True
        elif isinstance(node, (MergeNode, SplitNode)):
            return any(_walk(sub) for sub in node.axes)
        return False

    return any(_walk(n) for n in nodes)


def get_duplicate_axes(nodes):
    seen = set()
    duplicates = set()

    def _walk(node):
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


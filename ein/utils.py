from ein.nodes import AxisNode, MergeNode, SplitNode, EllipsisNode, AnonymousAxis

def get_named_axes(nodes):
    """Get all named axes from an expression tree."""
    result = set()
    
    def extract_names(node):
        if isinstance(node, AxisNode):
            result.add(node.name)
        elif isinstance(node, (MergeNode, SplitNode)):
            for subnode in node.axes:
                extract_names(subnode)
    
    for node in nodes:
        extract_names(node)
    
    return result


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


from ein.nodes import (
    AxisNode,
    MergeNode,
    SplitNode,
    EllipsisNode,
    AnonymousAxis,
    AnonymousAxisPlaceholder,
    NodeType,
)
from typing import List, Union, Dict, Any, Optional, Set
import re
import keyword
from ein.errors.exceptions import ValidationError
from ein.utils import get_named_axes, get_duplicate_axes, has_anonymous_axis


class ParsedExpression:
    """Parses an einops-style pattern into an expression tree."""

    def __init__(self, expression: str, is_input_pattern: bool = True, known_axes: Optional[Dict[str, int]] = None) -> None:
        self.expression: str = expression
        self.is_input_pattern: bool = is_input_pattern
        self.known_axes: Dict[str, int] = known_axes or {}
        self.nodes: List[NodeType] = self._parse(expression, is_input_pattern, self.known_axes)
        self._validate_tree()

    def _parse(
        self, expression: str, is_input_pattern: bool = True, known_axes: Optional[Dict[str, int]] = None
    ) -> List[NodeType]:
        """
        Parse an expression string into a tree of nodes.

        Args:
            expression: String expression to parse
            is_input_pattern: Whether this is an input pattern (affects split/merge detection)
            known_axes: Dictionary of axis names that are explicitly specified in kwargs

        Returns:
            List of parsed nodes forming the expression tree
        """
        tokens = self._tokenize(expression)
        return self._build_tree(tokens, is_input_pattern, known_axes)

    def _tokenize(self, expression: str) -> List[str]:
        """
        Splits the input string into meaningful tokens.

        Args:
            expression: String expression to tokenize

        Returns:
            List of tokens

        Raises:
            ValidationError: If invalid token patterns are found
        """
        # Replace ellipsis with unicode character for easier handling
        expression = expression.replace("...", "…")

        # First check for invalid token patterns that should cause errors
        invalid_patterns: List[str] = [
            r"…\w+",  # Ellipsis immediately followed by text (no space)
            r"\w+…",  # Text immediately followed by ellipsis (no space)
        ]

        for pattern in invalid_patterns:
            if re.search(pattern, expression):
                match = re.search(pattern, expression)
                if match:  # For type safety
                    matched_str = match.group(0)
                    # Convert back to original format for error message
                    original = matched_str.replace("…", "...")
                    raise ValidationError(f"Invalid token: '{original}'")

        # Now tokenize valid patterns
        expression_list: List[str] = re.findall(r"\w+|\(|\)|…", expression)
        return expression_list

    def _build_tree(
        self, tokens: List[str], is_input_pattern: bool = True, known_axes: Optional[Dict[str, int]] = None
    ) -> List[NodeType]:
        """
        Converts tokens into an expression tree.

        Args:
            tokens: List of tokens to parse
            is_input_pattern: Whether this is an input pattern (affects split/merge detection)
            known_axes: Dictionary of axis names that are explicitly specified in kwargs
            
        Returns:
            List of nodes forming the expression tree
            
        Raises:
            ValidationError: If the token structure is invalid
        """
        if known_axes is None:
            known_axes = {}

        stack: List[List[NodeType]] = []
        current_group: List[NodeType] = []

        for token in tokens:
            if token == "(":
                stack.append(current_group)
                current_group = []
            elif token == ")":
                if not stack:
                    raise ValidationError("Unmatched closing parenthesis in pattern.")
                last_group = stack.pop()

                # Special handling for input pattern with known explicit axes
                if is_input_pattern and len(current_group) > 1:
                    # Check if any axes in the group are in known_axes
                    axis_names: List[str] = [
                        node.name
                        for node in current_group
                        if isinstance(node, AxisNode)
                    ]
                    is_split: bool = any(name in known_axes for name in axis_names)

                    if is_split:
                        # We found axes that match our explicit kwargs, treat as SplitNode
                        last_group.append(SplitNode(current_group))
                    elif any(isinstance(x, AnonymousAxis) for x in current_group):
                        # Traditional detection via anonymous axes
                        last_group.append(SplitNode(current_group))
                    else:
                        # Default to MergeNode
                        last_group.append(MergeNode(current_group))
                else:
                    # Existing logic for simple cases and output patterns
                    if len(current_group) == 1:
                        last_group.append(
                            current_group[0]
                        )  # Single axis, no merge/split
                    elif any(isinstance(x, AnonymousAxis) for x in current_group):
                        last_group.append(SplitNode(current_group))
                    else:
                        last_group.append(MergeNode(current_group))

                current_group = last_group
            elif token == "…":
                current_group.append(EllipsisNode())
            elif token.isdigit():
                # Special handling for '1' as a literal rather than anonymous axis
                if token == '1' and is_input_pattern:
                    current_group.append(AxisNode(token))  # Treat as named axis
                else:
                    current_group.append(AnonymousAxis(token))
            else:
                current_group.append(AxisNode(token))

        if stack:
            raise ValidationError("Unmatched opening parenthesis in pattern.")

        return current_group

    def _validate_tree(self) -> None:
        """
        Ensures the parsed expression follows einops rules.
        
        Raises:
            ValidationError: If validation fails
        """
        # First check for empty expression
        if not self.nodes:
            raise ValidationError("Expression cannot be empty.")

        seen_axes: Set[str] = set()
        ellipsis_count: int = 0

        # Store a reference to known_axes that was passed during initialization
        known_axes: Dict[str, int] = getattr(self, "known_axes", {})
        is_input_pattern: bool = getattr(self, "is_input_pattern", True)

        def is_valid_identifier(ident: Any) -> bool:
            """
            Determines if string is valid Python identifier and not a keyword.
            
            Args:
                ident: Value to check
                
            Returns:
                True if valid identifier, False otherwise
            """
            if not isinstance(ident, str):
                return False

            # Special case: Allow '1' as valid identifier for singleton dimensions
            if ident == '1':
                return True
                
            # Check for empty string
            if not ident:
                return False

            # Check if it's a valid Python identifier
            if not ident.isidentifier():
                return False

            # Check if it's not a keyword
            if keyword.iskeyword(ident):
                return False

            # Check for leading/trailing underscores
            if ident.startswith("_") or ident.endswith("_"):
                return False

            return True

        def validate_node(node: NodeType) -> None:
            """
            Validates a single node in the expression tree.
            
            Args:
                node: Node to validate
                
            Raises:
                ValidationError: If node validation fails
            """
            nonlocal ellipsis_count

            if isinstance(node, AxisNode):
                # Check for duplicate axis names
                if node.name in seen_axes:
                    raise ValidationError(f"Duplicate axis name found: '{node.name}'")
                seen_axes.add(node.name)

                # Validate axis name using our helper function
                if not is_valid_identifier(node.name):
                    raise ValidationError(
                        f"Invalid axis name: '{node.name}'. "
                        f"Must be a valid Python identifier, "
                        f"not a keyword, and not start/end with underscore."
                    )

            elif isinstance(node, MergeNode):
                # Merging should only contain named axes (no ellipsis)
                for subnode in node.axes:
                    if isinstance(subnode, AnonymousAxis):
                        raise ValidationError(
                            f"Cannot merge anonymous axis {subnode} inside parentheses."
                        )
                    if isinstance(subnode, EllipsisNode):
                        raise ValidationError(
                            "Ellipsis cannot appear inside parentheses."
                        )
                    validate_node(subnode)

            elif isinstance(node, SplitNode):
                # For input patterns, we allow SplitNodes without AnonymousAxis if the axes
                # are listed in known_axes
                if is_input_pattern:
                    # Check if any axes in the SplitNode are in known_axes
                    axis_names: List[str] = [
                        subnode.name
                        for subnode in node.axes
                        if isinstance(subnode, AxisNode)
                    ]
                    has_known_axes: bool = any(name in known_axes for name in axis_names)

                    # Only enforce the AnonymousAxis requirement if we don't have known axes
                    if not has_known_axes:
                        has_anonymous: bool = any(
                            isinstance(subnode, AnonymousAxis) for subnode in node.axes
                        )
                        if not has_anonymous:
                            raise ValidationError(
                                f"Splitting requires at least one numeric size or explicit size parameter: {node}"
                            )
                else:
                    # For output patterns, we still require at least one AnonymousAxis
                    has_anonymous: bool = any(
                        isinstance(subnode, AnonymousAxis) for subnode in node.axes
                    )
                    if not has_anonymous:
                        raise ValidationError(
                            f"Splitting requires at least one numeric size: {node}"
                        )

                for subnode in node.axes:
                    if isinstance(subnode, EllipsisNode):
                        raise ValidationError(
                            "Ellipsis cannot appear inside parentheses."
                        )
                    if not isinstance(
                        subnode, AnonymousAxis
                    ):  # Only validate non-anonymous nodes
                        validate_node(subnode)

            elif isinstance(node, EllipsisNode):
                ellipsis_count += 1
                if ellipsis_count > 1:
                    raise ValidationError(
                        "Expression cannot contain more than one ellipsis ('...')."
                    )

        # Validate the entire tree
        for node in self.nodes:
            validate_node(node)

    def __repr__(self) -> str:
        return f"ParsedExpression({self.nodes})"


def validate_pair(input_expr: ParsedExpression, output_expr: ParsedExpression) -> None:
    """
    Validate that the output expression is compatible with the input expression.
    
    Args:
        input_expr: Input expression
        output_expr: Output expression
        
    Raises:
        ValidationError: If validation fails
    """
    input_axes: Set[str] = get_named_axes(input_expr.nodes)
    output_axes: Set[str] = get_named_axes(output_expr.nodes)

    # Rule 1: Named output axes must exist in input
    for axis in output_axes:
        if axis not in input_axes and axis not in getattr(input_expr, 'known_axes', {}):
            raise ValidationError(f"Output axis '{axis}' not found in input.")

    # Rule 2: No duplicates
    input_dups: Set[str] = get_duplicate_axes(input_expr.nodes)
    output_dups: Set[str] = get_duplicate_axes(output_expr.nodes)
    if input_dups:
        raise ValidationError(f"Duplicate axes in input: {input_dups}")
    if output_dups:
        raise ValidationError(f"Duplicate axes in output: {output_dups}")

    # Rule 3: Anonymous axes not allowed in output without named binding
    # Get the known_axes from the input expression
    known_axes: Dict[str, int] = getattr(input_expr, 'known_axes', {})
    
    # Check if any anonymous axes exist in the output
    has_anon: bool = has_anonymous_axis(output_expr.nodes)
    
    # If we have anonymous axes in output, ensure we have corresponding known_axes
    if has_anon and not known_axes:
        raise ValidationError("Anonymous axis in output pattern must be named using kwargs.")


def flatten_expr(nodes: List[NodeType]) -> List[Union[AxisNode, EllipsisNode, AnonymousAxis, AnonymousAxisPlaceholder]]:
    """
    Flattens a list of parsed expression nodes into a linear axis sequence.
    Used to align with actual tensor dimensions.
    
    Args:
        nodes: List of nodes to flatten
        
    Returns:
        Flattened list of nodes
        
    Raises:
        TypeError: If an unsupported node type is encountered
    """
    result: List[Union[AxisNode, EllipsisNode, AnonymousAxis, AnonymousAxisPlaceholder]] = []

    for node in nodes:
        if isinstance(node, (AxisNode, AnonymousAxis, EllipsisNode)):
            result.append(node)
        elif isinstance(node, (MergeNode, SplitNode)):
            result.extend(flatten_expr(node.axes))
        else:
            raise TypeError(f"Unsupported node type: {type(node)}")

    return result


def flatten_expr_for_dim_count(nodes: List[NodeType]) -> List[Union[AxisNode, AnonymousAxis, EllipsisNode]]:
    """
    Flattens an expression tree to determine how many actual dimensions it represents.
    This is different from regular flatten_expr as it treats SplitNode as a single dimension.
    
    Args:
        nodes: List of nodes to flatten
        
    Returns:
        Flattened list of nodes counting dimensions
        
    Raises:
        TypeError: If an unsupported node type is encountered
    """
    result: List[Union[AxisNode, AnonymousAxis, EllipsisNode]] = []

    for node in nodes:
        if isinstance(node, (AxisNode, AnonymousAxis, EllipsisNode)):
            result.append(node)
        elif isinstance(node, MergeNode):
            result.extend(flatten_expr_for_dim_count(node.axes))
        elif isinstance(node, SplitNode):
            # For dimension counting purposes, a SplitNode is a single dimension
            # Find the first AxisNode to represent it
            for subnode in node.axes:
                if isinstance(subnode, AxisNode):
                    result.append(subnode)
                    break
            else:
                # If no AxisNode, append something to represent the dimension
                result.append(AnonymousAxis("2"))  # Default placeholder
        else:
            raise TypeError(f"Unsupported node type: {type(node)}")

    return result


def infer_axis_sizes(
    flat_input_seq: List[Union[AxisNode, EllipsisNode, AnonymousAxis, AnonymousAxisPlaceholder]], 
    input_shape: List[int], 
    axis_sizes: Optional[Dict[str, int]] = None
) -> Dict[str, int]:
    """
    Given the flattened input expression and actual tensor shape, return a dictionary
    mapping axis names to dimensions.
    
    Args:
        flat_input_seq: Flattened sequence of nodes
        input_shape: Shape of the input tensor
        axis_sizes: Optional dictionary of known axis sizes
        
    Returns:
        Dictionary mapping axis names to their sizes
        
    Raises:
        ValidationError: If dimensions don't match or other validation fails
    """
    if axis_sizes is None:
        axis_sizes = {}
    else:
        axis_sizes = axis_sizes.copy()  # Don't modify the input

    shape_dict: Dict[str, int] = {}
    input_index: int = 0

    for node in flat_input_seq:
        if isinstance(node, AxisNode):
            if input_index >= len(input_shape):
                raise ValidationError(
                    f"Not enough dimensions in input shape to bind axis '{node.name}'"
                )
            shape_dict[node.name] = input_shape[input_index]
            input_index += 1
        elif isinstance(node, EllipsisNode):
            # Handle ellipsis by consuming the remaining dimensions
            num_remaining: int = (
                len(input_shape)
                - input_index
                - (len(flat_input_seq) - flat_input_seq.index(node) - 1)
            )
            if num_remaining < 0:
                raise ValidationError(
                    "Not enough dimensions in input shape to match ellipsis."
                )
            for i in range(num_remaining):
                shape_dict[f"...{i}"] = input_shape[input_index]
                input_index += 1
        elif isinstance(node, AnonymousAxis):
            if input_index >= len(input_shape):
                raise ValidationError(
                    "Not enough dimensions in input shape to bind anonymous axis."
                )
            input_index += 1  # Consume one dimension for the anonymous axis

    if input_index != len(input_shape):
        raise ValidationError("Input shape does not match number of axes in pattern.")

    # Add any explicitly provided sizes
    shape_dict.update(axis_sizes)

    return shape_dict

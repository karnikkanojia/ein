import pytest
from ein.parsing import ParsedExpression, validate_pair, flatten_expr, infer_axis_sizes
from ein.nodes import (
    AnonymousAxis,
    AnonymousAxisPlaceholder,
    AxisNode,
    EllipsisNode,
    MergeNode,
    SplitNode,
)
from ein.errors.exceptions import ValidationError


def test_anonymous_axes():
    a, b = AnonymousAxis("2"), AnonymousAxis("2")
    assert a != b, "AnonymousAxis instances with the same value should not be equal"
    c, d = AnonymousAxisPlaceholder(2), AnonymousAxisPlaceholder(3)
    assert (
        a == c and b == c
    ), "AnonymousAxis should match AnonymousAxisPlaceholder with the same value"
    assert (
        a != d and b != d
    ), "AnonymousAxis should not match AnonymousAxisPlaceholder with a different value"
    assert [a, 2, b] == [c, 2, c], "AnonymousAxis should behave correctly in lists"


@pytest.mark.parametrize(
    "valid_name",
    [
        "a",
        "b",
        "h",
        "dx",
        "h1",
        "zz",
        "i9123",
        "somelongname",
        "Alex",
        "camelCase",
        "u_n_d_e_r_score",
        "unreasonablyLongAxisName",
    ],
)
def test_valid_axis_names(valid_name):
    # This should not raise an exception for valid axis names
    ParsedExpression(valid_name)


@pytest.mark.parametrize(
    "invalid_name",
    ["", "2b", "_ startWithUnderscore", "endWithUnderscore_", "_"],
)
def test_invalid_axis_names(invalid_name):
    with pytest.raises((ValueError, ValidationError)):
        ParsedExpression(invalid_name)


@pytest.mark.parametrize(
    "expression, should_raise",
    [
        ("... a b c d", False),
        ("... a b c d ...", True),
        ("... a b c (d ...)", True),
        ("(... a) b c (d ...)", True),
        ("(a)) b c (d ...)", True),
        ("(a b c (d ...)", True),
        ("(a) (()) b c (d ...)", True),
        ("(a) ((b c) (d ...))", True),
        ("camelCase under_scored cApiTaLs ß ...", False),
        ("1a", True),
        ("_pre", True),
        ("...pre", True),
        ("pre...", True),
    ],
)
def test_parsed_expression_exceptions(expression, should_raise):
    if should_raise:
        with pytest.raises((ValueError, ValidationError)):
            ParsedExpression(expression)
    else:
        ParsedExpression(expression)


def test_parse_expression():
    parsed = ParsedExpression("a b c d")
    assert parsed.nodes == [AxisNode("a"), AxisNode("b"), AxisNode("c"), AxisNode("d")]

    parsed = ParsedExpression("a ... b")
    assert parsed.nodes == [AxisNode("a"), EllipsisNode(), AxisNode("b")]

    parsed = ParsedExpression("(a b c) ... d")
    assert parsed.nodes == [
        MergeNode([AxisNode("a"), AxisNode("b"), AxisNode("c")]),
        EllipsisNode(),
        AxisNode("d"),
    ]


@pytest.mark.parametrize(
    "input_expr, output_expr, known_axes",
    [
        ("b h w c", "b c h w", {}),
        ("b h w", "b (h w)", {}),
        ("b ... c", "b c ...", {}),
        ("b (h h1) (w w1)", "b h w h1 w1", {"h1": 2, "w1": 3}),
        ("(b h) w", "b h w", {}),
        ("... h w", "h ... w", {}),
        ("b (h h1) w", "b h h1 w", {"h1": 2}),
        ("(b h) w", "b h w", {"b": 2, "h": 2}),
        # Add split node test cases with known_axes
        ("a (b c)", "a b c", {"b": 3, "c": 4}),
        ("batch (height width)", "batch height width", {"height": 224, "width": 224}),
    ],
)
def test_valid_pairs(input_expr, output_expr, known_axes):
    input_tree = ParsedExpression(
        input_expr, is_input_pattern=True, known_axes=known_axes
    )
    output_tree = ParsedExpression(output_expr, is_input_pattern=False)
    validate_pair(input_tree, output_tree)  # should not raise


@pytest.mark.parametrize(
    "input_expr, output_expr",
    [
        ("b h h", "b h"),
        ("b 2 h", "b h 2"),
        ("(b 2) h", "b 2 h"),
        ("(... b) h", "b h"),
        ("b 1x h", "b h"),
        ("b ... h ...", "b h"),
        ("b h", "b h x"),
        ("b ... h", "b h x"),
        ("b h", "b c d"),
        ("b h", "b 1 h"),
        ("b h", "b two h"),
        ("", "b h"),
        ("(b h", "b h"),
        ("b h)", "b h"),
        ("b (h 2)", "b h 2"),
        ("b h", "b 2 h"),
    ],
)
def test_invalid_pairs(input_expr, output_expr):
    with pytest.raises(ValidationError):
        input_tree = ParsedExpression(input_expr)
        output_tree = ParsedExpression(output_expr)
        validate_pair(input_tree, output_tree)


def test_flatten_expr_simple():
    """Test flattening of simple expressions."""
    # Simple list of axis nodes
    nodes = [AxisNode("a"), AxisNode("b"), AxisNode("c")]
    assert flatten_expr(nodes) == nodes


def test_flatten_expr_with_merge():
    """Test flattening expressions with merge nodes."""
    # Expression with merge: a (b c) d
    nodes = [
        AxisNode("a"),
        MergeNode([AxisNode("b"), AxisNode("c")]),
        AxisNode("d"),
    ]
    expected = [AxisNode("a"), AxisNode("b"), AxisNode("c"), AxisNode("d")]
    assert flatten_expr(nodes) == expected


def test_flatten_expr_with_split():
    """Test flattening expressions with split nodes."""
    # Expression with split: a (b 2) d
    nodes = [
        AxisNode("a"),
        SplitNode([AxisNode("b"), AnonymousAxis("2")]),
        AxisNode("d"),
    ]
    expected = [
        AxisNode("a"),
        AxisNode("b"),
        AnonymousAxisPlaceholder(2),
        AxisNode("d"),
    ]
    assert flatten_expr(nodes) == expected


def test_flatten_expr_nested():
    """Test flattening of nested expressions."""
    # Expression with nested structures: (a (b c)) (d (e 2))
    nodes = [
        MergeNode([AxisNode("a"), MergeNode([AxisNode("b"), AxisNode("c")])]),
        SplitNode([AxisNode("d"), SplitNode([AxisNode("e"), AnonymousAxis("2")])]),
    ]
    expected = [
        AxisNode("a"),
        AxisNode("b"),
        AxisNode("c"),
        AxisNode("d"),
        AxisNode("e"),
        AnonymousAxisPlaceholder(2),
    ]
    assert flatten_expr(nodes) == expected


def test_flatten_expr_with_ellipsis():
    """Test flattening expressions with ellipsis."""
    # Expression with ellipsis: a ... b
    nodes = [AxisNode("a"), EllipsisNode(), AxisNode("b")]
    expected = [AxisNode("a"), EllipsisNode(), AxisNode("b")]
    assert flatten_expr(nodes) == expected


def test_flatten_expr_complex():
    """Test flattening of complex expressions."""
    # Complex expression: a (b (c 2)) ... (d e)
    nodes = [
        AxisNode("a"),
        SplitNode([AxisNode("b"), SplitNode([AxisNode("c"), AnonymousAxis("2")])]),
        EllipsisNode(),
        MergeNode([AxisNode("d"), AxisNode("e")]),
    ]
    expected = [
        AxisNode("a"),
        AxisNode("b"),
        AxisNode("c"),
        AnonymousAxisPlaceholder(2),
        EllipsisNode(),
        AxisNode("d"),
        AxisNode("e"),
    ]
    assert flatten_expr(nodes) == expected


def test_flatten_expr_invalid_node():
    """Test flattening with invalid node type."""
    # Invalid node type
    with pytest.raises(TypeError):
        flatten_expr([AxisNode("a"), "not_a_node", AxisNode("b")])


def test_infer_axis_sizes_simple():
    """Test inferring axis sizes for simple expressions."""
    flat_seq = [AxisNode("a"), AxisNode("b"), AxisNode("c")]
    input_shape = [2, 3, 4]
    expected = {"a": 2, "b": 3, "c": 4}
    assert infer_axis_sizes(flat_seq, input_shape) == expected


def test_infer_axis_sizes_with_ellipsis():
    """Test inferring axis sizes with ellipsis."""
    flat_seq = [AxisNode("a"), EllipsisNode(), AxisNode("b")]
    input_shape = [2, 3, 4, 5]
    expected = {"a": 2, "...0": 3, "...1": 4, "b": 5}
    assert infer_axis_sizes(flat_seq, input_shape) == expected


def test_infer_axis_sizes_multiple_ellipsis():
    """Test handling of multiple ellipsis (shouldn't happen in valid input)."""
    flat_seq = [AxisNode("a"), EllipsisNode(), AxisNode("b"), EllipsisNode()]
    input_shape = [2, 3, 4, 5, 6]
    with pytest.raises(ValidationError):
        infer_axis_sizes(flat_seq, input_shape)


def test_infer_axis_sizes_with_anonymous_axis():
    """Test inferring axis sizes with anonymous axes."""
    flat_seq = [AxisNode("a"), AnonymousAxis("2"), AxisNode("b")]
    input_shape = [2, 3, 4]
    expected = {"a": 2, "b": 4}  # AnonymousAxis should be skipped
    assert infer_axis_sizes(flat_seq, input_shape) == expected


def test_infer_axis_sizes_insufficient_dims():
    """Test when there are insufficient dimensions in input shape."""
    flat_seq = [AxisNode("a"), AxisNode("b"), AxisNode("c")]
    input_shape = [2, 3]
    with pytest.raises(ValidationError):
        infer_axis_sizes(flat_seq, input_shape)


def test_infer_axis_sizes_too_many_dims():
    """Test when there are too many dimensions in input shape."""
    flat_seq = [AxisNode("a"), AxisNode("b")]
    input_shape = [2, 3, 4]
    with pytest.raises(ValidationError):
        infer_axis_sizes(flat_seq, input_shape)


def test_infer_axis_sizes_complex():
    """Test inferring axis sizes for complex expressions."""
    flat_seq = [
        AxisNode("batch"),
        AxisNode("height"),
        AxisNode("width"),
        EllipsisNode(),
        AxisNode("channels"),
    ]
    input_shape = [32, 224, 224, 3, 3, 3]
    expected = {
        "batch": 32,
        "height": 224,
        "width": 224,
        "...0": 3,
        "...1": 3,
        "channels": 3,
    }
    assert infer_axis_sizes(flat_seq, input_shape) == expected

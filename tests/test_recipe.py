import pytest
import numpy as np
from ein.recipe import rearrange, RecipeBuilder, Recipe
from ein.parsing import ParsedExpression
from ein.errors.exceptions import ValidationError


def test_simple_transpose():
    """Test a simple axis transpose."""
    tensor = np.arange(24).reshape(2, 3, 4)
    result = rearrange(tensor, "a b c -> a c b")
    expected = np.transpose(tensor, (0, 2, 1))
    np.testing.assert_array_equal(result, expected)


def test_merge_axes():
    """Test merging of axes."""
    tensor = np.arange(24).reshape(2, 3, 4)
    result = rearrange(tensor, "a b c -> a (b c)")
    expected = tensor.reshape(2, 12)
    np.testing.assert_array_equal(result, expected)


def test_split_axes():
    """Test splitting of axes."""
    tensor = np.arange(24).reshape(2, 12)
    result = rearrange(tensor, "a (b c) -> a b c", b=3, c=4)
    expected = tensor.reshape(2, 3, 4)
    np.testing.assert_array_equal(result, expected)


def test_complex_rearrangement():
    """Test a more complex rearrangement with multiple operations."""
    tensor = np.arange(30).reshape(2, 3, 5)
    result = rearrange(tensor, "a b c -> c (a b)")
    expected = np.transpose(tensor, (2, 0, 1)).reshape(5, 6)
    np.testing.assert_array_equal(result, expected)


def test_ellipsis():
    """Test patterns with ellipsis."""
    tensor = np.arange(120).reshape(2, 3, 4, 5)
    result = rearrange(tensor, "a ... b -> b ... a")
    expected = np.transpose(tensor, (3, 1, 2, 0))
    np.testing.assert_array_equal(result, expected)


def test_validation_errors():
    """Test that appropriate validation errors are raised."""
    tensor = np.arange(24).reshape(2, 3, 4)

    with pytest.raises(ValidationError):
        rearrange(tensor, "a b c -> a b d")  # 'd' not defined

    with pytest.raises(ValidationError):
        rearrange(tensor, "a b -> a b c")  # Output has more dimensions

    with pytest.raises(ValueError):
        rearrange(tensor, "a b c")  # Missing arrow


def test_recipe_builder():
    """Test the RecipeBuilder class directly."""
    input_expr = ParsedExpression("a b c")
    output_expr = ParsedExpression("c a b")
    shape = (2, 3, 4)

    builder = RecipeBuilder(input_expr, output_expr, shape)
    recipe = builder.build()

    assert isinstance(recipe, Recipe)
    assert recipe.input_shape == shape
    assert recipe.output_shape == (4, 2, 3)


def test_repeating_axes():
    """Test repeating singleton dimensions."""
    # Single repeating axis
    tensor = np.random.rand(3, 1, 5)
    result = rearrange(tensor, 'a 1 c -> a b c', b=4)
    assert result.shape == (3, 4, 5)
    for i in range(4):
        np.testing.assert_array_equal(result[:, i, :], tensor[:, 0, :])

    # Multiple repeating axes
    tensor = np.random.rand(2, 1, 3, 1)
    result = rearrange(tensor, 'a 1 c 1 -> a b c d', b=2, d=3)
    assert result.shape == (2, 2, 3, 3)
    for i in range(2):
        for j in range(3):
            np.testing.assert_array_equal(result[:, i, :, j], tensor[:, 0, :, 0])

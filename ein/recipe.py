from ein.parsing import ParsedExpression, flatten_expr, infer_axis_sizes, flatten_expr_for_dim_count
from ein.errors.exceptions import ValidationError
from ein.nodes import (
    AxisNode,
    MergeNode,
    SplitNode,
    EllipsisNode,
    AnonymousAxis,
)
from typing import List, Tuple, Dict, Union
import numpy as np


class Operation:
    """Base class for transformation operations."""
    def __init__(self):
        pass

    def apply(self, tensor: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement apply")

    def __repr__(self):
        return self.__class__.__name__


class Reshape(Operation):
    """Reshape operation that changes tensor dimensions."""
    def __init__(self, shape: List[int]):
        self.shape = shape

    def apply(self, tensor: np.ndarray) -> np.ndarray:
        return np.reshape(tensor, self.shape)

    def __repr__(self):
        return f"Reshape(shape={self.shape})"


class Transpose(Operation):
    """Transpose operation that permutes tensor dimensions."""
    def __init__(self, axes: List[int]):
        self.axes = axes

    def apply(self, tensor: np.ndarray) -> np.ndarray:
        return np.transpose(tensor, self.axes)

    def __repr__(self):
        return f"Transpose(axes={self.axes})"


class Recipe:
    """A transformation recipe containing a sequence of operations."""
    def __init__(self):
        self.operations: List[Operation] = []
        self.input_shape: Tuple[int, ...] = None
        self.output_shape: Tuple[int, ...] = None
        self.axis_sizes: Dict[str, int] = {}

    def add_operation(self, op: Operation) -> 'Recipe':
        self.operations.append(op)
        return self

    def execute(self, tensor: np.ndarray) -> np.ndarray:
        """Apply all operations in sequence to transform the tensor."""
        result = tensor
        for op in self.operations:
            result = op.apply(result)
        return result

    def __repr__(self):
        return f"Recipe(operations={self.operations}, input_shape={self.input_shape}, output_shape={self.output_shape})"


class RecipeBuilder:
    """Builds a recipe for transforming a tensor according to input and output expressions."""

    def __init__(self, input_expr: ParsedExpression, output_expr: ParsedExpression, input_shape: Tuple[int, ...]):
        self.input_expr = input_expr
        self.output_expr = output_expr
        self.input_shape = input_shape
        self.recipe = Recipe()
        self.recipe.input_shape = input_shape

        # Maps axis names to their sizes
        self.axis_sizes: Dict[str, int] = {}

    def infer_axis_sizes(self) -> Dict[str, int]:
        """Infer sizes of all named axes from the input expression and shape."""
        # First infer basic axis sizes from the tree structure
        self.infer_axis_sizes_from_tree()

        self.recipe.axis_sizes = self.axis_sizes
        return self.axis_sizes

    def infer_axis_sizes_from_tree(self):
        """
        Maps tensor dimensions to axis names by traversing the expression tree.
        Handles split nodes directly by assigning correct sizes to component axes.
        """
        shape_idx = 0

        def process_node(node, shape_idx):
            """Process a node in the expression tree and return the updated index."""
            if isinstance(node, AxisNode):
                if shape_idx >= len(self.input_shape):
                    raise ValidationError(f"Not enough dimensions in shape for axis '{node.name}'")
                self.axis_sizes[node.name] = self.input_shape[shape_idx]
                return shape_idx + 1
                
            elif isinstance(node, EllipsisNode):
                # Calculate ellipsis size and store dimensions
                flat_input = flatten_expr(self.input_expr.nodes)
                remaining_named = sum(
                    1 for n in flat_input
                    if isinstance(n, AxisNode) and n != node
                )
                ellipsis_size = len(self.input_shape) - remaining_named
                
                if ellipsis_size < 0:
                    raise ValidationError("Not enough dimensions in shape for ellipsis")
                    
                for i in range(ellipsis_size):
                    self.axis_sizes[f"...{i}"] = self.input_shape[shape_idx + i]
                    
                return shape_idx + ellipsis_size
                
            elif isinstance(node, MergeNode):
                # Process each axis in the merge node
                for subnode in node.axes:
                    shape_idx = process_node(subnode, shape_idx)
                return shape_idx
                
            elif isinstance(node, SplitNode):
                # A split node consumes exactly ONE dimension from input shape
                if shape_idx >= len(self.input_shape):
                    raise ValidationError("Not enough dimensions in shape for split node")
                
                split_size = self.input_shape[shape_idx]
                
                # Get the named and anonymous components
                named_components = []
                anonymous_components = []
                
                for subnode in node.axes:
                    if isinstance(subnode, AxisNode):
                        named_components.append(subnode)
                    elif isinstance(subnode, AnonymousAxis):
                        anonymous_components.append(subnode)
                
                # Map component sizes directly
                component_sizes = {}
                
                # 1. First check for explicitly provided sizes in kwargs
                for comp in named_components:
                    if comp.name in self.axis_sizes and self.axis_sizes[comp.name] is not None:
                        component_sizes[comp.name] = self.axis_sizes[comp.name]
                
                # 2. Add sizes from anonymous components
                for comp in anonymous_components:
                    component_sizes[f"_anon_{len(component_sizes)}"] = comp.value
                
                # 3. Calculate and validate product
                if component_sizes:
                    product = 1
                    for size in component_sizes.values():
                        product *= size
                        
                    # If we know all component sizes, verify they match the parent size
                    if len(component_sizes) == len(named_components) + len(anonymous_components) - 1:
                        # Only one component missing, calculate its size
                        if split_size % product != 0:
                            raise ValidationError(
                                f"Cannot evenly divide {split_size} by {product}"
                            )
                        missing_size = split_size // product
                        
                        # Find the missing component and assign its size
                        for comp in named_components:
                            if comp.name not in component_sizes:
                                self.axis_sizes[comp.name] = missing_size
                                break
                    elif len(component_sizes) == len(named_components) + len(anonymous_components):
                        # All components accounted for, validate product
                        if product != split_size:
                            raise ValidationError(
                                f"Split axis sizes {list(component_sizes.values())} don't multiply to {split_size}"
                            )
                    elif len(component_sizes) < len(named_components) + len(anonymous_components) - 1:
                        # More than one component missing
                        raise ValidationError(
                            f"Need at least all but one component size for split node. Missing {len(named_components) + len(anonymous_components) - len(component_sizes)} sizes."
                        )
                    
                    # Update all component sizes
                    for comp in named_components:
                        if comp.name in component_sizes:
                            self.axis_sizes[comp.name] = component_sizes[comp.name]
                else:
                    # No component sizes available, can't split
                    raise ValidationError(
                        f"No component sizes available for split node. Provide at least one component size via kwargs."
                    )
                
                # Only consume one dimension for the entire split node
                return shape_idx + 1
                
            elif isinstance(node, AnonymousAxis):
                # Skip dimensions for anonymous axes
                return shape_idx + 1
                
            else:
                raise TypeError(f"Unsupported node type: {type(node)}")
        
        # Process all nodes in the expression
        for node in self.input_expr.nodes:
            shape_idx = process_node(node, shape_idx)
        
        # Verify we've consumed all dimensions
        if shape_idx != len(self.input_shape):
            raise ValidationError(
                f"Input shape has {len(self.input_shape)} dimensions but pattern only used {shape_idx}"
            )
        
        return self.axis_sizes

    def analyze_split_nodes(self):
        """
        Analyze split nodes in the input expression to correctly infer axis sizes.
        
        Steps:
        1. Identify split nodes in the input expression
        2. Treat the entire split node as a single dimension mapped to input size
        3. Extract component axes and their sizes from kwargs
        4. Validate that component sizes multiply to match the parent dimension
        5. Update axis sizes dictionary with correct values
        """
        # Find all SplitNode instances in the input expression
        split_nodes = []
        
        def find_split_nodes(nodes):
            for node in nodes:
                if isinstance(node, SplitNode):
                    split_nodes.append(node)
                elif isinstance(node, MergeNode):
                    find_split_nodes(node.axes)
        
        find_split_nodes(self.input_expr.nodes)
        
        # Process each split node
        for node in split_nodes:
            # Step 1: Get the total size of this split node
            # (should have been assigned during infer_axis_sizes_from_tree)
            
            # Find a reference axis to get the total size
            reference_axis = None
            for subnode in node.axes:
                if isinstance(subnode, AxisNode):
                    if subnode.name in self.axis_sizes and self.axis_sizes[subnode.name] is not None:
                        reference_axis = subnode
                        break
            
            if not reference_axis:
                continue  # Can't process this split node without a reference
            
            parent_size = self.axis_sizes[reference_axis.name]
            
            # Step 2: Get all component axes (excluding the reference axis)
            components = [subnode for subnode in node.axes if subnode != reference_axis]
            named_components = [c for c in components if isinstance(c, AxisNode)]
            anonymous_components = [c for c in components if isinstance(c, AnonymousAxis)]
            
            # Step 3: Calculate sizes for all components
            component_sizes = {}
            
            # Add sizes from named components in kwargs
            for comp in named_components:
                if comp.name in self.axis_sizes and self.axis_sizes[comp.name] is not None:
                    component_sizes[comp.name] = self.axis_sizes[comp.name]
            
            # Add sizes from anonymous components
            for i, comp in enumerate(anonymous_components):
                component_sizes[f"_anon_{i}"] = comp.value
            
            # Step 4: Validate or infer component sizes
            
            # If we know all component sizes, validate they multiply to match parent size
            if len(component_sizes) > 0:
                product = 1
                for size in component_sizes.values():
                    product *= size
                
                # If we have all components, verify they match the parent size
                if len(component_sizes) == len(components):
                    if product != parent_size:
                        raise ValidationError(
                            f"Split axis sizes don't multiply to match the parent size. "
                            f"Got product {product}, expected {parent_size}"
                        )
                
                # If we're missing exactly one component, calculate its size
                elif len(component_sizes) == len(components) - 1:
                    if parent_size % product != 0:
                        raise ValidationError(
                            f"Cannot evenly divide parent size {parent_size} by known factors {product}"
                        )
                    
                    missing_size = parent_size // product
                    
                    # Find the component with missing size and assign it
                    for comp in named_components:
                        if comp.name not in component_sizes:
                            self.axis_sizes[comp.name] = missing_size
                            break
                
                # If we're missing more than one component, we need more information
                else:
                    missing_count = len(components) - len(component_sizes)
                    if missing_count > 1:
                        raise ValidationError(
                            f"Need at least all but one component size to infer split dimensions. "
                            f"Missing {missing_count} sizes for split of size {parent_size}"
                        )
            else:
                # No component sizes available - can't proceed
                raise ValidationError(
                    f"No component sizes available for split node with total size {parent_size}. "
                    f"Provide at least one component size via kwargs."
                )
            
            # Step 5: Update all axis sizes for components
            for comp in named_components:
                if comp.name in component_sizes:
                    self.axis_sizes[comp.name] = component_sizes[comp.name]

    def build_transpose_operation(self):
        """
        Build a transpose operation if the axis order changes.
        Properly handles ellipsis in both input and output.
        """
        # Get flat representations of input and output expressions
        flat_input = flatten_expr(self.input_expr.nodes)
        flat_output = flatten_expr(self.output_expr.nodes)
        
        # Create mapping from axis names to positions in the input
        axis_positions = {}
        pos = 0
        ellipsis_pos = -1
        ellipsis_size = 0
        
        # First pass: Map positions and handle ellipsis in input
        for node in flat_input:
            if isinstance(node, AxisNode):
                axis_positions[node.name] = pos
                pos += 1
            elif isinstance(node, EllipsisNode):
                # Calculate how many dimensions the ellipsis represents
                ellipsis_size = sum(1 for k in self.axis_sizes if k.startswith("..."))
                ellipsis_pos = pos
                pos += ellipsis_size
        
        # Build the list of dimensions for ellipsis
        ellipsis_dims = []
        if ellipsis_pos >= 0:
            ellipsis_dims = list(range(ellipsis_pos, ellipsis_pos + ellipsis_size))
        
        # Build the permutation indices based on output order
        perm = []
        for node in flat_output:
            if isinstance(node, AxisNode):
                if node.name in axis_positions:
                    perm.append(axis_positions[node.name])
            elif isinstance(node, EllipsisNode):
                # Include all ellipsis dimensions in order
                perm.extend(ellipsis_dims)
        
        # Only add transpose if the permutation changes dimension order
        if perm and perm != list(range(len(perm))):
            self.recipe.add_operation(Transpose(perm))

    def calculate_output_shape(self) -> List[int]:
        """Calculate the final output shape based on the output expression."""
        output_shape = []
        
        def process_node(node):
            """Process a node in the output expression to determine its shape contribution."""
            if isinstance(node, AxisNode):
                if node.name in self.axis_sizes:
                    output_shape.append(self.axis_sizes[node.name])
                else:
                    raise ValidationError(f"Output axis '{node.name}' not found in input or explicitly provided")
            
            elif isinstance(node, MergeNode):
                # Calculate merged size as product of component sizes
                merged_size = 1
                for subnode in node.axes:
                    if isinstance(subnode, AxisNode) and subnode.name in self.axis_sizes:
                        merged_size *= self.axis_sizes[subnode.name]
                output_shape.append(merged_size)
            
            elif isinstance(node, SplitNode):
                # Add each component size separately
                for subnode in node.axes:
                    if isinstance(subnode, AxisNode) and subnode.name in self.axis_sizes:
                        output_shape.append(self.axis_sizes[subnode.name])
                    elif isinstance(subnode, AnonymousAxis):
                        output_shape.append(subnode.value)
            
            elif isinstance(node, EllipsisNode):
                # Add all ellipsis dimensions in order
                ellipsis_dims = sorted([k for k in self.axis_sizes if k.startswith("...")])
                for key in ellipsis_dims:
                    output_shape.append(self.axis_sizes[key])
        
        # Process each node in the output expression
        for node in self.output_expr.nodes:
            process_node(node)
        
        return output_shape

    def build(self) -> Recipe:
        """Build the complete recipe."""
        # Step 1: Infer sizes of all axes
        self.infer_axis_sizes()
        
        # Step 2: Build transpose operation if needed
        self.build_transpose_operation()
        
        # Step 3: Calculate the final output shape
        output_shape = self.calculate_output_shape()
        
        # Step 4: Calculate the shape after existing operations
        current_shape = list(self.input_shape)
        for op in self.recipe.operations:
            if isinstance(op, Transpose):
                current_shape = [current_shape[i] for i in op.axes]
        
        # Step 5: Add reshape operation if needed
        if tuple(current_shape) != tuple(output_shape):
            self.recipe.add_operation(Reshape(output_shape))
        
        # Step 6: Set the final output shape on the recipe
        self.recipe.output_shape = tuple(output_shape)
        
        return self.recipe


def rearrange(tensor: np.ndarray, pattern: str, **axis_sizes) -> np.ndarray:
    """
    Rearrange a tensor according to the pattern.
    
    Args:
        tensor: Input tensor (numpy array)
        pattern: Pattern string like "b c h w -> b (c h) w"
        **axis_sizes: Additional axis sizes for splits
        
    Returns:
        Transformed tensor
    """
    if "->" not in pattern:
        raise ValueError("Pattern must contain '->' to separate input and output expressions")
        
    input_pattern, output_pattern = pattern.split("->")
    
    # Parse expressions, passing along known axis sizes to help identify split nodes
    input_expr = ParsedExpression(input_pattern.strip(), is_input_pattern=True, known_axes=axis_sizes)
    output_expr = ParsedExpression(output_pattern.strip(), is_input_pattern=False)
    
    # Build and execute the recipe
    builder = RecipeBuilder(input_expr, output_expr, tensor.shape)
    
    # Add any explicit axis sizes BEFORE inference
    for axis, size in axis_sizes.items():
        builder.axis_sizes[axis] = size
    
    recipe = builder.build()
    
    return recipe.execute(tensor)

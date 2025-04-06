import numpy as np

# Try to import Numba, fall back gracefully if not available
try:
    import numba
    from numba.core.errors import TypingError

    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    print("Numba not found, falling back to standard NumPy operations")

# Define JIT-compiled operations if numba is available
if HAVE_NUMBA:

    @numba.njit(cache=True, fastmath=True)
    def jit_reshape_internal(tensor, shape):
        """Internal JIT-compiled reshape for contiguous arrays only"""
        return tensor.reshape(shape)

    def jit_reshape(tensor, shape):
        """Wrapper for reshape that ensures array is contiguous"""
        # Ensure array is contiguous by copying if needed
        if not tensor.flags.c_contiguous and not tensor.flags.f_contiguous:
            tensor = np.ascontiguousarray(tensor)

        try:
            return jit_reshape_internal(tensor, shape)
        except (TypingError, ValueError):
            # Fall back to numpy if JIT fails
            return tensor.reshape(shape)

    @numba.njit(cache=True, fastmath=True)
    def jit_transpose(tensor, axes):
        """JIT-compiled version of transpose operation"""
        return np.transpose(tensor, axes)

    # Special case optimizations for common operations
    @numba.njit(cache=True, fastmath=True)
    def jit_transpose_2d(tensor):
        """Fast transpose for 2D matrices"""
        return tensor.T

    def jit_merge_last_two_dims(tensor):
        """Merge the last two dimensions into one"""
        shape = tensor.shape
        new_shape = shape[:-2] + (shape[-2] * shape[-1],)

        # Ensure array is contiguous
        if not tensor.flags.c_contiguous and not tensor.flags.f_contiguous:
            tensor = np.ascontiguousarray(tensor)

        try:
            return jit_reshape_internal(tensor, new_shape)
        except (TypingError, ValueError):
            # Fall back to numpy
            return tensor.reshape(new_shape)

    def jit_split_last_dim(tensor, dim1, dim2):
        """Split the last dimension into two dimensions"""
        shape = tensor.shape
        new_shape = shape[:-1] + (dim1, dim2)

        # Ensure array is contiguous
        if not tensor.flags.c_contiguous and not tensor.flags.f_contiguous:
            tensor = np.ascontiguousarray(tensor)

        try:
            return jit_reshape_internal(tensor, new_shape)
        except (TypingError, ValueError):
            # Fall back to numpy
            return tensor.reshape(new_shape)

else:
    # Fallback implementations using standard NumPy
    def jit_reshape(tensor, shape):
        return tensor.reshape(shape)

    def jit_transpose(tensor, axes):
        return np.transpose(tensor, axes)

    def jit_transpose_2d(tensor):
        return tensor.T

    def jit_merge_last_two_dims(tensor):
        shape = tensor.shape
        new_shape = shape[:-2] + (shape[-2] * shape[-1],)
        return tensor.reshape(new_shape)

    def jit_split_last_dim(tensor, dim1, dim2):
        shape = tensor.shape
        new_shape = shape[:-1] + (dim1, dim2)
        return tensor.reshape(new_shape)

# ein
Implementation for rearrange function from einops library

## Benchmarks

> With no optimization and bare implementation

Our implementation: 0.021 ms  
Einops: 0.002 ms  
Ratio (our/einops): 12.59x  
Results match ✓

> Implemented expression caching and refactored operations

```
================================================================================
BENCHMARK RESULTS - 2025-04-06 15:05:52
================================================================================
Running 6 test cases with 20 iterations each (after 5 warmup runs)
================================================================================
```

Running benchmark: Simple Transpose - Basic dimension reordering
- Ein: 0.001 ms (± 0.000 ms) | 1188124.8 ops/sec
- einops: 0.001 ms (± 0.000 ms) | 1016879.2 ops/sec
- Ratio (Ein/einops): 0.86x | Results match: ✓

Running benchmark: Merge Axes - Combining dimensions
- Ein: 0.001 ms (± 0.000 ms) | 902289.8 ops/sec
- einops: 0.001 ms (± 0.000 ms) | 948681.8 ops/sec
- Ratio (Ein/einops): 1.05x | Results match: ✓

Running benchmark: Split Axes - Splitting a dimension into multiple
- Ein: 0.004 ms (± 0.000 ms) | 276654.9 ops/sec
- einops: 0.004 ms (± 0.000 ms) | 244894.2 ops/sec
- Ratio (Ein/einops): 0.89x | Results match: ✓

Running benchmark: Complex Rearrangement - Common CNN tensor reshaping
- Ein: 0.001 ms (± 0.000 ms) | 716462.3 ops/sec
- einops: 0.001 ms (± 0.000 ms) | 824846.9 ops/sec
- Ratio (Ein/einops): 1.15x | Results match: ✓

Running benchmark: Ellipsis Handling - Using ellipsis to handle variable dimensions
- Ein: 0.001 ms (± 0.000 ms) | 1019059.7 ops/sec
- einops: 0.001 ms (± 0.000 ms) | 902287.5 ops/sec
- Ratio (Ein/einops): 0.89x | Results match: ✓

Running benchmark: Large Tensor - Performance with larger tensors
- Ein: 0.001 ms (± 0.000 ms) | 1066781.1 ops/sec
- einops: 0.001 ms (± 0.000 ms) | 983674.2 ops/sec
- Ratio (Ein/einops): 0.92x | Results match: ✓

| Test Case             | Shape              | Ein (ms)    | Ein (ops/s)   | einops (ms)   | einops (ops/s)  | Ratio   | Match   |
|-----------------------|--------------------|-------------|---------------|---------------|-----------------|---------|---------|
| Simple Transpose      | (100, 100, 100)    | 0.001±0.000 | 1.18812e+06   | 0.001±0.000   | 1.01688e+06     | 0.86x   | ✓       |
| Merge Axes            | (32, 64, 128)      | 0.001±0.000 | 902290        | 0.001±0.000   | 948682          | 1.05x   | ✓       |
| Split Axes            | (32, 1024)         | 0.004±0.000 | 276655        | 0.004±0.000   | 244894          | 0.89x   | ✓       |
| Complex Rearrangement | (8, 3, 32, 32)     | 0.001±0.000 | 716462        | 0.001±0.000   | 824847          | 1.15x   | ✓       |
| Ellipsis Handling     | (8, 3, 32, 32, 16) | 0.001±0.000 | 1.01906e+06   | 0.001±0.000   | 902288          | 0.89x   | ✓       |
| Large Tensor          | (64, 128, 256)     | 0.001±0.000 | 1.06678e+06   | 0.001±0.000   | 983674          | 0.92x   | ✓       |

Performance comparison chart saved to 'benchmark_results.png'

```
================================================================================
CACHE EFFECTIVENESS TEST
================================================================================
First run: 0.026 ms
Average cached run: 0.001 ms
Cache speedup: 26.87x
================================================================================

================================================================================
MEMORY USAGE TEST
================================================================================
Baseline memory usage: 167.70 MB
Memory usage before operation: 283.66 MB
Memory usage after operation: 283.66 MB
Memory increase: 0.00 MB
================================================================================

================================================================================
DETAILED PROFILING
================================================================================
```

Detailed profile for: Complex Rearrangement - Common CNN tensor reshaping
18 function calls in 0.000 seconds

Ordered by: cumulative time
List reduced from 16 to 15 due to restriction <15>

```
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     1    0.000    0.000    0.000    0.000 /Users/karnikkanojia/Desktop/ein/ein/recipe.py:409(rearrange)
     1    0.000    0.000    0.000    0.000 /Users/karnikkanojia/Desktop/ein/ein/recipe.py:107(execute)
     1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
     1    0.000    0.000    0.000    0.000 /Users/karnikkanojia/Desktop/ein/ein/recipe.py:63(apply)
     2    0.000    0.000    0.000    0.000 /opt/homebrew/Caskroom/miniconda/base/envs/einops/lib/python3.12/site-packages/numpy/core/fromnumeric.py:53(_wrapfunc)
     1    0.000    0.000    0.000    0.000 /opt/homebrew/Caskroom/miniconda/base/envs/einops/lib/python3.12/site-packages/numpy/core/fromnumeric.py:588(transpose)
     1    0.000    0.000    0.000    0.000 /Users/karnikkanojia/Desktop/ein/ein/recipe.py:405(get_cache_key)
     1    0.000    0.000    0.000    0.000 /Users/karnikkanojia/Desktop/ein/ein/recipe.py:51(apply)
     1    0.000    0.000    0.000    0.000 /opt/homebrew/Caskroom/miniconda/base/envs/einops/lib/python3.12/site-packages/numpy/core/fromnumeric.py:200(reshape)
     1    0.000    0.000    0.000    0.000 {method 'reshape' of 'numpy.ndarray' objects}
     1    0.000    0.000    0.000    0.000 {method 'transpose' of 'numpy.ndarray' objects}
     2    0.000    0.000    0.000    0.000 {built-in method builtins.getattr}
     1    0.000    0.000    0.000    0.000 {built-in method builtins.sorted}
     1    0.000    0.000    0.000    0.000 {method 'items' of 'dict' objects}
     1    0.000    0.000    0.000    0.000 /opt/homebrew/Caskroom/miniconda/base/envs/einops/lib/python3.12/site-packages/numpy/core/fromnumeric.py:195(_reshape_dispatcher)
```

> Implemented numba compiled operations to speed up numpy operations

```
================================================================================
BENCHMARK RESULTS - 2025-04-07 00:07:23
================================================================================
Running 6 test cases with 20 iterations each (after 5 warmup runs)
================================================================================
```

Running benchmark: Simple Transpose - Basic dimension reordering
- Ein: 0.001 ms (± 0.000 ms) | 707914.0 ops/sec
- einops: 0.001 ms (± 0.000 ms) | 897142.2 ops/sec
- Ratio (Ein/einops): 1.27x | Results match: ✓

Running benchmark: Merge Axes - Combining dimensions
- Ein: 0.001 ms (± 0.000 ms) | 781734.6 ops/sec
- einops: 0.001 ms (± 0.000 ms) | 905507.1 ops/sec
- Ratio (Ein/einops): 1.16x | Results match: ✓

Running benchmark: Split Axes - Splitting a dimension into multiple
- Ein: 0.002 ms (± 0.000 ms) | 600779.8 ops/sec
- einops: 0.002 ms (± 0.000 ms) | 655718.9 ops/sec
- Ratio (Ein/einops): 1.09x | Results match: ✓

Running benchmark: Complex Rearrangement - Common CNN tensor reshaping
- Ein: 0.029 ms (± 0.000 ms) | 35067.2 ops/sec
- einops: 0.001 ms (± 0.000 ms) | 780578.5 ops/sec
- Ratio (Ein/einops): 22.26x | Results match: ✓

Running benchmark: Ellipsis Handling - Using ellipsis to handle variable dimensions
- Ein: 0.001 ms (± 0.000 ms) | 683762.3 ops/sec
- einops: 0.001 ms (± 0.000 ms) | 875889.4 ops/sec
- Ratio (Ein/einops): 1.28x | Results match: ✓

Running benchmark: Large Tensor - Performance with larger tensors
- Ein: 0.001 ms (± 0.000 ms) | 770473.7 ops/sec
- einops: 0.001 ms (± 0.000 ms) | 930195.3 ops/sec
- Ratio (Ein/einops): 1.21x | Results match: ✓

| Test Case             | Shape              | Ein (ms)    | Ein (ops/s) | einops (ms)   | einops (ops/s) | Ratio   | Match   |
|-----------------------|--------------------|-------------|-------------|---------------|---------------|---------|---------|
| Simple Transpose      | (100, 100, 100)    | 0.001±0.000 | 707914      | 0.001±0.000   | 897142        | 1.27x   | ✓       |
| Merge Axes            | (32, 64, 128)      | 0.001±0.000 | 781735      | 0.001±0.000   | 905507        | 1.16x   | ✓       |
| Split Axes            | (32, 1024)         | 0.002±0.000 | 600780      | 0.002±0.000   | 655719        | 1.09x   | ✓       |
| Complex Rearrangement | (8, 3, 32, 32)     | 0.029±0.000 | 35067.2     | 0.001±0.000   | 780578        | 22.26x  | ✓       |
| Ellipsis Handling     | (8, 3, 32, 32, 16) | 0.001±0.000 | 683762      | 0.001±0.000   | 875889        | 1.28x   | ✓       |
| Large Tensor          | (64, 128, 256)     | 0.001±0.000 | 770474      | 0.001±0.000   | 930195        | 1.21x   | ✓       |

Performance comparison chart saved to 'benchmark_results.png'

```
================================================================================
CACHE EFFECTIVENESS TEST
================================================================================
First run: 0.031 ms
Average cached run: 0.001 ms
Cache speedup: 23.28x
================================================================================

================================================================================
MEMORY USAGE TEST
================================================================================
Baseline memory usage: 239.00 MB
Memory usage before operation: 431.00 MB
Memory usage after operation: 431.00 MB
Memory increase: 0.00 MB
================================================================================

```

> Without numba JIT compilation

Numba not found, falling back to standard NumPy operations

```
================================================================================
BENCHMARK RESULTS - 2025-04-07 00:14:23
================================================================================
Running 6 test cases with 20 iterations each (after 5 warmup runs)
================================================================================
```

Running benchmark: Simple Transpose - Basic dimension reordering
- Ein: 0.001 ms (± 0.000 ms) | 969748.4 ops/sec
- einops: 0.001 ms (± 0.000 ms) | 952379.0 ops/sec
- Ratio (Ein/einops): 0.98x | Results match: ✓

Running benchmark: Merge Axes - Combining dimensions
- Ein: 0.001 ms (± 0.000 ms) | 1164950.1 ops/sec
- einops: 0.001 ms (± 0.000 ms) | 941218.3 ops/sec
- Ratio (Ein/einops): 0.81x | Results match: ✓

Running benchmark: Split Axes - Splitting a dimension into multiple
- Ein: 0.001 ms (± 0.000 ms) | 987649.6 ops/sec
- einops: 0.001 ms (± 0.000 ms) | 754721.2 ops/sec
- Ratio (Ein/einops): 0.76x | Results match: ✓

Running benchmark: Complex Rearrangement - Common CNN tensor reshaping
- Ein: 0.001 ms (± 0.000 ms) | 857227.6 ops/sec
- einops: 0.001 ms (± 0.000 ms) | 826141.9 ops/sec
- Ratio (Ein/einops): 0.96x | Results match: ✓

Running benchmark: Ellipsis Handling - Using ellipsis to handle variable dimensions
- Ein: 0.001 ms (± 0.000 ms) | 1003925.1 ops/sec
- einops: 0.001 ms (± 0.001 ms) | 817724.9 ops/sec
- Ratio (Ein/einops): 0.81x | Results match: ✓

Running benchmark: Large Tensor - Performance with larger tensors
- Ein: 0.001 ms (± 0.000 ms) | 1307876.2 ops/sec
- einops: 0.001 ms (± 0.000 ms) | 1036492.9 ops/sec
- Ratio (Ein/einops): 0.79x | Results match: ✓

| Test Case             | Shape              | Ein (ms)    | Ein (ops/s)    | einops (ms)   | einops (ops/s)  | Ratio   | Match   |
|-----------------------|--------------------|-------------|----------------|---------------|-----------------|---------|---------|
| Simple Transpose      | (100, 100, 100)    | 0.001±0.000 | 969748         | 0.001±0.000   | 952379          | 0.98x   | ✓       |
| Merge Axes            | (32, 64, 128)      | 0.001±0.000 | 1.16495e+06    | 0.001±0.000   | 941218          | 0.81x   | ✓       |
| Split Axes            | (32, 1024)         | 0.001±0.000 | 987650         | 0.001±0.000   | 754721          | 0.76x   | ✓       |
| Complex Rearrangement | (8, 3, 32, 32)     | 0.001±0.000 | 857228         | 0.001±0.000   | 826142          | 0.96x   | ✓       |
| Ellipsis Handling     | (8, 3, 32, 32, 16) | 0.001±0.000 | 1.00393e+06    | 0.001±0.001   | 817725          | 0.81x   | ✓       |
| Large Tensor          | (64, 128, 256)     | 0.001±0.000 | 1.30788e+06    | 0.001±0.000   | 1.03649e+06     | 0.79x   | ✓       |

Performance comparison chart saved to 'benchmark_results.png'

```
================================================================================
CACHE EFFECTIVENESS TEST
================================================================================
First run: 0.020 ms
Average cached run: 0.001 ms
Cache speedup: 25.13x
================================================================================

================================================================================
MEMORY USAGE TEST
================================================================================
Baseline memory usage: 175.06 MB
Memory usage before operation: 344.98 MB
Memory usage after operation: 344.98 MB
Memory increase: 0.00 MB
================================================================================

```

## Implementation Details

### Approach

The `ein` library is a clean-room implementation of the tensor rearrangement functionality provided by the popular `einops` library. Our key design principles were:

1. **Parser-First Architecture**: We built a robust parser for Einstein-notation expressions that constructs an expression tree, allowing for flexible tensor manipulations.

2. **Recipe-Based Execution**: Operations are compiled into "recipes" that can be cached and reused, avoiding repeated parsing overhead.

3. **Minimal Dependencies**: The core implementation relies only on NumPy, with optional Numba acceleration.

### Design Decisions

#### Expression Parsing

Expressions like `"b c h w -> b (h w) c"` are parsed into a tree of nodes representing different tensor operations:
- `AxisNode`: Named dimensions (e.g., 'b', 'c')
- `MergeNode`: Combined dimensions (e.g., '(h w)')
- `SplitNode`: Dimensions split into parts
- `EllipsisNode`: Variable dimensions (`...`)

This tree-based representation allows for easy validation and transformation of tensor operations.

#### Caching System

We implemented an aggressive caching system that stores:
1. Parsed expressions to avoid repeated parsing
2. Compiled recipes to avoid recomputing reshape and transpose parameters
3. Common transformation patterns for quick lookup

This results in a significant speedup for repeated operations (25x faster for cached vs. first-time calls).

#### Optimization Strategies

Our implementation uses several optimization strategies:
- Minimizing intermediate tensor allocations
- Using direct NumPy operations where possible
- Optimizing memory layout with contiguous arrays

#### Numba Integration

We experimented with Numba JIT compilation for core operations, but found that for tensor reshape and transpose operations, the overhead of Numba compilation often outweighs the benefits for small tensors. In particular:
- Simple operations are ~25% faster without Numba
- Complex rearrangements show dramatic differences (857K ops/sec without Numba vs 35K ops/sec with Numba)

### Comparison with `einops`

#### Similarities
- API compatible with `einops.rearrange`
- Support for the same expression syntax
- Handling of named dimensions, merging, splitting, and ellipsis

#### Differences
- Different internal representation of tensor operations
- Our implementation performs better without Numba JIT compilation
- More aggressive caching strategy

#### Performance

As shown in the benchmark results, our implementation:
- Outperforms einops on most operations when using standard NumPy (0.76x-0.98x ratio)
- Is particularly efficient for splitting axes (25% faster than einops)
- Shows consistent performance across different tensor shapes and operations
- Has excellent cache utilization (25x speedup for cached operations)

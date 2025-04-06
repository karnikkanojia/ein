import numpy as np
import time
import cProfile
import pstats
import io
import statistics
import matplotlib.pyplot as plt
from tabulate import tabulate
from datetime import datetime
from ein.recipe import rearrange

# Try to import einops, skip comparison if not installed
try:
    import einops
    HAVE_EINOPS = True
except ImportError:
    HAVE_EINOPS = False
    print("einops not found, skipping comparison")

# Define benchmark test cases
TEST_CASES = [
    {
        "name": "Simple Transpose",
        "tensor_shape": (100, 100, 100),
        "pattern": "a b c -> a c b",
        "kwargs": {},
        "description": "Basic dimension reordering"
    },
    {
        "name": "Merge Axes",
        "tensor_shape": (32, 64, 128),
        "pattern": "a b c -> a (b c)",
        "kwargs": {},
        "description": "Combining dimensions"
    },
    {
        "name": "Split Axes",
        "tensor_shape": (32, 1024),
        "pattern": "a (b c) -> a b c",
        "kwargs": {"b": 32, "c": 32},
        "description": "Splitting a dimension into multiple"
    },
    {
        "name": "Complex Rearrangement",
        "tensor_shape": (8, 3, 32, 32),
        "pattern": "b c h w -> b (h w) c",
        "kwargs": {},
        "description": "Common CNN tensor reshaping"
    },
    {
        "name": "Ellipsis Handling",
        "tensor_shape": (8, 3, 32, 32, 16),
        "pattern": "b ... h -> h ... b",
        "kwargs": {},
        "description": "Using ellipsis to handle variable dimensions"
    },
    {
        "name": "Large Tensor",
        "tensor_shape": (64, 128, 256),
        "pattern": "a b c -> a (b c)",
        "kwargs": {},
        "description": "Performance with larger tensors"
    },
]

def run_single_benchmark(test_case, iterations=10, warmup=3):
    """Run a single benchmark test comparing Ein vs einops."""
    tensor = np.random.rand(*test_case["tensor_shape"]).astype(np.float32)
    pattern = test_case["pattern"]
    kwargs = test_case["kwargs"]
    
    # Warmup runs
    for _ in range(warmup):
        rearrange(tensor, pattern, **kwargs)
        if HAVE_EINOPS:
            einops.rearrange(tensor, pattern, **kwargs)
    
    # Benchmark Ein
    ein_times = []
    for _ in range(iterations):
        start_time = time.perf_counter()
        result_ein = rearrange(tensor, pattern, **kwargs)
        ein_times.append(time.perf_counter() - start_time)
    
    # Benchmark einops if available
    einops_times = []
    if HAVE_EINOPS:
        for _ in range(iterations):
            start_time = time.perf_counter()
            result_einops = einops.rearrange(tensor, pattern, **kwargs)
            einops_times.append(time.perf_counter() - start_time)
        
        # Verify results match
        try:
            np.testing.assert_allclose(result_ein, result_einops, rtol=1e-5)
            match = "✓"
        except:
            match = "✗"
    else:
        match = "N/A"
    
    # Calculate statistics
    ein_mean = statistics.mean(ein_times) * 1000  # ms
    ein_median = statistics.median(ein_times) * 1000  # ms
    ein_stdev = statistics.stdev(ein_times) * 1000 if iterations > 1 else 0  # ms
    ein_ops_per_sec = 1.0 / statistics.mean(ein_times)
    
    if HAVE_EINOPS:
        einops_mean = statistics.mean(einops_times) * 1000  # ms
        einops_median = statistics.median(einops_times) * 1000  # ms
        einops_stdev = statistics.stdev(einops_times) * 1000 if iterations > 1 else 0  # ms
        einops_ops_per_sec = 1.0 / statistics.mean(einops_times)
        ratio = ein_mean / einops_mean
    else:
        einops_mean = einops_median = einops_stdev = einops_ops_per_sec = ratio = float('nan')
    
    return {
        "test_case": test_case["name"],
        "tensor_shape": test_case["tensor_shape"],
        "pattern": pattern,
        "ein_mean_ms": ein_mean,
        "ein_median_ms": ein_median,
        "ein_stdev_ms": ein_stdev,
        "ein_ops_per_sec": ein_ops_per_sec,
        "einops_mean_ms": einops_mean,
        "einops_median_ms": einops_median,
        "einops_stdev_ms": einops_stdev,
        "einops_ops_per_sec": einops_ops_per_sec,
        "ratio": ratio,
        "match": match
    }

def profile_operation(test_case):
    """Profile a single operation and return statistics."""
    tensor = np.random.rand(*test_case["tensor_shape"]).astype(np.float32)
    pattern = test_case["pattern"]
    kwargs = test_case["kwargs"]
    
    # Profile Ein
    pr = cProfile.Profile()
    pr.enable()
    rearrange(tensor, pattern, **kwargs)
    pr.disable()
    
    # Get stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(15)  # Print top 15 functions
    
    return s.getvalue()

def run_benchmarks(iterations=20, warmup=5):
    """Run all benchmarks and return results."""
    print(f"\n{'=' * 80}")
    print(f"BENCHMARK RESULTS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")
    print(f"Running {len(TEST_CASES)} test cases with {iterations} iterations each (after {warmup} warmup runs)")
    print(f"{'=' * 80}\n")
    
    results = []
    
    for test_case in TEST_CASES:
        print(f"Running benchmark: {test_case['name']} - {test_case['description']}")
        result = run_single_benchmark(test_case, iterations, warmup)
        results.append(result)
        
        # Print individual result
        print(f"  Ein: {result['ein_mean_ms']:.3f} ms (± {result['ein_stdev_ms']:.3f} ms) | {result['ein_ops_per_sec']:.1f} ops/sec")
        if HAVE_EINOPS:
            print(f"  einops: {result['einops_mean_ms']:.3f} ms (± {result['einops_stdev_ms']:.3f} ms) | {result['einops_ops_per_sec']:.1f} ops/sec")
            print(f"  Ratio (Ein/einops): {result['ratio']:.2f}x | Results match: {result['match']}")
        print()
    
    return results

def print_tabulated_results(results):
    """Print results in a formatted table."""
    if HAVE_EINOPS:
        headers = [
            "Test Case", "Shape", "Ein (ms)", "Ein (ops/s)", 
            "einops (ms)", "einops (ops/s)", "Ratio", "Match"
        ]
        
        table_data = []
        for r in results:
            table_data.append([
                r["test_case"],
                r["tensor_shape"],
                f"{r['ein_mean_ms']:.3f}±{r['ein_stdev_ms']:.3f}",
                f"{r['ein_ops_per_sec']:.1f}",
                f"{r['einops_mean_ms']:.3f}±{r['einops_stdev_ms']:.3f}",
                f"{r['einops_ops_per_sec']:.1f}",
                f"{r['ratio']:.2f}x",
                r["match"]
            ])
    else:
        headers = ["Test Case", "Shape", "Ein (ms)", "Ein (ops/s)"]
        
        table_data = []
        for r in results:
            table_data.append([
                r["test_case"],
                r["tensor_shape"],
                f"{r['ein_mean_ms']:.3f}±{r['ein_stdev_ms']:.3f}",
                f"{r['ein_ops_per_sec']:.1f}",
            ])
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def plot_results(results):
    """Generate a bar chart comparing Ein vs einops performance."""
    if not HAVE_EINOPS:
        return
        
    try:
        test_names = [r["test_case"] for r in results]
        ein_times = [r["ein_mean_ms"] for r in results]
        einops_times = [r["einops_mean_ms"] for r in results]
        
        # Create figure and axis
        plt.figure(figsize=(12, 8))
        x = np.arange(len(test_names))
        width = 0.35
        
        plt.bar(x - width/2, ein_times, width, label='Ein')
        plt.bar(x + width/2, einops_times, width, label='einops')
        
        plt.xlabel('Test Case')
        plt.ylabel('Time (ms)')
        plt.title('Performance Comparison: Ein vs einops')
        plt.xticks(x, test_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plt.savefig('benchmark_results.png')
        print(f"Performance comparison chart saved to 'benchmark_results.png'")
        
    except Exception as e:
        print(f"Could not generate plot: {e}")

def cache_effectiveness_test():
    """Test the effectiveness of Ein's caching mechanism."""
    print(f"\n{'=' * 80}")
    print("CACHE EFFECTIVENESS TEST")
    print(f"{'=' * 80}")
    
    # Define test cases
    tensor_shape = (64, 32, 32)
    pattern = "a b c -> a (b c)"
    tensor = np.random.rand(*tensor_shape).astype(np.float32)
    iterations = 100
    
    # First run - should include parsing and setup time
    start_time = time.perf_counter()
    rearrange(tensor, pattern)
    first_run_time = (time.perf_counter() - start_time) * 1000
    print(f"First run: {first_run_time:.3f} ms")
    
    # Subsequent runs - should be faster due to caching
    times = []
    for _ in range(iterations):
        start_time = time.perf_counter()
        rearrange(tensor, pattern)
        times.append((time.perf_counter() - start_time) * 1000)
    
    avg_cached_time = statistics.mean(times)
    speedup = first_run_time / avg_cached_time
    
    print(f"Average cached run: {avg_cached_time:.3f} ms")
    print(f"Cache speedup: {speedup:.2f}x")
    print(f"{'=' * 80}")

def memory_usage_test():
    """Estimate memory usage during operations."""
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        print(f"\n{'=' * 80}")
        print("MEMORY USAGE TEST")
        print(f"{'=' * 80}")
        
        # Get baseline memory usage
        baseline = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Baseline memory usage: {baseline:.2f} MB")
        
        # Test with large tensor
        tensor_shape = (256, 256, 256)  # ~64MB tensor
        tensor = np.random.rand(*tensor_shape).astype(np.float32)
        pattern = "a b c -> a (b c)"
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Measure memory before operation
        before = process.memory_info().rss / 1024 / 1024
        
        # Perform operation
        result = rearrange(tensor, pattern)
        
        # Measure memory after operation
        after = process.memory_info().rss / 1024 / 1024
        
        print(f"Memory usage before operation: {before:.2f} MB")
        print(f"Memory usage after operation: {after:.2f} MB")
        print(f"Memory increase: {after - before:.2f} MB")
        print(f"{'=' * 80}")
        
    except ImportError:
        print("psutil not installed, skipping memory usage test")

def profile_detailed_test_case():
    """Run a detailed profile on a complex test case."""
    print(f"\n{'=' * 80}")
    print("DETAILED PROFILING")
    print(f"{'=' * 80}")
    
    # Choose a complex test case for detailed profiling
    complex_case = next(tc for tc in TEST_CASES if tc["name"] == "Complex Rearrangement")
    
    print(f"Detailed profile for: {complex_case['name']} - {complex_case['description']}")
    profile_results = profile_operation(complex_case)
    print(profile_results)

def main():
    # Run full benchmark suite
    results = run_benchmarks()
    
    # Print tabulated results
    print_tabulated_results(results)
    
    # Generate performance plot
    plot_results(results)
    
    # Test cache effectiveness
    cache_effectiveness_test()
    
    # Test memory usage
    memory_usage_test()
    
    # Run detailed profiling
    profile_detailed_test_case()
    
    print("\nBenchmark completed!")

if __name__ == "__main__":
    main()
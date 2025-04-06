# Add to benchmark.py
import einops
import time
import numpy as np
from ein import rearrange

def compare_with_einops():
    """Compare performance with original einops library."""
    print("\n--- COMPARISON WITH EINOPS ---\n")
    
    # Test case: Reshape operation
    tensor = np.random.rand(100, 100, 100)
    pattern = "a b c -> a (b c)"
    
    # Time our implementation
    start_time = time.time()
    iterations = 100
    for _ in range(iterations):
        result_our = rearrange(tensor, pattern)
    our_time = time.time() - start_time
    print(f"Our implementation: {our_time/iterations*1000:.3f} ms")
    
    # Time einops
    start_time = time.time()
    for _ in range(iterations):
        result_einops = einops.rearrange(tensor, pattern)
    einops_time = time.time() - start_time
    print(f"Einops: {einops_time/iterations*1000:.3f} ms")
    print(f"Ratio (our/einops): {our_time/einops_time:.2f}x")
    
    # Verify results match
    np.testing.assert_array_equal(result_our, result_einops)
    print("Results match ✓")

# Add this call to run_benchmarks() or call directly
compare_with_einops()
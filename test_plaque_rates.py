#!/usr/bin/env python3
"""
Test script to verify the new plaque rate calculation feature.
"""

import sys
import os

# Add the parent directory to the path to import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from K_rates_extrapolate import calculate_k_rates, calculate_plaque_rates

def test_plaque_rates():
    """Test the plaque rate calculation with and without size multiplier."""
    
    print("Testing plaque rate calculation...")
    print("=" * 50)
    
    # Test with size multiplier enabled
    print("\n1. Testing with size multiplier enabled:")
    rates_with_multiplier = calculate_k_rates(
        baseline_ab40_plaque_rate=0.000005,
        baseline_ab42_plaque_rate=0.00005,
        enable_plaque_size_multiplier=True
    )
    
    # Check some specific rates
    test_sizes = [13, 16, 17, 20]
    for size in test_sizes:
        if size < 17:
            key_40 = f'k_O{size}_Plaque_forty'
            key_42 = f'k_O{size}_Plaque_fortytwo'
        else:
            key_40 = f'k_F{size}_Plaque_forty'
            key_42 = f'k_F{size}_Plaque_fortytwo'
        
        expected_40 = 0.000005 * size
        expected_42 = 0.00005 * size
        
        actual_40 = rates_with_multiplier[key_40]
        actual_42 = rates_with_multiplier[key_42]
        
        print(f"  Size {size}:")
        print(f"    AB40: expected {expected_40:.6f}, got {actual_40:.6f} {'✓' if abs(actual_40 - expected_40) < 1e-10 else '✗'}")
        print(f"    AB42: expected {expected_42:.6f}, got {actual_42:.6f} {'✓' if abs(actual_42 - expected_42) < 1e-10 else '✗'}")
    
    # Test with size multiplier disabled
    print("\n2. Testing with size multiplier disabled:")
    rates_without_multiplier = calculate_k_rates(
        baseline_ab40_plaque_rate=0.000005,
        baseline_ab42_plaque_rate=0.00005,
        enable_plaque_size_multiplier=False
    )
    
    for size in test_sizes:
        if size < 17:
            key_40 = f'k_O{size}_Plaque_forty'
            key_42 = f'k_O{size}_Plaque_fortytwo'
        else:
            key_40 = f'k_F{size}_Plaque_forty'
            key_42 = f'k_F{size}_Plaque_fortytwo'
        
        expected_40 = 0.000005
        expected_42 = 0.00005
        
        actual_40 = rates_without_multiplier[key_40]
        actual_42 = rates_without_multiplier[key_42]
        
        print(f"  Size {size}:")
        print(f"    AB40: expected {expected_40:.6f}, got {actual_40:.6f} {'✓' if abs(actual_40 - expected_40) < 1e-10 else '✗'}")
        print(f"    AB42: expected {expected_42:.6f}, got {actual_42:.6f} {'✓' if abs(actual_42 - expected_42) < 1e-10 else '✗'}")
    
    # Test the standalone calculate_plaque_rates function
    print("\n3. Testing standalone calculate_plaque_rates function:")
    plaque_rates = calculate_plaque_rates(0.000005, 0.00005, True)
    
    for size in test_sizes:
        if size < 17:
            key_40 = f'k_O{size}_Plaque_forty'
            key_42 = f'k_O{size}_Plaque_fortytwo'
        else:
            key_40 = f'k_F{size}_Plaque_forty'
            key_42 = f'k_F{size}_Plaque_fortytwo'
        
        expected_40 = 0.000005 * size
        expected_42 = 0.00005 * size
        
        actual_40 = plaque_rates[key_40]
        actual_42 = plaque_rates[key_42]
        
        print(f"  Size {size}:")
        print(f"    AB40: expected {expected_40:.6f}, got {actual_40:.6f} {'✓' if abs(actual_40 - expected_40) < 1e-10 else '✗'}")
        print(f"    AB42: expected {expected_42:.6f}, got {actual_42:.6f} {'✓' if abs(actual_42 - expected_42) < 1e-10 else '✗'}")
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    test_plaque_rates() 
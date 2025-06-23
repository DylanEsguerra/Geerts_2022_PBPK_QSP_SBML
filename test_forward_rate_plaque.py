#!/usr/bin/env python3
"""
Test script to verify the new forward rate multiplier for plaque rates.
"""

import sys
import os

# Add the parent directory to the path to import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from K_rates_extrapolate import calculate_k_rates, calculate_plaque_rates

def test_forward_rate_plaque():
    """Test the plaque rate calculation with forward rate multiplier."""
    
    print("Testing forward rate multiplier for plaque rates...")
    print("=" * 60)
    
    # Test with forward rate multiplier enabled
    print("\n1. Testing with forward rate multiplier enabled:")
    rates_with_multiplier = calculate_k_rates(
        baseline_ab40_plaque_rate=0.000005,
        baseline_ab42_plaque_rate=0.00005,
        enable_plaque_forward_rate_multiplier=True
    )
    
    # Check some specific rates
    test_cases = [
        ('k_O13_Plaque_forty', 'k_O12_O13_forty', 0.000005),
        ('k_O16_Plaque_forty', 'k_O15_O16_forty', 0.000005),
        ('k_F17_Plaque_forty', 'k_O16_F17_forty', 0.000005),
        ('k_F20_Plaque_forty', 'k_F19_F20_forty', 0.000005),
        ('k_O13_Plaque_fortytwo', 'k_O12_O13_fortytwo', 0.00005),
        ('k_O16_Plaque_fortytwo', 'k_O15_O16_fortytwo', 0.00005),
        ('k_F17_Plaque_fortytwo', 'k_O16_F17_fortytwo', 0.00005),
        ('k_F20_Plaque_fortytwo', 'k_F19_F20_fortytwo', 0.00005)
    ]
    
    print("\nVerifying forward rate multiplier calculation:")
    for plaque_rate_name, forward_rate_name, baseline_rate in test_cases:
        if plaque_rate_name in rates_with_multiplier and forward_rate_name in rates_with_multiplier:
            actual_plaque_rate = rates_with_multiplier[plaque_rate_name]
            forward_rate = rates_with_multiplier[forward_rate_name]
            expected_plaque_rate = baseline_rate * forward_rate
            
            if abs(actual_plaque_rate - expected_plaque_rate) < 1e-10:
                print(f"  ✓ {plaque_rate_name}: {actual_plaque_rate:.6f} (forward_rate = {forward_rate:.6f} × {baseline_rate:.6f})")
            else:
                print(f"  ✗ {plaque_rate_name}: {actual_plaque_rate:.6f} (expected {expected_plaque_rate:.6f})")
        else:
            missing = []
            if plaque_rate_name not in rates_with_multiplier:
                missing.append(plaque_rate_name)
            if forward_rate_name not in rates_with_multiplier:
                missing.append(forward_rate_name)
            print(f"  ✗ Missing parameters: {missing}")
    
    # Test with forward rate multiplier disabled
    print("\n2. Testing with forward rate multiplier disabled:")
    rates_without_multiplier = calculate_k_rates(
        baseline_ab40_plaque_rate=0.000005,
        baseline_ab42_plaque_rate=0.00005,
        enable_plaque_forward_rate_multiplier=False
    )
    
    for plaque_rate_name, forward_rate_name, baseline_rate in test_cases:
        if plaque_rate_name in rates_without_multiplier:
            actual_plaque_rate = rates_without_multiplier[plaque_rate_name]
            expected_plaque_rate = baseline_rate
            
            if abs(actual_plaque_rate - expected_plaque_rate) < 1e-10:
                print(f"  ✓ {plaque_rate_name}: {actual_plaque_rate:.6f} (baseline rate)")
            else:
                print(f"  ✗ {plaque_rate_name}: {actual_plaque_rate:.6f} (expected {expected_plaque_rate:.6f})")
        else:
            print(f"  ✗ {plaque_rate_name}: NOT FOUND")
    
    # Test the standalone calculate_plaque_rates function
    print("\n3. Testing standalone calculate_plaque_rates function:")
    
    # Create forward rates dictionaries for testing
    forward_rates_forty = {
        'k_O12_O13_forty': 0.001,
        'k_O15_O16_forty': 0.002,
        'k_O16_F17_forty': 0.003,
        'k_F19_F20_forty': 0.004
    }
    
    forward_rates_fortytwo = {
        'k_O12_O13_fortytwo': 0.005,
        'k_O15_O16_fortytwo': 0.006,
        'k_O16_F17_fortytwo': 0.007,
        'k_F19_F20_fortytwo': 0.008
    }
    
    plaque_rates = calculate_plaque_rates(
        0.000005,  # baseline_ab40_rate
        0.00005,   # baseline_ab42_rate
        forward_rates_forty,
        forward_rates_fortytwo,
        enable_forward_rate_multiplier=True
    )
    
    # Test specific cases
    test_standalone_cases = [
        ('k_O13_Plaque_forty', 0.000005 * 0.001),
        ('k_O16_Plaque_forty', 0.000005 * 0.002),
        ('k_F17_Plaque_forty', 0.000005 * 0.003),
        ('k_F20_Plaque_forty', 0.000005 * 0.004),
        ('k_O13_Plaque_fortytwo', 0.00005 * 0.005),
        ('k_O16_Plaque_fortytwo', 0.00005 * 0.006),
        ('k_F17_Plaque_fortytwo', 0.00005 * 0.007),
        ('k_F20_Plaque_fortytwo', 0.00005 * 0.008)
    ]
    
    for rate_name, expected_rate in test_standalone_cases:
        if rate_name in plaque_rates:
            actual_rate = plaque_rates[rate_name]
            if abs(actual_rate - expected_rate) < 1e-10:
                print(f"  ✓ {rate_name}: {actual_rate:.6f} (expected {expected_rate:.6f})")
            else:
                print(f"  ✗ {rate_name}: {actual_rate:.6f} (expected {expected_rate:.6f})")
        else:
            print(f"  ✗ {rate_name}: NOT FOUND")
    
    print("\n" + "=" * 60)
    print("Test completed!")

if __name__ == "__main__":
    test_forward_rate_plaque() 
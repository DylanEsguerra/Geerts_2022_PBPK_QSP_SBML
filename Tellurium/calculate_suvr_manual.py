#!/usr/bin/env python3
"""
Script to calculate SUVR manually without volume scaling factors.
The input values are assumed to already be in concentration units.
"""

import math

def calculate_suvr_manual(oligomer, fibril, plaque, c0=1.0, c1=2.52, c2=400000, c3=1.3, Hill=3.5):
    """
    SUVR = C0 + C1*(Ab42_oligo + Ab42_proto + C3*Ab42_plaque)^Hill/((Ab42_oligo + Ab42_proto + C3*Ab42_plaque)^Hill + C2^Hill)
    
    Args:
        oligomer: AB42 oligomer concentration
        fibril: AB42 protofibril concentration  
        plaque: AB42 plaque concentration
        c0, c1, c2, c3, Hill: SUVR formula parameters
        
    Returns:
        SUVR value
    """
    
    print(f"=== SUVR Calculation ===")
    print(f"Input values:")
    print(f"  Oligomer: {oligomer:.2e}")
    print(f"  Fibril: {fibril:.2e}")
    print(f"  Plaque: {plaque:.2e}")
    print(f"  C0: {c0}")
    print(f"  C1: {c1}")
    print(f"  C2: {c2}")
    print(f"  C3: {c3}")
    print(f"  Hill: {Hill}")
    print()
    
    # Calculate weighted sum
    weighted_sum = oligomer + fibril + c3 * plaque
    print(f"Weighted sum = oligomer + fibril + C3*plaque")
    print(f"  = {oligomer:.2e} + {fibril:.2e} + {c3}*{plaque:.2e}")
    print(f"  = {weighted_sum:.2e}")
    print()
    
    # Calculate SUVR
    numerator = c1 * (weighted_sum ** Hill)
    denominator = (weighted_sum ** Hill) + (c2 ** Hill)
    suvr = c0 + (numerator / denominator)
    
    print(f"SUVR = C0 + C1*(weighted_sum^Hill)/((weighted_sum^Hill) + C2^Hill)")
    print(f"  = {c0} + {c1}*({weighted_sum:.2e}^{Hill})/(({weighted_sum:.2e}^{Hill}) + {c2}^{Hill})")
    print(f"  = {c0} + {numerator:.2e}/({denominator:.2e})")
    print(f"  = {c0} + {numerator/denominator:.6f}")
    print(f"  = {suvr:.6f}")
    print()
    
    return suvr

def main():
    """Main function to run the SUVR calculation"""
    
    # Example values from user query
    oligomer = 1.2e4  # 1.2 x 10^4
    fibril = 7.0e4     # 7 x 10^4  
    plaque = 5100     # 5100
    
    print("Calculating SUVR for the given values...")
    print()
    
    suvr = calculate_suvr_manual(oligomer, fibril, plaque)
    
    print(f"Final SUVR: {suvr:.4f}")

if __name__ == "__main__":
    main()

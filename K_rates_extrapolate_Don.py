"""
Differences I noticed: 
-Plaque values are imporved because Don has the baseline * forward as part of reaction formula rather than having them calulated in here. 
-I need to validate that the rates are in the correct units (nM⁻¹h⁻¹, h⁻¹)
- This is fine. the defaults are the Garai Units but Don just passes the correct units in and also outputs the correct units. 

Rate Constant Extrapolation Module for Amyloid Beta Aggregation

This module calculates forward and backward rate constants for amyloid beta (Aβ) oligomerization
and fibrillization processes. It uses known experimental values for small oligomers (dimers and trimers)
to extrapolate rates for larger species using Hill function-based relationships.

Key Features:
- Extrapolates forward (association) and backward (dissociation) rates for oligomers up to size 24
- Handles both Aβ40 and Aβ42 species separately
- Uses experimentally determined rates for small oligomers as anchor points
- Implements Hill function-based extrapolation with asymptotic behavior
- Converts rates from literature units (M⁻¹s⁻¹, s⁻¹) to model units (nM⁻¹h⁻¹, h⁻¹)
- Calculates gain factors (forward/backward ratios) to assess aggregation propensity
- Calculates size-dependent plaque formation rates (multiplied by oligomer/fibril length)

The extrapolated rates are used by the oligomer and fibril modules to model the complete
aggregation cascade from monomers through oligomers to fibrils and plaques.

References:
- Original rate constants from literature for AB40 and AB42 dimer/trimer formation
- Hill function parameters tuned to match experimental observations of aggregation behavior
"""

# Calculation of forward and backward Rates 
# Note no plaque rates 
# Note no 24 to 12 breakdown
# import jax
# jax.config.update("jax_enable_x64", True)
# import jax.numpy as np
# import matplotlib.pyplot as plt
# from tabulate import tabulate

# Define the equations for forward (KF) and backward (KB) rate extrapolation
def extrapolate_kf(kf0, kf1, j, Asymp, HillA,rate_cutoff=None):
    """
    Extrapolate forward rate constants for higher order oligomers
    kf0: Rate constant for O1->O2 (known)
    kf1: Rate constant for O2->O3 (known)
    j: Oligomer size
    Asymp: Asymptotic value
    HillA: Hill coefficient for forward rates
    """
    if kf0 == kf1:
        kj_f = rate_cutoff
    else:
        KF = (kf1 - Asymp * kf1) / (kf0 - kf1)
        kj_f = (kf0 - Asymp * kf1) * (KF / (j**HillA + KF)) + Asymp * kf1
    if rate_cutoff is not None and kj_f < rate_cutoff:
        kj_f = rate_cutoff
    return kj_f

def extrapolate_kb(kb0, kb1, j, Asymp, HillB, rate_cutoff=None):
    """
    Extrapolate backward rate constants for higher order oligomers
    kb0: Rate constant for O2->O1 (known)
    kb1: Rate constant for O3->O2 (known)
    j: Oligomer size
    Asymp: Asymptotic value
    HillB: Hill coefficient for backward rates
    rate_cutoff: Minimum allowed rate, if None no cutoff is applied
    """
    KB = (kb1 - Asymp * kb1) / (kb0 - kb1)
    kj_b = (kb0 - Asymp * kb1) * (KB / (j**HillB + KB)) + Asymp * kb1
    
    # Apply rate cutoff if specified
    if rate_cutoff is not None and kj_b < rate_cutoff:
        kj_b = rate_cutoff
        
    return kj_b




def convert_forward_rate(rate_M_s):
    """Convert from M⁻¹s⁻¹ to nM⁻¹h⁻¹"""
    # 1 M⁻¹s⁻¹ = 3.6 × 10⁻⁶ nM⁻¹h⁻¹
    return rate_M_s * 3.6e-6

def convert_backward_rate(rate_s):
    """Convert from s⁻¹ to h⁻¹"""
    # 1 s⁻¹ = 3600 h⁻¹
    return rate_s * 3600

def calculate_k_rates(
    # Original rates from literature (M⁻¹s⁻¹ for forward, s⁻¹ for backward)
    # original_kf0_forty=0.5 * 10**2,  # AB40 monomer to dimer
    k_O1_O2_AB42_ISF=9.9 * 10**2,  # AB42 monomer to dimer
    # original_kf1_forty=20.0,  # AB40 dimer to trimer
    k_O2_O3_AB42_ISF=38.0,  # AB42 dimer to trimer
    # original_kb0_forty=2.7 * 10**-3,  # AB40 dimer to monomer
    k_O2_O1_AB42_ISF=12.7 * 10**-3,  # AB42 dimer to monomer
    # original_kb1_forty=0.00001 / 3600,  # AB40 trimer to dimer
    k_O3_O2_AB42_ISF=0.00001 / 3600,  # AB42 trimer to dimer
    
    # Hill coefficients and asymptotic values
    # forAsymp40=0.3,  # Asymptotic value for AB40 forward rates
    
    
):
    """
    Calculate forward and backward rates for both AB40 and AB42 oligomers
    using provided parameters based on known values from literature
    
    Parameters:
    -----------
    original_kf0_forty: float
        AB40 monomer to dimer forward rate (M⁻¹s⁻¹)
    original_kf0_fortytwo: float
        AB42 monomer to dimer forward rate (M⁻¹s⁻¹)
    k_O1_O2_AB42_ISF: float
        AB40 dimer to trimer forward rate (M⁻¹s⁻¹)
    k_O2_O3_AB42_ISF: float
        AB42 dimer to trimer forward rate (M⁻¹s⁻¹)
    k_O2_O1_AB42_ISF: float
        AB40 dimer to monomer backward rate (s⁻¹)
    k_O3_O2_AB42_ISF: float
        AB42 dimer to monomer backward rate (s⁻¹)
    k_O3_O2_AB42_ISF: float
        AB40 trimer to dimer backward rate (s⁻¹)
    k_O3_O2_AB42_ISF: float
        AB42 trimer to dimer backward rate (s⁻¹)
    forAsymp40: float
        Asymptotic value for AB40 forward rates
    forAsymp42: float
        Asymptotic value for AB42 forward rates
    backAsymp40: float
        Asymptotic value for AB40 backward rates
    backAsymp42: float
        Asymptotic value for AB42 backward rates
    forHill40: float
        Hill coefficient for AB40 forward rates
    forHill42: float
        Hill coefficient for AB42 forward rates
    BackHill40: float
        Hill coefficient for AB40 backward rates
    BackHill42: float
        Hill coefficient for AB42 backward rates
    rate_cutoff: float
        Minimum allowed rate for backward reactions

    Returns:
    --------
    dict
        Dictionary containing all extrapolated rate constants including plaque rates
    """
    forAsymp42=2.0  # Asymptotic value for AB42 forward rates
    # backAsymp40=0.3,  # Asymptotic value for AB40 backward rates
    backAsymp42=2.0  # Asymptotic value for AB42 backward rates
    # forHill40=2.0,    # Hill coefficient for AB40 forward rates
    forHill42=3.0   # Hill coefficient for AB42 forward rates
    # BackHill40=2.5,   # Hill coefficient for AB40 backward rates
    BackHill42=3.0   # Hill coefficient for AB42 backward rates
    
    # Rate cutoff
    rate_cutoff=0.00001
    # Convert rates to appropriate units
    # kf0_forty = convert_forward_rate(original_kf0_forty)  
    # kb0_forty = convert_backward_rate(original_kb0_forty)
    # kf0_fortytwo = convert_forward_rate(original_kf0_fortytwo)
    # kb0_fortytwo = convert_backward_rate(original_kb0_fortytwo)
    # # kf1_forty = convert_forward_rate(original_kf1_forty)
    # # kb1_forty = convert_backward_rate(original_kb1_forty)
    # kf1_fortytwo = convert_forward_rate(original_kf1_fortytwo)
    # kb1_fortytwo = convert_backward_rate(original_kb1_fortytwo)
    
    # # Create and print conversion table
    # #print("\nRate Constant Unit Conversion:")
    # table_data = [
    #     ["k+12 (Aβ40)", f"{original_kf0_forty:.1f} M⁻¹s⁻¹", f"{kf0_forty:.6f} nM⁻¹h⁻¹"],
    #     ["k-12 (Aβ40)", f"{original_kb0_forty:.3f} s⁻¹", f"{kb0_forty:.6f} h⁻¹"],
    #     ["k+23 (Aβ40)", f"{original_kf1_forty:.1f} M⁻¹s⁻¹", f"{kf1_forty:.6f} nM⁻¹h⁻¹"],
    #     ["k-23 (Aβ40)", f"{original_kb1_forty:.3f} s⁻¹", f"{kb1_forty:.6f} h⁻¹"],
    #     ["k+12 (Aβ42)", f"{original_kf0_fortytwo:.1f} M⁻¹s⁻¹", f"{kf0_fortytwo:.6f} nM⁻¹h⁻¹"],
    #     ["k-12 (Aβ42)", f"{original_kb0_fortytwo:.3f} s⁻¹", f"{kb0_fortytwo:.6f} h⁻¹"],
    #     ["k+23 (Aβ42)", f"{original_kf1_fortytwo:.1f} M⁻¹s⁻¹", f"{kf1_fortytwo:.6f} nM⁻¹h⁻¹"],
    #     ["k-23 (Aβ42)", f"{original_kb1_fortytwo:.3f} s⁻¹", f"{kb1_fortytwo:.6f} h⁻¹"]
    # ]
    # headers = ["Rate Constant", "Original Value", "Converted Value"]
    #print(tabulate(table_data, headers, tablefmt="grid"))
    
    # Generate oligomer sizes from 4 to 24
    oligomer_sizes = list(range(4, 25))

    # Calculate rates for each oligomer size
    # kf_forty = [extrapolate_kf(kf0_forty, kf1_forty, size, forAsymp40, forHill40) for size in oligomer_sizes]
    # kb_forty = [extrapolate_kb(kb0_forty, kb1_forty, size, backAsymp40, BackHill40, rate_cutoff) for size in oligomer_sizes]
    kf_fortytwo = [extrapolate_kf(k_O1_O2_AB42_ISF, k_O2_O3_AB42_ISF, size, forAsymp42, forHill42, rate_cutoff) for size in oligomer_sizes]
    kb_fortytwo = [extrapolate_kb(k_O2_O1_AB42_ISF, k_O3_O2_AB42_ISF, size, backAsymp42, BackHill42, rate_cutoff) for size in oligomer_sizes]
    
    # Create dictionary to store the rates with proper naming convention
    rates = {}
    
    # Store rates with appropriate prefixes (O for oligomers, F for fibrils)
    for i, size in enumerate(oligomer_sizes):
        
        # Oligomer rates (size < 17)
        # rates[f'k_O{size-1}_O{size}_AB40_ISF'] = kf_forty[i]
        # rates[f'k_O{size}_O{size-1}_AB40_ISF'] = kb_forty[i]
        rates[f'k_O{size-1}_O{size}_AB42_ISF'] = kf_fortytwo[i]
        rates[f'k_O{size}_O{size-1}_AB42_ISF'] = kb_fortytwo[i]

    
    return rates

def calculate_gain_factors(rates):
    """
    Calculate the gain factor (ratio of forward to backward rates) for each oligomer size
    
    Parameters:
    rates: Dictionary of rate constants
    
    Returns:
    Dictionary of gain factors for AB40 and AB42
    """
    gain_factors = {}
    
    # Calculate gain factors for sizes 4 to 24
    for size in range(4, 25):
            # Oligomer gain factors
        # gain_factors[f'gain_O{size}_AB40_ISF'] = rates[f'k_O{size-1}_O{size}_AB40_ISF'] / rates[f'k_O{size}_O{size-1}_AB40_ISF']
        gain_factors[f'gain_O{size}_AB42_ISF'] = rates[f'k_O{size-1}_O{size}_AB42_ISF'] / rates[f'k_O{size}_O{size-1}_AB42_ISF']

    return gain_factors

if __name__ == "__main__":
    # Example usage with default parameters
    rates = calculate_k_rates()
    print(rates)
    # Calculate gain factors
    gain_factors = calculate_gain_factors(rates)
    print(gain_factors)
    # Print the calculated rates
    #print("\nCalculated rate constants:")
    #for size in range(4, 25):
        #print(f"\nFor size {size}:")
        #if size < 17:
            # Oligomer rates
            #print(f"AB40 forward (O{size-1}->O{size}): {rates[f'k_O{size-1}_O{size}_forty']:.3e} nM⁻¹h⁻¹")
            #print(f"AB40 backward (O{size}->O{size-1}): {rates[f'k_O{size}_O{size-1}_forty']:.3e} h⁻¹")
            #print(f"AB40 gain factor: {gain_factors[f'gain_O{size}_forty']:.3e}")
            #print(f"AB42 forward (O{size-1}->O{size}): {rates[f'k_O{size-1}_O{size}_fortytwo']:.3e} nM⁻¹h⁻¹")
            #print(f"AB42 backward (O{size}->O{size-1}): {rates[f'k_O{size}_O{size-1}_fortytwo']:.3e} h⁻¹")
            #print(f"AB42 gain factor: {gain_factors[f'gain_O{size}_fortytwo']:.3e}")
        #elif size == 17:
            # Transition between oligomer and fibril
            #print(f"AB40 forward (O{size-1}->F{size}): {rates[f'k_O{size-1}_F{size}_forty']:.3e} nM⁻¹h⁻¹")
            #print(f"AB40 backward (F{size}->O{size-1}): {rates[f'k_F{size}_O{size-1}_forty']:.3e} h⁻¹")
            #print(f"AB40 gain factor: {gain_factors[f'gain_F{size}_forty']:.3e}")
            #print(f"AB42 forward (O{size-1}->F{size}): {rates[f'k_O{size-1}_F{size}_fortytwo']:.3e} nM⁻¹h⁻¹")
            #print(f"AB42 backward (F{size}->O{size-1}): {rates[f'k_F{size}_O{size-1}_fortytwo']:.3e} h⁻¹")
            #print(f"AB42 gain factor: {gain_factors[f'gain_F{size}_fortytwo']:.3e}")
        #else:
            # Fibril rates
            #print(f"AB40 forward (F{size-1}->F{size}): {rates[f'k_F{size-1}_F{size}_forty']:.3e} nM⁻¹h⁻¹")
            #print(f"AB40 backward (F{size}->F{size-1}): {rates[f'k_F{size}_F{size-1}_forty']:.3e} h⁻¹")
            #print(f"AB40 gain factor: {gain_factors[f'gain_F{size}_forty']:.3e}")
            #print(f"AB42 forward (F{size-1}->F{size}): {rates[f'k_F{size-1}_F{size}_fortytwo']:.3e} nM⁻¹h⁻¹")
            #print(f"AB42 backward (F{size}->F{size-1}): {rates[f'k_F{size}_F{size-1}_fortytwo']:.3e} h⁻¹")
            #print(f"AB42 gain factor: {gain_factors[f'gain_F{size}_fortytwo']:.3e}")
        
    
    # # Optional: Plot the rates to visualize the extrapolation
    # sizes = list(range(4, 25))
    
    # # Create a more compact figure
    # plt.figure(figsize=(10, 6))
    # plt.rcParams.update({'font.size': 10})
    
    # # Plot AB40 rates
    # plt.subplot(2, 1, 1)
    
    # # Plot AB40 rates with appropriate naming
    # forward_rates_40 = []
    # backward_rates_40 = []
    # for size in sizes:
    #     if size < 17:
    #         forward_rates_40.append(rates[f'k_O{size-1}_O{size}_forty'])
    #         backward_rates_40.append(rates[f'k_O{size}_O{size-1}_forty'])
    #     elif size == 17:
    #         forward_rates_40.append(rates[f'k_O{size-1}_F{size}_forty'])
    #         backward_rates_40.append(rates[f'k_F{size}_O{size-1}_forty'])
    #     else:
    #         forward_rates_40.append(rates[f'k_F{size-1}_F{size}_forty'])
    #         backward_rates_40.append(rates[f'k_F{size}_F{size-1}_forty'])
    
    # plt.plot(sizes, forward_rates_40, 'b-', linewidth=2.5, label='Forward')
    # plt.plot(sizes, backward_rates_40, 'b--', linewidth=2.5, label='Backward')
    # plt.axvline(x=17, color='k', linestyle=':', linewidth=1.5, label='Transition')
    # plt.yscale('log')
    # plt.ylabel('Rate (nM⁻¹h⁻¹ or h⁻¹)', fontsize=11)
    # plt.title('AB40 Rate Constants', fontsize=12)
    # plt.legend(fontsize=10, loc='upper right')
    # plt.grid(True, alpha=0.3)
    
    # # Plot AB42 rates
    # plt.subplot(2, 1, 2)
    
    # # Plot AB42 rates with appropriate naming
    # forward_rates_42 = []
    # backward_rates_42 = []
    # for size in sizes:
    #     if size < 17:
    #         forward_rates_42.append(rates[f'k_O{size-1}_O{size}_fortytwo'])
    #         backward_rates_42.append(rates[f'k_O{size}_O{size-1}_fortytwo'])
    #     elif size == 17:
    #         forward_rates_42.append(rates[f'k_O{size-1}_F{size}_fortytwo'])
    #         backward_rates_42.append(rates[f'k_F{size}_O{size-1}_fortytwo'])
    #     else:
    #         forward_rates_42.append(rates[f'k_F{size-1}_F{size}_fortytwo'])
    #         backward_rates_42.append(rates[f'k_F{size}_F{size-1}_fortytwo'])
    
    # plt.plot(sizes, forward_rates_42, 'r-', linewidth=2.5, label='Forward')
    # plt.plot(sizes, backward_rates_42, 'r--', linewidth=2.5, label='Backward')
    # plt.axvline(x=17, color='k', linestyle=':', linewidth=1.5, label='Transition')
    # plt.yscale('log')
    # plt.xlabel('Size', fontsize=11)
    # plt.ylabel('Rate (nM⁻¹h⁻¹ or h⁻¹)', fontsize=11)
    # plt.title('AB42 Rate Constants', fontsize=12)
    # plt.legend(fontsize=10, loc='upper right')
    # plt.grid(True, alpha=0.3)
    
    # plt.tight_layout(pad=1.5)
    # plt.savefig('rate_extrapolation.png', dpi=300, bbox_inches='tight')
    # plt.show()
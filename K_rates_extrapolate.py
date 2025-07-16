"""
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

import matplotlib.pyplot as plt

# Define the equations for forward (KF) and backward (KB) rate extrapolation
def extrapolate_kf(kf0, kf1, j, Asymp, HillA):
    """
    Extrapolate forward rate constants for higher order oligomers
    kf0: Rate constant for O1->O2 (known)
    kf1: Rate constant for O2->O3 (known)
    j: Oligomer size
    Asymp: Asymptotic value
    HillA: Hill coefficient for forward rates
    """
    KF = (kf1 - Asymp * kf1) / (kf0 - kf1)
    kj_f = (kf0 - Asymp * kf1) * (KF / (j**HillA + KF)) + Asymp * kf1
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


def calculate_plaque_rates(baseline_ab40_rate, baseline_ab42_rate, forward_rates_forty, forward_rates_fortytwo, enable_forward_rate_multiplier=True):
    """
    Calculate plaque formation rates for oligomers and fibrils.
    If enable_forward_rate_multiplier is True, rates are multiplied by the forward rate for that aggregate.
    
    Args:
        baseline_ab40_rate: Baseline plaque formation rate for AB40
        baseline_ab42_rate: Baseline plaque formation rate for AB42
        forward_rates_forty: Dictionary of forward rates for AB40 aggregates
        forward_rates_fortytwo: Dictionary of forward rates for AB42 aggregates
        enable_forward_rate_multiplier: If True, multiply rates by forward rate for that aggregate
    
    Returns:
        Dictionary containing plaque formation rates
    """
    plaque_rates = {}
    
    # Generate oligomer sizes from 13 to 16 (oligomers that can form plaques)
    oligomer_sizes = list(range(13, 17))
    
    # Generate fibril sizes from 17 to 20 (fibrils that can form plaques)
    fibril_sizes = list(range(17, 21))
    
    # Calculate plaque rates for oligomers
    for size in oligomer_sizes:
        if enable_forward_rate_multiplier:
            # Get the forward rate for this oligomer size
            forward_rate_key_40 = f'k_O{size-1}_O{size}_forty'
            forward_rate_key_42 = f'k_O{size-1}_O{size}_fortytwo'
            
            if forward_rate_key_40 in forward_rates_forty and forward_rate_key_42 in forward_rates_fortytwo:
                # Multiply baseline by forward rate
                plaque_rates[f'k_O{size}_Plaque_forty'] = baseline_ab40_rate * forward_rates_forty[forward_rate_key_40]
                plaque_rates[f'k_O{size}_Plaque_fortytwo'] = baseline_ab42_rate * forward_rates_fortytwo[forward_rate_key_42]
            else:
                # Fallback to baseline rates if forward rates not found
                print(f"Warning: Forward rates not found for oligomer size {size}, using baseline rates")
                plaque_rates[f'k_O{size}_Plaque_forty'] = baseline_ab40_rate
                plaque_rates[f'k_O{size}_Plaque_fortytwo'] = baseline_ab42_rate
        else:
            # Use baseline rates
            plaque_rates[f'k_O{size}_Plaque_forty'] = baseline_ab40_rate
            plaque_rates[f'k_O{size}_Plaque_fortytwo'] = baseline_ab42_rate
    
    # Calculate plaque rates for fibrils
    for size in fibril_sizes:
        if enable_forward_rate_multiplier:
            # Get the forward rate for this fibril size
            if size == 17:
                # Special case: transition from oligomer to fibril
                forward_rate_key_40 = f'k_O{size-1}_F{size}_forty'
                forward_rate_key_42 = f'k_O{size-1}_F{size}_fortytwo'
            else:
                # Normal fibril growth
                forward_rate_key_40 = f'k_F{size-1}_F{size}_forty'
                forward_rate_key_42 = f'k_F{size-1}_F{size}_fortytwo'
            
            if forward_rate_key_40 in forward_rates_forty and forward_rate_key_42 in forward_rates_fortytwo:
                # Multiply baseline by forward rate
                plaque_rates[f'k_F{size}_Plaque_forty'] = baseline_ab40_rate * forward_rates_forty[forward_rate_key_40]
                plaque_rates[f'k_F{size}_Plaque_fortytwo'] = baseline_ab42_rate * forward_rates_fortytwo[forward_rate_key_42]
            else:
                # Fallback to baseline rates if forward rates not found
                print(f"Warning: Forward rates not found for fibril size {size}, using baseline rates")
                plaque_rates[f'k_F{size}_Plaque_forty'] = baseline_ab40_rate
                plaque_rates[f'k_F{size}_Plaque_fortytwo'] = baseline_ab42_rate
        else:
            # Use baseline rates
            plaque_rates[f'k_F{size}_Plaque_forty'] = baseline_ab40_rate
            plaque_rates[f'k_F{size}_Plaque_fortytwo'] = baseline_ab42_rate
    
    return plaque_rates

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
    original_kf0_forty=0.5 * 10**2,  # AB40 monomer to dimer
    original_kf0_fortytwo=9.9 * 10**2,  # AB42 monomer to dimer
    original_kf1_forty=20.0,  # AB40 dimer to trimer
    original_kf1_fortytwo=38.0,  # AB42 dimer to trimer
    original_kb0_forty=2.7 * 10**-3,  # AB40 dimer to monomer
    original_kb0_fortytwo=12.7 * 10**-3,  # AB42 dimer to monomer
    original_kb1_forty=0.00001 / 3600,  # AB40 trimer to dimer
    original_kb1_fortytwo=0.00001 / 3600,  # AB42 trimer to dimer
    
    # Hill coefficients and asymptotic values
    forAsymp40=0.3,  # Asymptotic value for AB40 forward rates
    forAsymp42=2.0,  # Asymptotic value for AB42 forward rates
    backAsymp40=0.3,  # Asymptotic value for AB40 backward rates
    backAsymp42=2.0,  # Asymptotic value for AB42 backward rates
    forHill40=2.0,    # Hill coefficient for AB40 forward rates
    forHill42=3.0,    # Hill coefficient for AB42 forward rates
    BackHill40=2.5,   # Hill coefficient for AB40 backward rates
    BackHill42=3.0,   # Hill coefficient for AB42 backward rates
    
    # Rate cutoff
    rate_cutoff=0.00001,
    
    # Plaque formation parameters
    baseline_ab40_plaque_rate=0.000005,  # Baseline plaque formation rate for AB40
    baseline_ab42_plaque_rate=0.00005,   # Baseline plaque formation rate for AB42
    enable_plaque_forward_rate_multiplier=True   # If True, multiply plaque rates by forward rate for that aggregate
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
    original_kf1_forty: float
        AB40 dimer to trimer forward rate (M⁻¹s⁻¹)
    original_kf1_fortytwo: float
        AB42 dimer to trimer forward rate (M⁻¹s⁻¹)
    original_kb0_forty: float
        AB40 dimer to monomer backward rate (s⁻¹)
    original_kb0_fortytwo: float
        AB42 dimer to monomer backward rate (s⁻¹)
    original_kb1_forty: float
        AB40 trimer to dimer backward rate (s⁻¹)
    original_kb1_fortytwo: float
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
    baseline_ab40_plaque_rate: float
        Baseline plaque formation rate for AB40
    baseline_ab42_plaque_rate: float
        Baseline plaque formation rate for AB42
    enable_plaque_forward_rate_multiplier: bool
        If True, multiply plaque rates by forward rate for that aggregate
    
    Returns:
    --------
    dict
        Dictionary containing all extrapolated rate constants including plaque rates
    """
    # Convert rates to appropriate units
    kf0_forty = convert_forward_rate(original_kf0_forty)  
    kb0_forty = convert_backward_rate(original_kb0_forty)
    kf0_fortytwo = convert_forward_rate(original_kf0_fortytwo)
    kb0_fortytwo = convert_backward_rate(original_kb0_fortytwo)
    kf1_forty = convert_forward_rate(original_kf1_forty)
    kb1_forty = convert_backward_rate(original_kb1_forty)
    kf1_fortytwo = convert_forward_rate(original_kf1_fortytwo)
    kb1_fortytwo = convert_backward_rate(original_kb1_fortytwo)
    '''
    # Create and print conversion table
    print("\nRate Constant Unit Conversion:")
    table_data = [
        ["k+12 (Aβ40)", f"{original_kf0_forty:.1f} M⁻¹s⁻¹", f"{kf0_forty:.6f} nM⁻¹h⁻¹"],
        ["k-12 (Aβ40)", f"{original_kb0_forty:.3f} s⁻¹", f"{kb0_forty:.6f} h⁻¹"],
        ["k+23 (Aβ40)", f"{original_kf1_forty:.1f} M⁻¹s⁻¹", f"{kf1_forty:.6f} nM⁻¹h⁻¹"],
        ["k-23 (Aβ40)", f"{original_kb1_forty:.3f} s⁻¹", f"{kb1_forty:.6f} h⁻¹"],
        ["k+12 (Aβ42)", f"{original_kf0_fortytwo:.1f} M⁻¹s⁻¹", f"{kf0_fortytwo:.6f} nM⁻¹h⁻¹"],
        ["k-12 (Aβ42)", f"{original_kb0_fortytwo:.3f} s⁻¹", f"{kb0_fortytwo:.6f} h⁻¹"],
        ["k+23 (Aβ42)", f"{original_kf1_fortytwo:.1f} M⁻¹s⁻¹", f"{kf1_fortytwo:.6f} nM⁻¹h⁻¹"],
        ["k-23 (Aβ42)", f"{original_kb1_fortytwo:.3f} s⁻¹", f"{kb1_fortytwo:.6f} h⁻¹"]
    ]
    headers = ["Rate Constant", "Original Value", "Converted Value"]
    print(tabulate(table_data, headers, tablefmt="grid"))
    '''
    # Generate oligomer sizes from 4 to 24
    oligomer_sizes = list(range(4, 25))

    # Calculate rates for each oligomer size
    kf_forty = [extrapolate_kf(kf0_forty, kf1_forty, size, forAsymp40, forHill40) for size in oligomer_sizes]
    kb_forty = [extrapolate_kb(kb0_forty, kb1_forty, size, backAsymp40, BackHill40, rate_cutoff) for size in oligomer_sizes]
    kf_fortytwo = [extrapolate_kf(kf0_fortytwo, kf1_fortytwo, size, forAsymp42, forHill42) for size in oligomer_sizes]
    kb_fortytwo = [extrapolate_kb(kb0_fortytwo, kb1_fortytwo, size, backAsymp42, BackHill42, rate_cutoff) for size in oligomer_sizes]
    
    # Create dictionary to store the rates with proper naming convention
    rates = {}
    
    # Store rates with appropriate prefixes (O for oligomers, F for fibrils)
    for i, size in enumerate(oligomer_sizes):
        if size < 17:
            # Oligomer rates (size < 17)
            rates[f'k_O{size-1}_O{size}_forty'] = kf_forty[i]
            rates[f'k_O{size}_O{size-1}_forty'] = kb_forty[i]
            rates[f'k_O{size-1}_O{size}_fortytwo'] = kf_fortytwo[i]
            rates[f'k_O{size}_O{size-1}_fortytwo'] = kb_fortytwo[i]
        elif size == 17:
            # Special case: transition between oligomer and fibril
            rates[f'k_O{size-1}_F{size}_forty'] = kf_forty[i]
            rates[f'k_F{size}_O{size-1}_forty'] = kb_forty[i]
            rates[f'k_O{size-1}_F{size}_fortytwo'] = kf_fortytwo[i]
            rates[f'k_F{size}_O{size-1}_fortytwo'] = kb_fortytwo[i]
        else:
            # Fibril rates (size >= 17)
            rates[f'k_F{size-1}_F{size}_forty'] = kf_forty[i]
            rates[f'k_F{size}_F{size-1}_forty'] = kb_forty[i]
            rates[f'k_F{size-1}_F{size}_fortytwo'] = kf_fortytwo[i]
            rates[f'k_F{size}_F{size-1}_fortytwo'] = kb_fortytwo[i]
    
    # Create separate dictionaries for forward rates only (needed for plaque calculation)
    forward_rates_forty = {}
    forward_rates_fortytwo = {}
    
    for i, size in enumerate(oligomer_sizes):
        if size < 17:
            # Oligomer forward rates
            forward_rates_forty[f'k_O{size-1}_O{size}_forty'] = kf_forty[i]
            forward_rates_fortytwo[f'k_O{size-1}_O{size}_fortytwo'] = kf_fortytwo[i]
        elif size == 17:
            # Transition forward rates
            forward_rates_forty[f'k_O{size-1}_F{size}_forty'] = kf_forty[i]
            forward_rates_fortytwo[f'k_O{size-1}_F{size}_fortytwo'] = kf_fortytwo[i]
        else:
            # Fibril forward rates
            forward_rates_forty[f'k_F{size-1}_F{size}_forty'] = kf_forty[i]
            forward_rates_fortytwo[f'k_F{size-1}_F{size}_fortytwo'] = kf_fortytwo[i]
    
    # Calculate and add plaque formation rates
    plaque_rates = calculate_plaque_rates(
        baseline_ab40_plaque_rate, 
        baseline_ab42_plaque_rate, 
        forward_rates_forty,
        forward_rates_fortytwo,
        enable_forward_rate_multiplier=enable_plaque_forward_rate_multiplier
    )
    rates.update(plaque_rates)
    
    # Print plaque rate information if forward rate multiplier is enabled
    if enable_plaque_forward_rate_multiplier:
        #print(f"\nPlaque formation rates with forward rate multiplier enabled:")
        #print(f"Baseline AB40 rate: {baseline_ab40_plaque_rate:.6f} L/(nM·h)")
        #print(f"Baseline AB42 rate: {baseline_ab42_plaque_rate:.6f} L/(nM·h)")
        #print("\nExample plaque rates (rate = baseline × forward_rate):")
        example_sizes = [13, 16, 17, 20]
        for size in example_sizes:
            if size < 17:
                key_40 = f'k_O{size}_Plaque_forty'
                key_42 = f'k_O{size}_Plaque_fortytwo'
                forward_key_40 = f'k_O{size-1}_O{size}_forty'
                forward_key_42 = f'k_O{size-1}_O{size}_fortytwo'
            else:
                key_40 = f'k_F{size}_Plaque_forty'
                key_42 = f'k_F{size}_Plaque_fortytwo'
                forward_key_40 = f'k_F{size-1}_F{size}_forty'
                forward_key_42 = f'k_F{size-1}_F{size}_fortytwo'
            
            if key_40 in plaque_rates and key_42 in plaque_rates:
                forward_rate_40 = forward_rates_forty.get(forward_key_40, 0)
                forward_rate_42 = forward_rates_fortytwo.get(forward_key_42, 0)
                #print(f"  Size {size}: AB40 = {plaque_rates[key_40]:.6f} (forward_rate = {forward_rate_40:.6f}), AB42 = {plaque_rates[key_42]:.6f} (forward_rate = {forward_rate_42:.6f})")
    #else:
        #print(f"\nPlaque formation rates using baseline values (no forward rate multiplier):")
        #print(f"AB40 rate: {baseline_ab40_plaque_rate:.6f} L/(nM·h)")
        #print(f"AB42 rate: {baseline_ab42_plaque_rate:.6f} L/(nM·h)")
    
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
        if size < 17:
            # Oligomer gain factors
            gain_factors[f'gain_O{size}_forty'] = rates[f'k_O{size-1}_O{size}_forty'] / rates[f'k_O{size}_O{size-1}_forty']
            gain_factors[f'gain_O{size}_fortytwo'] = rates[f'k_O{size-1}_O{size}_fortytwo'] / rates[f'k_O{size}_O{size-1}_fortytwo']
        elif size == 17:
            # Transition gain factors
            gain_factors[f'gain_F{size}_forty'] = rates[f'k_O{size-1}_F{size}_forty'] / rates[f'k_F{size}_O{size-1}_forty']
            gain_factors[f'gain_F{size}_fortytwo'] = rates[f'k_O{size-1}_F{size}_fortytwo'] / rates[f'k_F{size}_O{size-1}_fortytwo']
        else:
            # Fibril gain factors
            gain_factors[f'gain_F{size}_forty'] = rates[f'k_F{size-1}_F{size}_forty'] / rates[f'k_F{size}_F{size-1}_forty']
            gain_factors[f'gain_F{size}_fortytwo'] = rates[f'k_F{size-1}_F{size}_fortytwo'] / rates[f'k_F{size}_F{size-1}_fortytwo']
    
    return gain_factors

if __name__ == "__main__":
    # Example usage with default parameters
    rates = calculate_k_rates()
    
    # Calculate gain factors
    gain_factors = calculate_gain_factors(rates)
    
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
        
    
    # Optional: Plot the rates to visualize the extrapolation
    sizes = list(range(4, 25))
    
    # Create a more compact figure
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 10})
    
    # Plot AB40 rates
    plt.subplot(2, 1, 1)
    
    # Plot AB40 rates with appropriate naming
    forward_rates_40 = []
    backward_rates_40 = []
    for size in sizes:
        if size < 17:
            forward_rates_40.append(rates[f'k_O{size-1}_O{size}_forty'])
            backward_rates_40.append(rates[f'k_O{size}_O{size-1}_forty'])
        elif size == 17:
            forward_rates_40.append(rates[f'k_O{size-1}_F{size}_forty'])
            backward_rates_40.append(rates[f'k_F{size}_O{size-1}_forty'])
        else:
            forward_rates_40.append(rates[f'k_F{size-1}_F{size}_forty'])
            backward_rates_40.append(rates[f'k_F{size}_F{size-1}_forty'])
    
    plt.plot(sizes, forward_rates_40, 'b-', linewidth=2.5, label='Forward')
    plt.plot(sizes, backward_rates_40, 'b--', linewidth=2.5, label='Backward')
    plt.axvline(x=17, color='k', linestyle=':', linewidth=1.5, label='Transition')
    plt.yscale('log')
    plt.ylabel('Rate (nM⁻¹h⁻¹ or h⁻¹)', fontsize=11)
    plt.title('AB40 Rate Constants', fontsize=12)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Plot AB42 rates
    plt.subplot(2, 1, 2)
    
    # Plot AB42 rates with appropriate naming
    forward_rates_42 = []
    backward_rates_42 = []
    for size in sizes:
        if size < 17:
            forward_rates_42.append(rates[f'k_O{size-1}_O{size}_fortytwo'])
            backward_rates_42.append(rates[f'k_O{size}_O{size-1}_fortytwo'])
        elif size == 17:
            forward_rates_42.append(rates[f'k_O{size-1}_F{size}_fortytwo'])
            backward_rates_42.append(rates[f'k_F{size}_O{size-1}_fortytwo'])
        else:
            forward_rates_42.append(rates[f'k_F{size-1}_F{size}_fortytwo'])
            backward_rates_42.append(rates[f'k_F{size}_F{size-1}_fortytwo'])
    
    plt.plot(sizes, forward_rates_42, 'r-', linewidth=2.5, label='Forward')
    plt.plot(sizes, backward_rates_42, 'r--', linewidth=2.5, label='Backward')
    plt.axvline(x=17, color='k', linestyle=':', linewidth=1.5, label='Transition')
    plt.yscale('log')
    plt.xlabel('Size', fontsize=11)
    plt.ylabel('Rate (nM⁻¹h⁻¹ or h⁻¹)', fontsize=11)
    plt.title('AB42 Rate Constants', fontsize=12)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=1.5)
    plt.savefig('rate_extrapolation.png', dpi=300, bbox_inches='tight')
    plt.show()

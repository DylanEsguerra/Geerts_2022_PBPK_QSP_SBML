import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def load_solution(csv_path, scale=1.0):
    df = pd.read_csv(csv_path)
    # Detect time column
    if 'time' in df.columns:
        time = df['time'].values
        years = time / (24 * 365)  # convert hours to years
        df = df.drop(columns=['time'])
    elif 'time_years' in df.columns:
        years = df['time_years'].values
        df = df.drop(columns=['time_years'])
    else:
        raise ValueError("No recognized time column in CSV")
    if scale != 1.0:
        df = df / scale
    species_data = df.values
    y_indexes = {name: idx for idx, name in enumerate(df.columns)}
    class Model:
        def __init__(self, y_indexes):
            self.y_indexes = y_indexes
    class Solution:
        def __init__(self, ts, ys):
            self.ts = ts
            self.ys = ys
    return Solution(years, species_data), Model(y_indexes)

def plot_ratios_and_loads(sol1, model1, sol2, model2, label1, label2, outdir):
    # Set global matplotlib parameters for better readability
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'lines.linewidth': 3,
        'axes.linewidth': 2,
        'xtick.major.width': 2,
        'ytick.major.width': 2,
        'xtick.minor.width': 1.5,
        'ytick.minor.width': 1.5,
        'xtick.major.size': 8,
        'ytick.major.size': 8,
        'xtick.minor.size': 5,
        'ytick.minor.size': 5,
    })
    years1, ys1, y1 = sol1.ts, sol1.ys, model1.y_indexes
    years2, ys2, y2 = sol2.ts, sol2.ys, model2.y_indexes

    def safe_get(ys, yidx, name):
        return ys[:, yidx[name]] if name in yidx else np.full(ys.shape[0], np.nan)

    # --- Monomer Ratio Only ---
    plt.figure(figsize=(12, 8))
    for sol, yidx, label in [(sol1, y1, label1), (sol2, y2, label2)]:
        years, ys = sol.ts, sol.ys
        ab40_monomer_brain_plasma = safe_get(ys, yidx, 'AB40Mu_Brain_Plasma')
        ab42_monomer_brain_plasma = safe_get(ys, yidx, 'AB42Mu_Brain_Plasma')
        plt.plot(years, ab42_monomer_brain_plasma/ab40_monomer_brain_plasma, label=f'Monomer Ratio ({label})', linewidth=3)
    plt.xlabel('Time (years)')
    plt.ylabel('AB42/AB40 Brain Plasma Ratio')
    plt.title('AB42/AB40 Brain Plasma Ratio Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(outdir) / 'compare_ab42_ab40_monomer_ratio.png', dpi=300)
    plt.close()

    # --- Ratios ---
    plt.figure(figsize=(12, 8))
    for sol, yidx, label in [(sol1, y1, label1), (sol2, y2, label2)]:
        years, ys = sol.ts, sol.ys
        
        ab40_monomer = safe_get(ys, yidx, 'AB40_Monomer')
        ab42_monomer = safe_get(ys, yidx, 'AB42_Monomer')
        ab40_plaque = safe_get(ys, yidx, 'AB40_Plaque_unbound')
        ab42_plaque = safe_get(ys, yidx, 'AB42_Plaque_unbound')
        # Oligomer and fibril sums
        ab40_oligomers = np.sum([safe_get(ys, yidx, f'AB40_Oligomer{str(i).zfill(2)}') for i in range(2, 17)], axis=0)
        ab42_oligomers = np.sum([safe_get(ys, yidx, f'AB42_Oligomer{str(i).zfill(2)}') for i in range(2, 17)], axis=0)
        ab40_fibrils = np.sum([safe_get(ys, yidx, f'AB40_Fibril{str(i).zfill(2)}') for i in range(17, 24)], axis=0)
        ab42_fibrils = np.sum([safe_get(ys, yidx, f'AB42_Fibril{str(i).zfill(2)}') for i in range(17, 24)], axis=0)
        # Plot ratios
        plt.plot(years, ab42_monomer/ab40_monomer, label=f'Monomer Ratio ({label})', linewidth=3)
        plt.plot(years, ab42_oligomers/ab40_oligomers, label=f'Oligomer Ratio ({label})', linewidth=3)
        plt.plot(years, ab42_fibrils/ab40_fibrils, label=f'Fibril Ratio ({label})', linewidth=3)
        plt.plot(years, ab42_plaque/ab40_plaque, label=f'Plaque Ratio ({label})', linewidth=3)
    plt.xlabel('Time (years)')
    plt.ylabel('AB42/AB40 Ratio')
    plt.title('AB42/AB40 Ratios Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(outdir) / 'compare_ab42_ab40_ratios.png', dpi=300)
    plt.close()

    # --- AB40 and AB42 Monomer Loads (Both Models, Side-by-Side) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True)
    ab40_monomer_1 = safe_get(ys1, y1, 'AB40_Monomer')
    ab40_monomer_2 = safe_get(ys2, y2, 'AB40_Monomer')
    ab42_monomer_1 = safe_get(ys1, y1, 'AB42_Monomer')
    ab42_monomer_2 = safe_get(ys2, y2, 'AB42_Monomer')
    ax1.plot(years1, ab40_monomer_1, label=f'AB40 Monomer ({label1})', color='C0', linewidth=3)
    ax1.plot(years2, ab40_monomer_2, label=f'AB40 Monomer ({label2})', color='C1', linewidth=3)
    ax1.set_xlabel('Time (years)', fontsize=20)
    ax1.set_ylabel('Concentration (nM)', fontsize=20)
    ax1.set_title('AB40 Monomer Load', fontsize=22)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(years1, ab42_monomer_1, label=f'AB42 Monomer ({label1})', color='C0', linewidth=3)
    ax2.plot(years2, ab42_monomer_2, label=f'AB42 Monomer ({label2})', color='C1', linewidth=3)
    ax2.set_xlabel('Time (years)', fontsize=20)
    ax2.set_ylabel('Concentration (nM)', fontsize=20)
    ax2.set_title('AB42 Monomer Load', fontsize=22)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(outdir) / 'compare_monomer_loads.png', dpi=300)
    plt.close()

    # --- Loads ---
    for species, title in [('oligomers', 'Oligomer'), ('fibrils', 'Fibril'), ('plaque', 'Plaque')]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True)
        for sol, yidx, label, color in [
            (sol1, y1, label1, 'C0'),
            (sol2, y2, label2, 'C1')
        ]:
            years, ys = sol.ts, sol.ys
            if species == 'oligomers':
                ab40 = np.sum([safe_get(ys, yidx, f'AB40_Oligomer{str(i).zfill(2)}') for i in range(2, 17)], axis=0)
                ab42 = np.sum([safe_get(ys, yidx, f'AB42_Oligomer{str(i).zfill(2)}') for i in range(2, 17)], axis=0)
            elif species == 'fibrils':
                ab40 = np.sum([safe_get(ys, yidx, f'AB40_Fibril{str(i).zfill(2)}') for i in range(17, 24)], axis=0)
                ab42 = np.sum([safe_get(ys, yidx, f'AB42_Fibril{str(i).zfill(2)}') for i in range(17, 24)], axis=0)
            elif species == 'plaque':
                ab40 = safe_get(ys, yidx, 'AB40_Plaque_unbound')
                ab42 = safe_get(ys, yidx, 'AB42_Plaque_unbound')
            ax1.plot(years, ab40, label=label, linewidth=3)
            ax2.plot(years, ab42, label=label, linewidth=3)
        ax1.set_xlabel('Time (years)', fontsize=20)
        ax1.set_ylabel('Concentration (nM)', fontsize=20)
        ax1.set_title(f'AB40 {title} Load', fontsize=22)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.set_xlabel('Time (years)', fontsize=20)
        ax2.set_ylabel('Concentration (nM)', fontsize=20)
        ax2.set_title(f'AB42 {title} Load', fontsize=22)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(outdir) / f'compare_{species}_loads.png', dpi=300)
        plt.close()

    # --- CL_AB40_IDE and CL_AB42_IDE (Both Models, Unscaled) ---
    # Reload unscaled data for IDE and Microglia plots
    sol1_unscaled, model1_unscaled = load_solution(csv1, scale=1.0)
    sol2_unscaled, model2_unscaled = load_solution(csv2, scale=1.0)
    years1u, ys1u, y1u = sol1_unscaled.ts, sol1_unscaled.ys, model1_unscaled.y_indexes
    years2u, ys2u, y2u = sol2_unscaled.ts, sol2_unscaled.ys, model2_unscaled.y_indexes

    plt.figure(figsize=(12, 8))
    cl_ab40_ide_1 = safe_get(ys1u, y1u, 'CL_AB40_IDE')
    cl_ab42_ide_1 = safe_get(ys1u, y1u, 'CL_AB42_IDE')
    cl_ab40_ide_2 = safe_get(ys2u, y2u, 'CL_AB40_IDE')
    cl_ab42_ide_2 = safe_get(ys2u, y2u, 'CL_AB42_IDE')
    plt.plot(years1u, cl_ab40_ide_1, label=f'CL_AB40_IDE ({label1})', color='C0', linewidth=3, linestyle='-')
    plt.plot(years1u, cl_ab42_ide_1, label=f'CL_AB42_IDE ({label1})', color='C1', linewidth=3, linestyle='-')
    plt.plot(years2u, cl_ab40_ide_2, label=f'CL_AB40_IDE ({label2})', color='C0', linewidth=3, linestyle='--')
    plt.plot(years2u, cl_ab42_ide_2, label=f'CL_AB42_IDE ({label2})', color='C1', linewidth=3, linestyle='--')
    plt.xlabel('Time (years)')
    plt.ylabel('Clearance (nM/year or units)')
    plt.title('IDE Clearance of AB40 and AB42 (Both Models, Unscaled)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(outdir) / 'compare_cl_ide_clearance.png', dpi=300)
    plt.close()

    # --- Microglia Cell Count (Both Models, Unscaled) ---
    plt.figure(figsize=(12, 8))
    microglia_cell_count_1 = safe_get(ys1u, y1u, 'Microglia_cell_count')
    microglia_cell_count_2 = safe_get(ys2u, y2u, 'Microglia_cell_count')
    plt.plot(years1u, microglia_cell_count_1, label=f'Microglia Cell Count ({label1})', color='C0', linewidth=3)
    plt.plot(years2u, microglia_cell_count_2, label=f'Microglia Cell Count ({label2})', color='C1', linewidth=3)
    plt.xlabel('Time (years)')
    plt.ylabel('Cell Count')
    plt.title('Microglia Cell Count Over Time (Unscaled)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(outdir) / 'compare_microglia_cell_count.png', dpi=300)
    plt.close()

    # --- Microglia High Activity Fraction (Both Models, Unscaled) ---
    plt.figure(figsize=(12, 8))
    microglia_hi_fract_1 = safe_get(ys1u, y1u, 'Microglia_Hi_Fract')
    microglia_hi_fract_2 = safe_get(ys2u, y2u, 'Microglia_Hi_Fract')
    plt.plot(years1u, microglia_hi_fract_1, label=f'Microglia High Activity Fraction ({label1})', color='C0', linewidth=3)
    plt.plot(years2u, microglia_hi_fract_2, label=f'Microglia High Activity Fraction ({label2})', color='C1', linewidth=3)
    plt.xlabel('Time (years)')
    plt.ylabel('High Activity Fraction')
    plt.title('Microglia High Activity Fraction Over Time (Unscaled)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(outdir) / 'compare_microglia_hi_fract.png', dpi=300)
    plt.close()

    # --- Save final values for total oligomer, fibril, monomer, and plaque concentrations (both species, both models) ---
    def get_final_values(ys, yidx):
        ab40_monomer = safe_get(ys, yidx, 'AB40_Monomer')
        ab42_monomer = safe_get(ys, yidx, 'AB42_Monomer')
        ab40_oligomers = np.sum([safe_get(ys, yidx, f'AB40_Oligomer{str(i).zfill(2)}') for i in range(2, 17)], axis=0)
        ab42_oligomers = np.sum([safe_get(ys, yidx, f'AB42_Oligomer{str(i).zfill(2)}') for i in range(2, 17)], axis=0)
        ab40_fibrils = np.sum([safe_get(ys, yidx, f'AB40_Fibril{str(i).zfill(2)}') for i in range(17, 24)], axis=0)
        ab42_fibrils = np.sum([safe_get(ys, yidx, f'AB42_Fibril{str(i).zfill(2)}') for i in range(17, 24)], axis=0)
        ab40_plaque = safe_get(ys, yidx, 'AB40_Plaque_unbound')
        ab42_plaque = safe_get(ys, yidx, 'AB42_Plaque_unbound')
        return {
            'AB40_Monomer': ab40_monomer[-1],
            'AB42_Monomer': ab42_monomer[-1],
            'AB40_Oligomer': ab40_oligomers[-1],
            'AB42_Oligomer': ab42_oligomers[-1],
            'AB40_Fibril': ab40_fibrils[-1],
            'AB42_Fibril': ab42_fibrils[-1],
            'AB40_Plaque': ab40_plaque[-1],
            'AB42_Plaque': ab42_plaque[-1],
        }
    final1 = get_final_values(ys1, y1)
    final2 = get_final_values(ys2, y2)
    final_df = pd.DataFrame({
        label1: final1,
        label2: final2
    })
    final_df = final_df.T  # models as rows, species as columns
    final_df.to_csv(Path(outdir) / 'final_species_concentrations.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare SBML and ODE model results")
    parser.add_argument("--years", type=float, default=100.0, help="Number of years simulated")
    parser.add_argument("--drug", type=str, choices=["lecanemab", "gantenerumab"], default="gantenerumab", help="Drug type simulated")
    args = parser.parse_args()

    # Paths to your two CSVs
    csv1 = f'generated/{args.years}_year_simulation_results_{args.drug}.csv'
    csv2 = f'ODE_version/results/no_dose/{args.drug.capitalize()}_no_dose_{args.years}yr_results.csv'
    outdir = 'generated/figures/comparison'
    Path(outdir).mkdir(parents=True, exist_ok=True)
    sol1, model1 = load_solution(csv1, scale=0.2505) # Division by volume if isf is applied for this comparison
    sol2, model2 = load_solution(csv2, scale=1.0)
    plot_ratios_and_loads(sol1, model1, sol2, model2, 'Combined Master', 'ODE Version', outdir)
    print(f"Comparison plots saved to {outdir}")
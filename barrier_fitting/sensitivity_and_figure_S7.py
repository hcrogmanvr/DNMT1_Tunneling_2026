# sensitivity_and_figure_S7.py
# Computes ΔS/ħ sensitivity to barrier shape and plots Figure S7

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Constants
hbar = 0.657       # kcal·fs/mol
kB = 1.987e-3      # kcal/mol/K
T = 298.0
beta = 1 / (kB * T)
P = 64             # number of beads
tau = beta / P
m_eff = 15.0 * 418.4  # amu → kcal·fs^2/Å^2

# Load scan_coords.dat: 2-column file (r in Å, E in kcal/mol)
data = np.loadtxt("scan_coords.dat")
r = data[:, 0]
E = data[:, 1]

# Fit smooth spline to PES
E_spline = CubicSpline(r, E)
r_dense = np.linspace(r.min(), r.max(), 300)
E_fit = E_spline(r_dense)

# Function to compute instanton action from energy profile
def compute_action(E_vals, r_vals):
    dr = np.gradient(r_vals)
    kin = 0.5 * m_eff * np.sum((dr / tau) ** 2)
    pot = np.sum(E_vals) * tau
    return tau * (kin + pot) / hbar

# Baseline action
S0 = compute_action(E_fit, r_dense)

# Define perturbations
dH = 1.15  # ±0.05 eV ≈ 1.15 kcal/mol height shift
results = []

variations = [
    ("Baseline", E_fit),
    ("+Height", E_fit + dH),
    ("–Height", E_fit - dH),
    ("+Width", CubicSpline(r_dense * 0.975, E_fit)(r_dense)),  # compress
    ("–Width", CubicSpline(r_dense * 1.025, E_fit)(r_dense))   # stretch
]

# Compute ΔS/ħ for each variation
for label, E_mod in variations:
    S = compute_action(E_mod, r_dense)
    delta = 100 * abs(S - S0) / S0
    results.append((label, S, delta))

# Display results
print("ΔS/ħ Sensitivity Test:")
for label, S, delta in results:
    print(f"{label}: ΔS/ħ = {S:.4f} ({delta:.2f}% change)")

# Plot Figure S7
plt.figure(figsize=(6, 4))
plt.plot(r, E, 'ko', label='QM Scan Points')
plt.plot(r_dense, E_fit, 'r-', label='Spline Fit')
plt.xlabel("C–C Distance (Å)")
plt.ylabel("Energy (kcal/mol)")
plt.title("Figure S7: 1D PES from QM/MM Scan")
plt.legend()
plt.tight_layout()
plt.savefig("Figure_S7_1D_PES.png")
plt.close()
print("✅ Figure S7 saved as 'Figure_S7_1D_PES.png'")

# psi4_relaxed_scan.py
# ------------------------------------------------------------
# HOW TO USE (step-by-step):
# 1) Put this file in the SAME folder as your coordinates file (e.g., qm_region.xyz).
# 2) EDIT the block under "USER SETTINGS — EDIT HERE" to point at your file and set indices.
# 3) In a terminal:  conda activate psi4
# 4) Run:            python -m psi4 psi4_relaxed_scan.py
# 5) Check the printed "Numbered geometry" to verify indices.
# 6) The script runs a RELAXED distance scan and writes pmf.csv (xi_A,G_kcal_mol).
# ------------------------------------------------------------

import os
import numpy as np
import psi4

# ===================== USER SETTINGS — EDIT HERE =====================
XYZ_PATH       = "qm_region.xyz"   # <-- Change if your file has a different name/path
TOTAL_CHARGE   = 0                  # <-- Try 0; if SCF fails or your cluster is +1, set to 1
MULTIPLICITY   = 1                  # <-- Usually 1 (singlet)

# Choose the two atoms (1-based indices) for the scanned distance:
DONOR_INDEX    = 28                 # <-- Set to SAM–CH3 carbon index
ACCEPTOR_INDEX = 18                 # <-- Set to cytosine ring C5 (acceptor) index

# (Optional) Freeze some atoms to stabilize the cluster (use 1-based indices):
FREEZE_ATOMS   = []                 # e.g., [5, 12]  -> will add lines "freeze atom 5" and "freeze atom 12"

# Scan grid (Å). Start coarse; densify around ~1.6–1.8 Å if needed.
XI_GRID = np.array([
    2.50, 2.10, 1.80, 1.70, 1.65, 1.60, 1.55, 1.50, 1.45, 1.40,
    1.35, 1.30, 1.25, 1.20, 1.15, 1.10, 1.05, 1.00, 0.95, 0.90
])

# Quantum chemistry settings (change if you like)
METHOD  = "B3LYP-D3BJ"            # e.g., B3LYP-D3BJ, wB97X-D, M06-2X
BASIS   = "def2-SVP"              # e.g., def2-SVP (scan), def2-TZVP (refine)
MEMORY  = "4 GB"
THREADS = 8

# Retry behavior if a point fails to optimize
ENABLE_SOSCF_ON_RETRY = True       # turn on SOSCF on retry
MAXITER_RETRY         = 200        # more SCF cycles on retry
# =====================================================================

Eh2kcal = 627.509474


def load_xyz_as_geom(xyz_path: str, charge: int, mult: int) -> str:
    """Read an XYZ (with or without header) and return a Psi4 molecule block."""
    if not os.path.exists(xyz_path):
        raise FileNotFoundError(f"Couldn't find {xyz_path}. Place it next to this script or update XYZ_PATH.")
    lines = open(xyz_path, "r").read().strip().splitlines()
    # Strip standard XYZ header (first 2 lines) if present
    try:
        int(lines[0].strip())
        coord_lines = lines[2:]
    except Exception:
        coord_lines = lines
    coords = []
    for ln in coord_lines:
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) < 4:
            continue
        el, x, y, z = parts[:4]
        coords.append(f"{el:2s} {float(x): .8f} {float(y): .8f} {float(z): .8f}")
    block = "\n  ".join(coords)
    return f"""
molecule qmmm {{
  units angstrom
  symmetry c1
  {charge} {mult}
  {block}
}}
""".strip()


def main():
    # --- Psi4 setup ---
    psi4.core.set_output_file("scan.log", False)
    psi4.set_memory(MEMORY)
    psi4.set_num_threads(THREADS)

    geom = load_xyz_as_geom(XYZ_PATH, TOTAL_CHARGE, MULTIPLICITY)
    mol = psi4.geometry(geom)

    # Print a numbered geometry so you can verify / change indices if needed
    print("\n=== Numbered geometry (1-based indices) ===")
    print(mol.save_string_xyz())

    # Base options for robust constrained optimizations
    psi4.set_options({
        "scf_type": "df",
        "reference": "rhf",          # use 'uhf' for open-shell
        "d_convergence": 1e-8,
        "e_convergence": 1e-8,
        "geom_maxiter": 200,
        "opt_coordinates": "cartesian",
        "g_convergence": "qchem",
        "maxiter": 125,
    })

    results = []

    # Build freeze lines if any
    freeze_lines = [f"freeze atom {idx}" for idx in FREEZE_ATOMS]
    freeze_block = ("\n        ".join(freeze_lines)) if freeze_lines else ""

    for xi in XI_GRID:
        # Geometry constraint block for this scan point
        constraints = f"""
    constraints
      set
        bond {DONOR_INDEX:d} {ACCEPTOR_INDEX:d} {xi:.3f}
        {freeze_block}
      end
    end
""".rstrip()
        psi4.set_options({"optking__constraints": constraints})

        try:
            E = psi4.optimize(f"{METHOD}/{BASIS}", molecule=mol)
        except Exception as e:
            print(f"[WARN] Opt failed at ξ={xi:.3f} Å ({e})")
            if ENABLE_SOSCF_ON_RETRY:
                print("       -> retrying with SOSCF and higher maxiter…")
                psi4.set_options({"soscf": True, "maxiter": MAXITER_RETRY})
                E = psi4.optimize(f"{METHOD}/{BASIS}", molecule=mol)
            else:
                raise

        results.append((float(xi), float(E)))
        print(f"[OK] ξ={xi:.3f} Å  E={E:.10f} Eh")

    # Write pmf.csv with minimum shifted to 0 kcal/mol
    Emin = min(E for _, E in results)
    with open("pmf.csv", "w") as f:
        f.write("xi_A,G_kcal_mol\n")
        for xi, E in results:
            f.write(f"{xi:.3f},{(E - Emin)*Eh2kcal:.6f}\n")
    print("\nWrote pmf.csv with", len(results), "points.")


if __name__ == "__main__":
    main()

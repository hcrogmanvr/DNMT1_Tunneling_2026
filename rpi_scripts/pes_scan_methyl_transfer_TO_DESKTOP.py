
import psi4
import numpy as np
import os

psi4.set_memory("2 GB")

psi4.set_options({
    "basis": "def2-SVP",
    "scf_type": "pk",
    "guess": "sad"
})

distances = np.linspace(3.5, 1.5, 21)
energies = []

# Coordinates of donor and acceptor atoms
donor_coords = np.array([-1.06285, 2.03821, 0.98943])
acceptor_coords = np.array([1.12218, 2.08796, -1.06515])
vector = acceptor_coords - donor_coords
unit_vector = vector / np.linalg.norm(vector)

# Elements and atomic coordinates from XYZ file (short version)
elements = ['N', 'N', 'N', 'N', 'N', 'N', 'C', 'C', 'C', 'C']
coordinates = [
    [1.83271, 1.0033, -0.36145],
    [1.57426, -1.81513, 3.58072],
    [3.06032, -3.41494, 4.11077],
    [3.7185, -3.72172, 6.84497],
    [1.48772, -2.91099, 7.45111],
    [0.40551, -1.70319, 5.6777],
    [1.12218, 2.08796, -1.06515],  # acceptor
    [1.43351, 1.67195, -2.59335],
    [-0.39652, 2.41102, -0.6178],
    [-1.06285, 2.03821, 0.98943]   # donor
]

for i, R in enumerate(distances):
    # Adjust acceptor position
    new_coords = np.array(coordinates)
    new_acceptor = donor_coords + R * unit_vector
    new_coords[6] = new_acceptor  # acceptor is at index 6

    # Create geometry string
    geom = "\n".join(
        f"{el} {x:.6f} {y:.6f} {z:.6f}"
        for el, (x, y, z) in zip(elements, new_coords)
    )

    print(f"# --- Step {i+1}: Distance {R:.2f} Å ---")
    mol = psi4.geometry(f"""
0 1
{geom}
no_com
no_reorient
""")

    try:
        energy = psi4.energy("B3LYP", molecule=mol)
        energies.append((R, energy))
        print("Distance: {:.2f} Å, Energy: {:.6f} Hartree".format(R, energy))
    except Exception as e:
        print("Step {} at R = {:.2f} Å failed with error: {}".format(i+1, R, str(e)))
        energies.append((R, None))

# ✅ Save results to Desktop
desktop_path = os.path.expanduser("~/Desktop/pes_results.txt")
with open(desktop_path, "w") as f:
    for R, E in energies:
        if E is not None:
            f.write(f"{R:.2f} {E:.10f}\n")
        else:
            f.write(f"{R:.2f} NaN\n")

import os
import random
from datetime import datetime

def generate_instances(n, m, p, num_instances, base_seed):
    """
    Generates multiple P-Median Problem instances in COORDS format.
    Each run creates a new folder named 'generated_instances_YYYYMMDD_HHMMSS'
    to avoid overwriting previous data.
    """

    # Base directory (where the script is located)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create a unique folder name using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"generated_instances_{timestamp}"
    folder_path = os.path.join(script_dir, folder_name)

    # Create the folder
    os.makedirs(folder_path, exist_ok=True)

    for i in range(num_instances):
        seed = base_seed + i
        random.seed(seed)

        filename = f"coords_n{n}_p{p}_seed{seed}.txt"
        file_path = os.path.join(folder_path, filename)

        with open(file_path, "w") as f:
            f.write(f"COORDS {n} {m} {p} {seed}\n")
            f.write("# Clients: id x y\n")

            for client_id in range(1, n + 1):
                x = random.uniform(0, 100)
                y = random.uniform(0, 100)
                f.write(f"C {client_id} {x:.4f} {y:.4f}\n")

            f.write("\n# Candidate Sites: id x y\n")

            for site_id in range(1, m + 1):
                x = random.uniform(0, 100)
                y = random.uniform(0, 100)
                f.write(f"S {site_id} {x:.4f} {y:.4f}\n")

    print(f"\n✅ Successfully generated {num_instances} instances in folder:")
    print(f"   → {folder_path}\n")
    return folder_path


# --------------------------------------------
# Interactive console input
# --------------------------------------------
if __name__ == "__main__":
    print("=== P-Median Instance Generator ===")
    n = int(input("Enter number of clients (n): "))
    m = int(input("Enter number of candidate sites (m): "))
    p = int(input("Enter number of facilities to open (p): "))
    num_instances = int(input("Enter number of instances to generate: "))
    base_seed = int(input("Enter base seed (e.g., 42): "))

    generate_instances(n, m, p, num_instances, base_seed)

import os
import random

def generate_instances(n, m, p, num_instances, base_seed):
    """
    Generates multiple P-Median Problem instances in COORDS format.
    Each instance will have a unique seed (base_seed + i) for reproducibility.
    All files are saved inside the folder 'generated_instances', located
    in the same directory as this Python script.
    """

    # Get the folder where the script itself is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_name = os.path.join(script_dir, "generated_instances")
    os.makedirs(folder_name, exist_ok=True)

    for i in range(num_instances):
        seed = base_seed + i
        random.seed(seed)

        filename = f"coords_n{n}_p{p}_seed{seed}.txt"
        file_path = os.path.join(folder_name, filename)

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

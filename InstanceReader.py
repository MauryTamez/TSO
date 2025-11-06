import os

def read_instance(file_path):
    """
    Reads a P-Median instance file and returns structured data.
    """
    clients = []
    sites = []
    n = m = p = seed = None

    with open(file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        # Header line
        if line.startswith("COORDS"):
            _, n, m, p, seed = line.split()
            n, m, p, seed = int(n), int(m), int(p), int(seed)

        # Clients
        elif line.startswith("C"):
            _, cid, x, y = line.split()
            clients.append((int(cid), float(x), float(y)))

        # Candidate Sites
        elif line.startswith("S"):
            _, sid, x, y = line.split()
            sites.append((int(sid), float(x), float(y)))

    return {
        "n": n,
        "m": m,
        "p": p,
        "seed": seed,
        "clients": clients,
        "sites": sites
    }


# --------------------------------------------
# Interactive test
# --------------------------------------------
if __name__ == "__main__":
    print("=== P-Median Instance Reader ===")

    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_instances")

    # List all .txt files inside the folder
    files = [f for f in os.listdir(folder) if f.endswith(".txt")]

    if not files:
        print("No instances found in 'generated_instances' folder.")
    else:
        print("Available instance files:")
        for i, file in enumerate(files):
            print(f"{i + 1}. {file}")

        choice = int(input("\nEnter the number of the file you want to read: ")) - 1
        file_path = os.path.join(folder, files[choice])

        data = read_instance(file_path)

        print("\n=== Instance Info ===")
        print(f"Clients: {data['n']}")
        print(f"Candidate Sites: {data['m']}")
        print(f"Facilities to open (p): {data['p']}")
        print(f"Seed: {data['seed']}")
        print(f"First client: {data['clients'][0]}")
        print(f"First site: {data['sites'][0]}")
        print("\nInstance loaded successfully!")

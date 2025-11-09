import os
import math
import random
import time
import pandas as pd

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
# Helper: Euclidean distance between two points
# --------------------------------------------
def euclidean_distance(point1, point2):
    x1, y1 = point1[1], point1[2]
    x2, y2 = point2[1], point2[2]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


# --------------------------------------------
# Evaluate candidate sites (first step of the heuristic)
# --------------------------------------------
def evaluate_candidates(clients, sites, open_sites):
    """
    Evaluates all non-open candidate sites.
    For each candidate, computes the total reduction in cost
    (sum of assignment distances) if the site were to be added.
    Handles both initial (no sites open) and iterative cases.
    """
    benefits = []

    # If no sites are open yet → compute initial total cost per candidate
    if not open_sites:
        for site in sites:
            total_cost = 0.0
            for client in clients:
                total_cost += euclidean_distance(client, site)
            benefits.append((site[0], -total_cost))  # negative because we minimize cost
        return benefits

    # Otherwise → compute improvement based on existing open sites
    current_best_distance = []
    for c in clients:
        dist_min = min(euclidean_distance(c, s) for s in sites if s[0] in open_sites)
        current_best_distance.append(dist_min)

    for site in sites:
        site_id = site[0]
        if site_id in open_sites:
            continue

        total_improvement = 0.0
        for i, client in enumerate(clients):
            d_current = current_best_distance[i]
            d_new = euclidean_distance(client, site)
            if d_new < d_current:
                total_improvement += (d_current - d_new)

        benefits.append((site_id, total_improvement))

    return benefits


# --------------------------------------------
# Construct Restricted Candidate List (RCL)
# --------------------------------------------

ALPHA = 0.3  

def build_rcl(candidates):
    """
    candidates: list of tuples (site_id, benefit)
      - benefit might be negative (we used -total_cost for initial step)
      - or positive (improvement) for later steps.
    This function converts benefits to a 'score' where larger means better,
    then builds RCL using alpha (ALPHA) and returns the selected (site_id, benefit).
    """

    if not candidates:
        raise ValueError("No candidates provided to build_rcl()")

    # Compute a score (higher = better) robustly:
    # If benefits are all <= 0 (likely initial step where benefits are -cost),
    # convert by score = -benefit (so lower cost -> higher score).
    # If benefits contain positives (improvements), use them as scores.
    benefits = [b for (_, b) in candidates]
    max_b = max(benefits)
    min_b = min(benefits)

    scores = []
    for site_id, b in candidates:
        if max_b <= 0:
            # all non-positive -> these are negative costs: convert
            score = -b
        else:
            # there is at least one positive improvement -> treat positive as better
            # but if some are negative (rare), convert those to small scores:
            score = b if b >= 0 else 0.0
        scores.append((site_id, b, score))

    # Determine best and worst score
    best_score = max(s[2] for s in scores)
    worst_score = min(s[2] for s in scores)

    # If all scores equal (degenerate), RCL = all
    if best_score == worst_score:
        rcl = [(sid, b) for sid, b, s in scores]
    else:
        # threshold: keep top ones: threshold_score = best - alpha*(best - worst)
        threshold_score = best_score - ALPHA * (best_score - worst_score)
        rcl = [(sid, b) for sid, b, s in scores if s >= threshold_score]

    # Debug prints (remove or comment if you don't want clutter)
    print(f"\n[RCL] alpha={ALPHA:.2f} | best_score={best_score:.4f} worst_score={worst_score:.4f} | threshold={threshold_score:.4f}")
    print(f"[RCL] contains {len(rcl)} / {len(candidates)} candidates")

    # Select one at random from RCL
    selected = random.choice(rcl)
    # selected is (site_id, benefit)
    print(f"[RCL] selected site {selected[0]} with benefit {selected[1]:.4f}")
    return selected


# --------------------------------------------
# Construct final solution
# --------------------------------------------
    
def calculate_total_cost(clients, sites, open_sites):
    total_cost = 0.0
    assignments = []  # (cliente_id, sitio_asignado, distancia)

    # Recorremos todos los clientes
    for cid, cx, cy in clients:
        best_site = None
        best_dist = float("inf")

        # Revisamos todos los sitios abiertos
        for sid in open_sites:
            _, sx, sy = sites[sid - 1]  # recordar que los IDs son 1-based
            dist = math.dist((cx, cy), (sx, sy))

            # Si este sitio está más cerca, lo actualizamos
            if dist < best_dist:
                best_dist = dist
                best_site = sid

        # Guardamos la asignación
        assignments.append((cid, best_site, best_dist))
        total_cost += best_dist

    return total_cost, assignments

# --------------------------------------------
# Main automático: procesa todas las instancias de una carpeta
# --------------------------------------------
import time
import pandas as pd
import os

def main():
    print("=== Automatic Greedy Randomized Heuristic for P-Median ===")

    # Buscar carpetas con instancias generadas
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folders = [f for f in os.listdir(base_dir)
               if os.path.isdir(os.path.join(base_dir, f)) and f.startswith("generated_instances")]

    if not folders:
        print("No se encontraron carpetas con instancias generadas.")
        return

    # Mostrar carpetas y seleccionar una
    print("\nAvailable instance folders:")
    for i, folder in enumerate(folders):
        print(f"{i + 1}. {folder}")

    folder_choice = int(input("\nSelect the folder number: ")) - 1
    folder_path = os.path.join(base_dir, folders[folder_choice])

    # Obtener todos los archivos .txt dentro de la carpeta seleccionada
    files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    if not files:
        print("No se encontraron archivos de instancia en la carpeta seleccionada.")
        return

    print(f"\nFound {len(files)} instances. Starting automatic analysis...\n")

    results = []

    for idx, file_name in enumerate(sorted(files), start=1):
        file_path = os.path.join(folder_path, file_name)
        print(f"\n[{idx}/{len(files)}] Processing instance: {file_name}")

        # Medir tiempo de ejecución
        start_time = time.time()

        # Leer instancia
        data = read_instance(file_path)

        # Paso constructivo
        open_sites = []
        iteration = 1
        while len(open_sites) < data["p"]:
            benefits = evaluate_candidates(data["clients"], data["sites"], open_sites)
            selected_site_id, _ = build_rcl(benefits)
            open_sites.append(selected_site_id)
            iteration += 1

        # Calcular costo total
        total_cost, assignments = calculate_total_cost(data["clients"], data["sites"], open_sites)

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"  -> Completed in {elapsed_time:.3f} s | Cost: {total_cost:.4f}")

        # Guardar resultado en la lista
        results.append({
            "Instance": file_name,
            "Heuristic_Cost": total_cost,
            "Heuristic_Time(s)": elapsed_time,
            "LocalSearch_Cost": None,
            "LocalSearch_Time(s)": None,
            "Absolute_Diff": None,
            "Improvement(%)": None
        })

    # Convertir resultados a DataFrame y exportar a Excel
    df = pd.DataFrame(results)
    output_path = os.path.join(base_dir, "PMP_Results.xlsx")
    df.to_excel(output_path, index=False)
    print(f"\n✅ All results saved successfully to '{output_path}'")

# --------------------------------------------
# Entry point
# --------------------------------------------
if __name__ == "__main__":
    main()

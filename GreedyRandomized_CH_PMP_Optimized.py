import os
import math
import random
import time
import pandas as pd
from datetime import datetime
import numpy as np

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


def construct_greedy_randomized_fast(clients, sites, p, alpha=0.3, batch_size=5000):
    """
    Constructive phase of Greedy Randomized Heuristic (vectorized & batched).
    Much faster for large n, m (e.g., 10k+). Keeps current best distances incrementally.
    """
    client_coords = np.array([(x, y) for _, x, y in clients], dtype=np.float32)
    site_coords   = np.array([(x, y) for _, x, y in sites], dtype=np.float32)
    m_sites       = site_coords.shape[0]
    n_clients     = client_coords.shape[0]

    open_sites = []
    current_best = np.full(n_clients, np.inf, dtype=np.float32)

    for it in range(p):
        # --- 1) Sample candidate sites (para no probar los m sitios)
        candidate_pool = [s for s in range(m_sites) if (s + 1) not in open_sites]
        sample_size = min(len(candidate_pool), 500)
        sampled = random.sample(candidate_pool, sample_size)

        # --- 2) Compute total cost improvement for each sampled site
        improv = []
        for j in sampled:
            # distance from all clients to this site
            d_new = np.linalg.norm(client_coords - site_coords[j], axis=1)
            delta = current_best - np.minimum(current_best, d_new)
            gain = float(np.sum(delta))
            improv.append((j + 1, gain))

        # --- 3) RCL (restricted candidate list)
        if not improv:
            break
        gains = np.array([g for (_, g) in improv], dtype=np.float32)
        g_max, g_min = np.max(gains), np.min(gains)
        threshold = g_max - alpha * (g_max - g_min)

        # Build RCL robustly
        rcl = [sid for (sid, g) in improv if g >= threshold]
        if not rcl:  # fallback safety
            # if RCL is empty, pick the best site (max gain)
            best_sid = improv[int(np.argmax(gains))][0]
            rcl = [best_sid]

        # --- 4) Pick one site randomly from RCL
        chosen = random.choice(rcl)
        open_sites.append(chosen)

        # --- 5) Update current_best distances incrementally
        j_idx = chosen - 1
        d_new = np.linalg.norm(client_coords - site_coords[j_idx], axis=1)
        current_best = np.minimum(current_best, d_new)

        if (it + 1) % 50 == 0:
            print(f"  → Iter {it + 1}/{p} | best mean dist: {np.mean(current_best):.4f}")

    # Final total cost
    total_cost = float(np.sum(current_best))
    return open_sites, total_cost


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
    # print(f"\n[RCL] alpha={ALPHA:.2f} | best_score={best_score:.4f} worst_score={worst_score:.4f} | threshold={threshold_score:.4f}")
    # print(f"[RCL] contains {len(rcl)} / {len(candidates)} candidates")

    # Select one at random from RCL
    selected = random.choice(rcl)
    # selected is (site_id, benefit)
    # print(f"[RCL] selected site {selected[0]} with benefit {selected[1]:.4f}")
    return selected


# --------------------------------------------
# Construct final solution
# --------------------------------------------
def calculate_total_cost(clients, sites, open_sites, batch_size=None):
    total_cost = 0.0
    assignments = []

    # Convertir coords a NumPy (más rápido)
    client_coords = np.array([(x, y) for _, x, y in clients], dtype=np.float32)
    site_coords   = np.array([(x, y) for _, x, y in sites], dtype=np.float32)
    open_idx      = np.array([sid - 1 for sid in open_sites], dtype=np.int32)

    if batch_size is None or batch_size >= len(client_coords):
        # Computo normal (sin batching)
        dists = np.linalg.norm(client_coords[:, None, :] - site_coords[open_idx][None, :, :], axis=2)
        min_dists = np.min(dists, axis=1)
    else:
        # Computo por lotes
        min_dists = []
        for start in range(0, len(client_coords), batch_size):
            end = start + batch_size
            chunk = client_coords[start:end]
            d = np.linalg.norm(chunk[:, None, :] - site_coords[open_idx][None, :, :], axis=2)
            min_dists.append(np.min(d, axis=1))
        min_dists = np.concatenate(min_dists)

    total_cost = float(np.sum(min_dists))
    assignments = [
        (cid, open_sites[int(np.argmin(np.linalg.norm(
            np.array([cx, cy]) - site_coords[open_idx], axis=1
        )))], dist)
        for (cid, cx, cy), dist in zip(clients, min_dists)
    ]

    return total_cost, assignments


# --------------------------------------------
# Local Search: 1-Swap Best Improvement (NumPy + capped swaps + client sampling)
# --------------------------------------------

def local_search_1swap(clients, sites, open_sites, current_cost,
                       sample_fraction=0.0005,           # 3% de clientes por iteración
                       open_sample_size=10,            # # sitios abiertos a muestrear por iteración
                       closed_sample_size=50,          # # sitios cerrados a muestrear por iteración
                       MAX_SWAPS_CHECKED=500,          # límite duro de swaps evaluados por iteración
                       batch_size=5000,                # batch para confirmación full-cost
                       verbose=False):
    """
    1-swap best-improvement con:
      - muestreo de clientes (3%) para estimar cada swap,
      - tope de swaps por iteración,
      - muestreo de abiertos/cerrados (10 x 50),
      - confirmación del mejor swap con coste completo (batched).
    No construye la matriz completa n x m en RAM (solo distancias para la muestra).
    """

    # Convertimos coords a NumPy (float32 para rendimiento/memoria)
    client_coords = np.asarray([(x, y) for _, x, y in clients], dtype=np.float32)
    site_coords   = np.asarray([(x, y) for _, x, y in sites],   dtype=np.float32)
    n_clients     = client_coords.shape[0]
    m_sites       = site_coords.shape[0]

    best_sites = open_sites[:]
    best_cost  = float(current_cost)
    improved   = True
    iteration  = 0

    # Helper: costo sobre conjunto de sitios (para confirmación full)
    def full_cost(open_list):
        # Usa tu versión batcheada/rápida existente
        cst, _ = calculate_total_cost(clients, sites, open_list, batch_size=batch_size)
        return float(cst)

    while improved:
        iteration += 1
        improved        = False
        best_swap       = None
        swaps_checked   = 0

        # ---- 1) Elegimos muestra de clientes de esta iteración
        sample_size = max(1, int(n_clients * sample_fraction))
        if sample_size > n_clients:
            sample_size = n_clients
        sample_idx = np.random.choice(n_clients, sample_size, replace=False)

        C = client_coords[sample_idx]             # (s,2)

        # ---- 2) Precomputamos distancias de la muestra a TODOS los sitios (s, m)
        # Esto es manejable: s<<n (3%); m puede ser 10k -> s*m cabe en memoria con float32
        # dist(C, S) = ||C - S||2
        diff = C[:, None, :] - site_coords[None, :, :]
        D_sample = np.sqrt(np.sum(diff * diff, axis=2, dtype=np.float32), dtype=np.float32)  # (s, m)

        # Distancias actuales de la muestra al conjunto abierto actual
        open_idx = np.array([sid - 1 for sid in best_sites], dtype=np.int32)
        current_sample_dists = np.min(D_sample[:, open_idx], axis=1)  # (s,)

        # ---- 3) Muestreamos sitios abiertos/cerrados y limitamos swaps
        sample_open = random.sample(best_sites, min(len(best_sites), open_sample_size))
        closed_pool = [s for s in range(1, m_sites + 1) if s not in best_sites]
        if closed_pool:
            sample_closed = random.sample(closed_pool, min(len(closed_pool), closed_sample_size))
        else:
            sample_closed = []

        # Mejor mejora estimada (sobre la muestra) en esta iteración
        best_sample_gain = 0.0

        # ---- 4) Exploramos SOLO los pares muestreados, con tope de swaps
        for i in sample_open:
            if swaps_checked >= MAX_SWAPS_CHECKED:
                break
            i_idx = i - 1

            # Distancias de la muestra al sitio cerrado i (para re-asignación si se cierra)
            d_i = D_sample[:, i_idx]  # (s,)

            for j in sample_closed:
                if swaps_checked >= MAX_SWAPS_CHECKED:
                    break
                swaps_checked += 1

                j_idx = j - 1

                # Distancias de la muestra al candidato j
                d_j = D_sample[:, j_idx]  # (s,)

                # Estimación rápida del nuevo coste en la muestra:
                # new_sample_dists = min( (min sobre open\{i}), d_j )
                if len(open_idx) > 1:
                    # quitamos i del conjunto abierto para el cálculo
                    mask = open_idx != i_idx
                    open_wo_i = open_idx[mask]
                    d_open_wo_i = np.min(D_sample[:, open_wo_i], axis=1)
                    new_sample_dists = np.minimum(d_open_wo_i, d_j)
                else:
                    # si solo había un abierto (i), el nuevo min es d_j
                    new_sample_dists = d_j

                old_sample_cost = float(np.sum(current_sample_dists))
                new_sample_cost = float(np.sum(new_sample_dists))
                sample_gain     = old_sample_cost - new_sample_cost  # >0 mejora

                if sample_gain > best_sample_gain + 1e-9:
                    # Candidato prometedor: confirmamos con coste COMPLETO (batched)
                    candidate_sites = best_sites[:]
                    candidate_sites.remove(i)
                    candidate_sites.append(j)

                    new_full = full_cost(candidate_sites)
                    if new_full + 1e-9 < best_cost:
                        best_cost   = new_full
                        best_swap   = (i, j)
                        best_sample_gain = sample_gain
                        improved    = True

        # ---- 5) Aplicamos el mejor swap si mejoró el coste completo
        if improved and best_swap:
            i, j = best_swap
            best_sites.remove(i)
            best_sites.append(j)
            if verbose:
                print(f"  → Iter {iteration}: swap ({i}→{j}) | New cost = {best_cost:.4f}")
        else:
            if verbose:
                print(f"  → No improving swap found. Local search ends at iteration {iteration}.")

    return best_sites, best_cost

# --------------------------------------------
# Automatic Main: Process all instances of a folder
# --------------------------------------------

def main():
    print("=== Automatic Greedy Randomized Heuristic for P-Median ===")

    # Search folders with generated instances
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folders = [f for f in os.listdir(base_dir)
               if os.path.isdir(os.path.join(base_dir, f)) and f.startswith("generated_instances")]

    if not folders:
        print("No se encontraron carpetas con instancias generadas.")
        return

    # Show folders and select one
    print("\nAvailable instance folders:")
    for i, folder in enumerate(folders):
        print(f"{i + 1}. {folder}")

    folder_choice = int(input("\nSelect the folder number: ")) - 1
    folder_path = os.path.join(base_dir, folders[folder_choice])

    # Get all .txt files inside the selected folder
    files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    if not files:
        print("No se encontraron archivos de instancia en la carpeta seleccionada.")
        return

    print(f"\nFound {len(files)} instances. Starting automatic analysis...\n")

    results = []

    for idx, file_name in enumerate(sorted(files), start=1):
        file_path = os.path.join(folder_path, file_name)
        print(f"\n[{idx}/{len(files)}] Processing instance: {file_name}")

        # Measure execution time
        start_time = time.time()

        # Read instance
        data = read_instance(file_path)

        # Constructive step
        open_sites = []
        iteration = 1
        
        print(f"  > p = {data['p']} | n = {data['n']} | m = {data['m']}")
        print(f"  > Expected open sites: {data['p']}")
        print(f"  > Starting construction phase...")

        # Constructive step (optimized)
        print("\nConstructing initial solution (optimized greedy randomized)...")
        open_sites, total_cost = construct_greedy_randomized_fast(
        data["clients"], data["sites"], data["p"], alpha=0.3, batch_size=5000
    )
        print(f"  → Construction complete | Cost: {total_cost:.4f}")

        # Calculate total cost
        total_cost, assignments = calculate_total_cost(data["clients"], data["sites"], open_sites)

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"  -> Completed in {elapsed_time:.3f} s | Cost: {total_cost:.4f}")
        
        # After constructive phase
        total_cost, assignments = calculate_total_cost(data["clients"], data["sites"], open_sites)

        # --- Apply Local Search ---
        print("\nApplying Local Search (1-swap best improvement)...")
        start_time_ls = time.time()
        improved_sites, improved_cost = local_search_1swap(
            data["clients"], data["sites"], open_sites, total_cost
        )
        end_time_ls = time.time()

        ls_time = end_time_ls - start_time_ls
        abs_diff = total_cost - improved_cost
        improvement_pct = (abs_diff / total_cost * 100) if total_cost != 0 else 0

        # print(f"Local Search completed. Improved cost: {improved_cost:.4f}")
        # print(f"Improvement: {abs_diff:.4f} ({improvement_pct:.2f}%)")
        # print(f"Execution time: {ls_time:.4f}s\n")

        # Save result to list
        results.append({
            "Instance": file_name,
            "Heuristic_Cost": total_cost,
            "Heuristic_Time(s)": elapsed_time,
            "LocalSearch_Cost": improved_cost,
            "LocalSearch_Time(s)": ls_time,
            "Absolute_Diff": abs_diff,
            "Improvement(%)": improvement_pct
        })

    # Convert results to DataFrame and export to Excel
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(base_dir, f"PMP_Results_{timestamp}.xlsx")
    df.to_excel(output_path, index=False)
    print(f"\n✅ All results saved successfully to '{output_path}'")

# --------------------------------------------
# Entry point
# --------------------------------------------
if __name__ == "__main__":
    main()

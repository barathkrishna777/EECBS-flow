import os
import subprocess
import re
import numpy as np
from scipy.signal import savgol_filter
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import deque

# Configuration
EECBS_BIN = "./build/eecbs"
DATA_DIR = "data"
MAP_DIR = os.path.join(DATA_DIR, "mapf-map")
SCEN_DIR = os.path.join(DATA_DIR, "scen-random")
OUTPUT_NPZ_DIR = os.path.join(DATA_DIR, "flow_training_data_multi")
BD_DIR = os.path.join(DATA_DIR, "bd_npzs", "large_scale") 

# Rishi's Omitted and Held-out (Test) Maps
OMITTED_MAPS = {"brc202d", "orz900", "maze-128-128-1", "maze-128-128-10"}
HELD_OUT_TEST = {
    "Paris_1_256", "empty_48_48", "maze_128_128_2", "random_64_64_10", 
    "random_32_32_10", "warehouse_10_20_10_2_1", "den312d", "den520d"
}

WINDOW_LENGTH = 3 
POLY_ORDER = 2    
MAX_WORKERS = os.cpu_count()

os.makedirs(OUTPUT_NPZ_DIR, exist_ok=True)
os.makedirs(BD_DIR, exist_ok=True)

def parse_paths_txt(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    all_paths = []
    for line in lines:
        if not line.startswith("Agent"): continue
        coords = re.findall(r"\((\d+),(\d+)\)", line)
        path = np.array([[int(r), int(c)] for r, c in coords], dtype=np.float32)
        all_paths.append(path)
    if not all_paths: return np.array([])
    max_len = max(len(p) for p in all_paths)
    paths_array = np.zeros((len(all_paths), max_len, 2), dtype=np.float32)
    for i, p in enumerate(all_paths):
        paths_array[i, :len(p), :] = p
        paths_array[i, len(p):, :] = p[-1] 
    return paths_array

def smooth_and_extract_velocities(paths_array):
    if len(paths_array) == 0: return paths_array, paths_array
    N, T, D = paths_array.shape
    window = min(WINDOW_LENGTH, T)
    if window % 2 == 0: window -= 1
    if window < 3: return paths_array, np.gradient(paths_array, axis=1)
    smoothed_positions = savgol_filter(paths_array, window_length=window, polyorder=POLY_ORDER, axis=1, deriv=0)
    velocities = savgol_filter(paths_array, window_length=window, polyorder=POLY_ORDER, axis=1, deriv=1)
    return smoothed_positions, velocities

def read_map(map_file):
    with open(map_file, 'r') as f:
        f.readline(); h = int(f.readline().split()[1]); w = int(f.readline().split()[1]); f.readline()
        map_data = np.zeros((h, w), dtype=np.int8)
        for r in range(h):
            line = f.readline().strip()
            for c in range(w):
                if line[c] in ['@', 'T', 'O']: map_data[r, c] = 1
    return map_data

def parse_scenario_goals(scen_file, max_agents=1000):
    goals = []
    with open(scen_file, 'r') as f:
        f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 8:
                goals.append([int(parts[7]), int(parts[6])]) 
            if len(goals) == max_agents: break
    return np.array(goals)

def compute_bd_heuristic(map_data, goals):
    H, W = map_data.shape
    N = len(goals)
    bd_array = np.full((N, H, W), 10000, dtype=np.int16) 
    
    for i, (gr, gc) in enumerate(goals):
        if map_data[gr, gc] == 1: continue
        queue = deque([(gr, gc, 0)])
        bd_array[i, gr, gc] = 0
        
        while queue:
            r, c, dist = queue.popleft()
            ndist = dist + 1
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if map_data[nr, nc] == 0 and bd_array[i, nr, nc] == 10000:
                        bd_array[i, nr, nc] = ndist
                        queue.append((nr, nc, ndist))
    return bd_array

# ==========================================
# PHASE 1: Generate BD Heuristics Safely
# ==========================================
def generate_scenario_bd(map_path, scen_path):
    scen_name = os.path.basename(scen_path).replace(".scen", "")
    map_name = os.path.basename(map_path).replace(".map", "")
    bd_key = f"{map_name}-random-{scen_name.split('-random-')[-1]}"
    out_bd_file = os.path.join(BD_DIR, f"{scen_name}_bds.npz")
    
    if os.path.exists(out_bd_file):
        return f"BD exists: {scen_name}"
        
    print(f"--> Building Heuristic Grid: {scen_name} (Takes ~30s)")
    try:
        map_data = read_map(map_path)
        goals = parse_scenario_goals(scen_path, max_agents=1000)
        bd_array = compute_bd_heuristic(map_data, goals)
        
        # Atomic write to prevent file corruption
        # FIX: Ensure the tmp file ends in .npz so numpy doesn't silently append it
        tmp_file = out_bd_file + f".{os.getpid()}.tmp.npz" 
        np.savez_compressed(tmp_file, **{bd_key: bd_array})
        os.rename(tmp_file, out_bd_file)
        return f"BD Generated: {scen_name}"
    except Exception as e:
        return f"BD Error: {scen_name} ({str(e)})"

# ==========================================
# PHASE 2: Generate EECBS Trajectories
# ==========================================
def generate_scenario_trajectory(map_path, scen_path, num_agents):
    scen_name = os.path.basename(scen_path).replace(".scen", "")
    out_traj_file = os.path.join(OUTPUT_NPZ_DIR, f"{scen_name}_{num_agents}.npz")
    tmp_path_file = f"tmp_{scen_name}_{num_agents}_{os.getpid()}.txt" 
    
    if os.path.exists(out_traj_file): 
        return f"Traj Exists: {scen_name} (N={num_agents})"
        
    cmd = [
        EECBS_BIN, "-m", map_path, "-a", scen_path, "-k", str(num_agents),
        "--outputPaths", tmp_path_file, "--suboptimality", "2.0"
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
        if os.path.exists(tmp_path_file):
            paths_discrete = parse_paths_txt(tmp_path_file)
            if len(paths_discrete) > 0:
                positions, velocities = smooth_and_extract_velocities(paths_discrete)
                np.savez_compressed(out_traj_file, discrete_positions=paths_discrete, 
                                   smoothed_positions=positions, expert_velocities=velocities)
                os.remove(tmp_path_file)
                return f"Traj Done: {scen_name} (N={num_agents})"
    except Exception as e:
        if os.path.exists(tmp_path_file): os.remove(tmp_path_file)
        return f"Traj Fail: {scen_name}_{num_agents} ({str(e)})"
    
    if os.path.exists(tmp_path_file): os.remove(tmp_path_file)
    return f"Traj Fail: {scen_name}_{num_agents}"

# ==========================================
# EXECUTION PIPELINE
# ==========================================
def process_benchmark_parallel():
    map_files = glob.glob(os.path.join(MAP_DIR, "*.map"))
    agent_counts = [20, 50, 100, 200, 400, 600, 800, 1000]
    
    bd_jobs = []
    traj_jobs = []
    
    for map_path in map_files:
        map_name = os.path.basename(map_path).replace(".map", "")
        if map_name in OMITTED_MAPS:
            continue
            
        scen_files = sorted(glob.glob(os.path.join(SCEN_DIR, f"{map_name}-random-*.scen")))
        
        # We need BD heuristics for the test set too!
        limit = 25 if map_name in HELD_OUT_TEST else 128
        
        for scen_path in scen_files[:limit]:
            # Queue exactly ONE heuristic calculation per scenario
            bd_jobs.append((map_path, scen_path))
            
            # Queue trajectory generations (skip for held-out test set)
            if map_name not in HELD_OUT_TEST:
                for n in agent_counts:
                    if n > 200 and "32-32" in map_name: continue
                    traj_jobs.append((map_path, scen_path, n))

    # Run Phase 1
    print(f"--- PHASE 1: Generating Heuristics ({len(bd_jobs)} files) ---")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(generate_scenario_bd, m, s) for m, s in bd_jobs]
        for future in as_completed(futures):
            # Print instantly when a job finishes
            print(future.result())

    # Run Phase 2
    print(f"\n--- PHASE 2: Generating Expert Trajectories ({len(traj_jobs)} files) ---")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(generate_scenario_trajectory, m, s, a): (m, s, a) for m, s, a in traj_jobs}
        for i, future in enumerate(as_completed(futures)):
            if i % 50 == 0: 
                print(f"Progress: {i}/{len(traj_jobs)} | {future.result()}")

if __name__ == "__main__":
    process_benchmark_parallel()
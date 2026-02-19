import os
import subprocess
import re
import numpy as np
from scipy.signal import savgol_filter
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration
EECBS_BIN = "./build/eecbs"
DATA_DIR = "data"
MAP_DIR = os.path.join(DATA_DIR, "mapf-map")
SCEN_DIR = os.path.join(DATA_DIR, "scen-random")
OUTPUT_NPZ_DIR = "data/flow_training_data_multi"

OMITTED_MAPS = {"brc202d", "orz900", "maze-128-128-1", "maze-128-128-10"}
WINDOW_LENGTH = 5 
POLY_ORDER = 2    
MAX_WORKERS = max(1, os.cpu_count() - 2) # Leave a couple of cores free

os.makedirs(OUTPUT_NPZ_DIR, exist_ok=True)

def parse_paths_txt(file_path):
    """Parses the Agent 0: (r,c)->(r,c)-> format into a Numpy array (N, T, 2)."""
    with open(file_path, "r") as f:
        lines = f.readlines()
        
    all_paths = []
    for line in lines:
        if not line.startswith("Agent"):
            continue
        coords = re.findall(r"\((\d+),(\d+)\)", line)
        path = np.array([[int(r), int(c)] for r, c in coords], dtype=np.float32)
        all_paths.append(path)
        
    if not all_paths:
        return np.array([])

    max_len = max(len(p) for p in all_paths)
    N = len(all_paths)
    
    paths_array = np.zeros((N, max_len, 2), dtype=np.float32)
    for i, p in enumerate(all_paths):
        paths_array[i, :len(p), :] = p
        paths_array[i, len(p):, :] = p[-1] 
        
    return paths_array

def smooth_and_extract_velocities(paths_array):
    """Applies Savitzky-Golay filter to smooth coordinates and extract velocity vectors."""
    if len(paths_array) == 0:
        return paths_array, paths_array

    N, T, D = paths_array.shape
    window = min(WINDOW_LENGTH, T)
    if window % 2 == 0:
        window -= 1
        
    if window < 3:
        velocities = np.gradient(paths_array, axis=1)
        return paths_array, velocities

    smoothed_positions = savgol_filter(paths_array, window_length=window, polyorder=POLY_ORDER, axis=1, deriv=0)
    velocities = savgol_filter(paths_array, window_length=window, polyorder=POLY_ORDER, axis=1, deriv=1)
    
    return smoothed_positions, velocities

def process_single_scenario(map_path, scen_path, num_agents):
    """Worker function to process a single scenario."""
    scen_name = os.path.basename(scen_path).replace(".scen", "")
    tmp_path_file = f"tmp_paths_{scen_name}_{num_agents}.txt" 
    out_file = os.path.join(OUTPUT_NPZ_DIR, f"{scen_name}_{num_agents}.npz")
    
    if os.path.exists(out_file):
        return f"Skipped {scen_name} (already exists)"
    
    cmd = [
        EECBS_BIN,
        "-m", map_path,
        "-a", scen_path,
        "-k", str(num_agents),
        "--outputPaths", tmp_path_file,
        "--suboptimality", "1.2"
    ]
    
    try:
        # Added a 60-second timeout to prevent infinite hangs on hard instances
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        
        if os.path.exists(tmp_path_file):
            paths_discrete = parse_paths_txt(tmp_path_file)
            
            if len(paths_discrete) > 0:
                positions, velocities = smooth_and_extract_velocities(paths_discrete)
                np.savez_compressed(
                    out_file, 
                    discrete_positions=paths_discrete, 
                    smoothed_positions=positions,
                    expert_velocities=velocities
                )
                os.remove(tmp_path_file)
                return f"Success: {scen_name}"
            else:
                os.remove(tmp_path_file)
                return f"Failed: {scen_name} (No paths found in output)"
                
    except subprocess.TimeoutExpired:
        if os.path.exists(tmp_path_file):
            os.remove(tmp_path_file)
        return f"Timeout: {scen_name} (Exceeded 60s)"
    except subprocess.CalledProcessError:
        if os.path.exists(tmp_path_file):
            os.remove(tmp_path_file)
        return f"Failed: {scen_name} (EECBS error)"
    except Exception as e:
        if os.path.exists(tmp_path_file):
            os.remove(tmp_path_file)
        return f"Error: {scen_name} ({str(e)})"
    
    return f"Failed: {scen_name} (Unknown error)"

def process_benchmark_parallel():
    map_files = glob.glob(os.path.join(MAP_DIR, "*.map"))
    
    jobs = []
    for map_path in map_files:
        map_name = os.path.basename(map_path).replace(".map", "")
        if map_name in OMITTED_MAPS:
            continue
            
        scen_files = glob.glob(os.path.join(SCEN_DIR, f"{map_name}-random-*.scen"))
        
        for scen_path in scen_files: 
            jobs.append((map_path, scen_path, 50))
            
    print(f"Starting {len(jobs)} jobs across {MAX_WORKERS} CPU cores...")
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_scenario, m, s, a): (m, s) for m, s, a in jobs}
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if i % 10 == 0: 
                print(f"[{i}/{len(jobs)}] {result}")

if __name__ == "__main__":
    process_benchmark_parallel()
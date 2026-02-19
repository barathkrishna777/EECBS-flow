import os
import subprocess
import re
import numpy as np
from scipy.signal import savgol_filter
import glob

# Configuration
EECBS_BIN = "./build/eecbs"
DATA_DIR = "data"
MAP_DIR = os.path.join(DATA_DIR, "mapf-map")
SCEN_DIR = os.path.join(DATA_DIR, "scen-random")
OUTPUT_NPZ_DIR = "data/flow_training_data"

# The maps omitted in "Work Smarter Not Harder"
OMITTED_MAPS = {"brc202d", "orz900", "maze-128-128-1", "maze-128-128-10"}

# Smoothing parameters
WINDOW_LENGTH = 5  # Must be odd. Larger = smoother but less faithful to the grid
POLY_ORDER = 2     # Polynomial order for the Savitzky-Golay filter

os.makedirs(OUTPUT_NPZ_DIR, exist_ok=True)

def run_eecbs(map_file, scen_file, num_agents, output_path_file):
    """Runs the EECBS solver and saves paths to a text file."""
    cmd = [
        EECBS_BIN,
        "-m", map_file,
        "-a", scen_file,
        "-k", str(num_agents),
        "--outputPaths", output_path_file,
        "--suboptimality", "1.2" # EECBS default from driver.cpp
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to solve {scen_file} with {num_agents} agents.")
        return False

def parse_paths_txt(file_path):
    """Parses the Agent 0: (r,c)->(r,c)-> format into a Numpy array (N, T, 2)."""
    with open(file_path, "r") as f:
        lines = f.readlines()
        
    all_paths = []
    for line in lines:
        if not line.startswith("Agent"):
            continue
        # Extract coordinates using regex
        coords = re.findall(r"\((\d+),(\d+)\)", line)
        path = np.array([[int(r), int(c)] for r, c in coords], dtype=np.float32)
        all_paths.append(path)
        
    # Pad to max length so we can create a dense numpy array
    max_len = max(len(p) for p in all_paths)
    N = len(all_paths)
    
    # Fill array. If an agent reaches its goal early, it rests there (repeats last position)
    paths_array = np.zeros((N, max_len, 2), dtype=np.float32)
    for i, p in enumerate(all_paths):
        paths_array[i, :len(p), :] = p
        paths_array[i, len(p):, :] = p[-1] # Pad with final goal position
        
    return paths_array

def smooth_and_extract_velocities(paths_array):
    """
    Applies Savitzky-Golay filter to smooth coordinates and extract velocity vectors.
    paths_array: shape (N, T, 2)
    """
    N, T, D = paths_array.shape
    
    # If the path is too short to smooth, pad it or skip smoothing
    window = min(WINDOW_LENGTH, T)
    if window % 2 == 0:
        window -= 1
        
    if window < 3:
        # Fallback to discrete differences if trajectory is extremely short
        velocities = np.gradient(paths_array, axis=1)
        return paths_array, velocities

    # 1. Smooth the positions (deriv=0)
    smoothed_positions = savgol_filter(paths_array, window_length=window, polyorder=POLY_ORDER, axis=1, deriv=0)
    
    # 2. Extract continuous velocities (deriv=1 gives the first derivative / slope)
    velocities = savgol_filter(paths_array, window_length=window, polyorder=POLY_ORDER, axis=1, deriv=1)
    
    return smoothed_positions, velocities

def process_benchmark():
    # Find all map files
    map_files = glob.glob(os.path.join(MAP_DIR, "*.map"))
    
    for map_path in map_files:
        map_name = os.path.basename(map_path).replace(".map", "")
        
        if map_name in OMITTED_MAPS:
            print(f"Skipping omitted map: {map_name}")
            continue
            
        print(f"Processing map: {map_name}")
        scen_files = glob.glob(os.path.join(SCEN_DIR, f"{map_name}-random-*.scen"))
        
        # Test just the first few scenarios per map for testing
        for scen_path in scen_files[:5]: 
            scen_name = os.path.basename(scen_path).replace(".scen", "")
            tmp_path_file = f"tmp_paths_{scen_name}.txt"
            
            # Example: Running for 50 agents. (You can iterate over agent counts as needed)
            num_agents = 50 
            
            success = run_eecbs(map_path, scen_path, num_agents, tmp_path_file)
            if success and os.path.exists(tmp_path_file):
                paths_discrete = parse_paths_txt(tmp_path_file)
                
                # Convert to continuous context for Flow Matching
                positions, velocities = smooth_and_extract_velocities(paths_discrete)
                
                # Save as npz. Your Flow model dataloader can read these easily
                out_file = os.path.join(OUTPUT_NPZ_DIR, f"{scen_name}_{num_agents}.npz")
                np.savez_compressed(
                    out_file, 
                    discrete_positions=paths_discrete, 
                    smoothed_positions=positions,
                    expert_velocities=velocities
                )
                
                os.remove(tmp_path_file) # Clean up

if __name__ == "__main__":
    process_benchmark()
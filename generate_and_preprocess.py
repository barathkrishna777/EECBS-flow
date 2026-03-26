"""
Unified pipeline: Generate missing EECBS trajectories + BD heuristics,
preprocess into PyG .pt files, then clean up raw data to free disk space.

Expected directory layout:
    barath/
        EECBS-flow/          <-- run this script from here
            build/eecbs      <-- EECBS binary
            generate_and_preprocess.py
        Flow-CS-PIBT/        <-- sibling repo with data/ and main_pys/
            data/
            main_pys/

Usage:
    cd EECBS-flow
    python generate_and_preprocess.py [--workers 32] [--skip-generation] [--skip-cleanup] [--custom-only]
"""
import os
import sys
import glob
import shutil
import argparse
import subprocess
import re
import numpy as np
import torch
from scipy.signal import savgol_filter
from collections import deque, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Resolve paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EECBS_REPO = SCRIPT_DIR  # This script lives in EECBS-flow/
FLOW_REPO = os.path.join(os.path.dirname(SCRIPT_DIR), "Flow-CS-PIBT")

# Add Flow-CS-PIBT to path so we can import main_pys
sys.path.insert(0, FLOW_REPO)
from main_pys.model_inputs import create_data_object, normalize_graph_data

# ============================================================
# Configuration — paths resolve relative to repo locations
# ============================================================
DATA_DIR = os.path.join(FLOW_REPO, "data")
MAP_DIR = os.path.join(DATA_DIR, "mapf-map")
SCEN_DIR = os.path.join(DATA_DIR, "scen-random")
TRAJ_DIR = os.path.join(DATA_DIR, "flow_training_data_multi")
BD_DIR = os.path.join(DATA_DIR, "bd_npzs", "large_scale")
PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed")
EECBS_BIN = os.path.join(EECBS_REPO, "build", "eecbs")

# Maps to skip entirely
OMITTED_MAPS = {"brc202d", "orz900", "maze-128-128-1", "maze-128-128-10"}

# Test maps: generate BDs but NOT trajectories (for evaluation only)
# Must match Rishi's paper — all 8 held-out maps excluded from training.
HELD_OUT_TEST = {
    "Paris_1_256", "empty-48-48", "maze-128-128-2", "random-64-64-10",
    "random-32-32-10", "warehouse-10-20-10-2-1", "den312d", "den520d"
}

# Full range of agent counts including high-density
AGENT_COUNTS = [20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

MANIFEST_FILE = os.path.join(PREPROCESSED_DIR, "preprocessed_manifest.txt")

# Savitzky-Golay filter settings
WINDOW_LENGTH = 3
POLY_ORDER = 2

# Preprocessing settings
K = 4  # local region size
M = 5  # nearest neighbors


# ============================================================
# Manifest: tracks which trajectories have been preprocessed
# so we don't regenerate them after cleanup deletes raw .npz
# ============================================================
def load_manifest():
    """Load set of trajectory basenames that have been preprocessed."""
    if not os.path.exists(MANIFEST_FILE):
        return set()
    with open(MANIFEST_FILE, 'r') as f:
        return set(line.strip() for line in f if line.strip())


def save_manifest(entries):
    """Append new trajectory basenames to the manifest."""
    os.makedirs(os.path.dirname(MANIFEST_FILE), exist_ok=True)
    with open(MANIFEST_FILE, 'a') as f:
        for entry in entries:
            f.write(entry + '\n')


def seed_manifest_from_preprocessed_dir(preprocessed_dir, map_dir, scen_dir):
    """Reconstruct manifest when raw .npz files were already cleaned up but
    preprocessed .pt files exist. Enumerates all (map, scenario, agent_count)
    combos the previous pipeline would have attempted, so only truly new
    agent counts (700, 900) trigger EECBS generation."""
    PREVIOUS_AGENT_COUNTS = [20, 50, 100, 200, 300, 400, 500, 600, 800, 1000]

    entries = set()
    map_files = glob.glob(os.path.join(map_dir, "*.map"))
    for map_path in map_files:
        map_name = os.path.basename(map_path).replace(".map", "")
        if map_name in OMITTED_MAPS or map_name in HELD_OUT_TEST:
            continue
        scen_files = sorted(glob.glob(
            os.path.join(scen_dir, f"{map_name}-random-*.scen")))
        for scen_path in scen_files[:128]:
            scen_name = os.path.basename(scen_path).replace(".scen", "")
            for n in PREVIOUS_AGENT_COUNTS:
                if n > 200 and ("32-32" in map_name or "32_32" in map_name):
                    continue
                if n > 20 and "corridor" in map_name:
                    continue
                entries.add(f"{scen_name}_{n}.npz")

    save_manifest(sorted(entries))
    return entries


# ============================================================
# EECBS helper functions (from generate_flow_data_multi.py)
# ============================================================
def parse_paths_txt(file_path):
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
    paths_array = np.zeros((len(all_paths), max_len, 2), dtype=np.float32)
    for i, p in enumerate(all_paths):
        paths_array[i, :len(p), :] = p
        paths_array[i, len(p):, :] = p[-1]
    return paths_array


def smooth_and_extract_velocities(paths_array):
    if len(paths_array) == 0:
        return paths_array, paths_array
    N, T, D = paths_array.shape
    window = min(WINDOW_LENGTH, T)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return paths_array, np.gradient(paths_array, axis=1)
    smoothed = savgol_filter(paths_array, window_length=window,
                             polyorder=POLY_ORDER, axis=1, deriv=0)
    velocities = savgol_filter(paths_array, window_length=window,
                               polyorder=POLY_ORDER, axis=1, deriv=1)
    return smoothed, velocities


def read_map_raw(map_file):
    """Read map without padding (for EECBS generation)."""
    with open(map_file, 'r') as f:
        f.readline()
        h = int(f.readline().split()[1])
        w = int(f.readline().split()[1])
        f.readline()
        map_data = np.zeros((h, w), dtype=np.int8)
        for r in range(h):
            line = f.readline().strip()
            for c in range(w):
                if line[c] in ['@', 'T', 'O']:
                    map_data[r, c] = 1
    return map_data


def read_map_padded(map_file, k):
    """Read map with padding (for preprocessing)."""
    return np.pad(read_map_raw(map_file), k, 'constant', constant_values=1)


def parse_scenario_goals(scen_file, max_agents=1000):
    goals = []
    with open(scen_file, 'r') as f:
        f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 8:
                goals.append([int(parts[7]), int(parts[6])])
            if len(goals) == max_agents:
                break
    return np.array(goals)


def compute_bd_heuristic(map_data, goals):
    H, W = map_data.shape
    N = len(goals)
    bd_array = np.full((N, H, W), 10000, dtype=np.int16)
    for i, (gr, gc) in enumerate(goals):
        if map_data[gr, gc] == 1:
            continue
        queue = deque([(gr, gc, 0)])
        bd_array[i, gr, gc] = 0
        while queue:
            r, c, dist = queue.popleft()
            ndist = dist + 1
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if map_data[nr, nc] == 0 and bd_array[i, nr, nc] == 10000:
                        bd_array[i, nr, nc] = ndist
                        queue.append((nr, nc, ndist))
    return bd_array


# ============================================================
# Phase 1: Generate BD heuristics
# ============================================================
def generate_scenario_bd(map_path, scen_path):
    scen_name = os.path.basename(scen_path).replace(".scen", "")
    map_name = os.path.basename(map_path).replace(".map", "")
    bd_key = f"{map_name}-random-{scen_name.split('-random-')[-1]}"
    out_bd_file = os.path.join(BD_DIR, f"{scen_name}_bds.npz")

    if os.path.exists(out_bd_file):
        return "skip"

    try:
        map_data = read_map_raw(map_path)
        goals = parse_scenario_goals(scen_path, max_agents=1000)
        bd_array = compute_bd_heuristic(map_data, goals)

        tmp_file = out_bd_file + f".{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp_file, **{bd_key: bd_array})
        os.rename(tmp_file, out_bd_file)
        return f"BD Generated: {scen_name}"
    except Exception as e:
        return f"BD Error: {scen_name} ({e})"


# ============================================================
# Phase 2: Generate EECBS trajectories
# ============================================================
def generate_scenario_trajectory(map_path, scen_path, num_agents):
    scen_name = os.path.basename(scen_path).replace(".scen", "")
    out_traj_file = os.path.join(TRAJ_DIR, f"{scen_name}_{num_agents}.npz")
    tmp_path_file = f"tmp_{scen_name}_{num_agents}_{os.getpid()}.txt"

    if os.path.exists(out_traj_file):
        return "skip"

    cmd = [
        EECBS_BIN, "-m", map_path, "-a", scen_path, "-k", str(num_agents),
        "--outputPaths", tmp_path_file, "--suboptimality", "2.0"
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=120)
        if os.path.exists(tmp_path_file):
            paths_discrete = parse_paths_txt(tmp_path_file)
            if len(paths_discrete) > 0:
                positions, velocities = smooth_and_extract_velocities(paths_discrete)
                np.savez_compressed(out_traj_file,
                                    discrete_positions=paths_discrete,
                                    smoothed_positions=positions,
                                    expert_velocities=velocities)
                os.remove(tmp_path_file)
                return f"Traj Done: {scen_name} (N={num_agents})"
    except Exception as e:
        if os.path.exists(tmp_path_file):
            os.remove(tmp_path_file)
        return f"Traj Fail: {scen_name}_{num_agents} ({e})"

    if os.path.exists(tmp_path_file):
        os.remove(tmp_path_file)
    return f"Traj Fail: {scen_name}_{num_agents}"


# ============================================================
# Phase 3: Preprocess trajectories into PyG .pt files
# ============================================================
def process_sample(args, maps, k, m, out_dir):
    """Build and save one PyG Data object."""
    npz_path, t_step, sample_idx = args
    out_path = os.path.join(out_dir, f"sample_{sample_idx:08d}.pt")
    if os.path.exists(out_path) and os.path.getsize(out_path) >= 100:
        return out_path

    try:
        with np.load(npz_path) as data:
            discrete_positions = data['discrete_positions']
            expert_velocities = data['expert_velocities']

        filename = os.path.basename(npz_path)
        map_name = filename.split("-random-")[0]
        grid_map = maps[map_name]

        cur_locs = discrete_positions[:, t_step, :].astype(float)
        rng = np.random.RandomState(sample_idx)
        noise = rng.normal(0, 0.15, cur_locs.shape)
        cur_locs_jittered = cur_locs + noise
        cur_locs_discrete = (np.round(cur_locs_jittered) + k).astype(int)

        max_r = grid_map.shape[0] - k - 1
        max_c = grid_map.shape[1] - k - 1
        cur_locs_discrete[:, 0] = np.clip(cur_locs_discrete[:, 0], k, max_r)
        cur_locs_discrete[:, 1] = np.clip(cur_locs_discrete[:, 1], k, max_c)

        target_velocity = expert_velocities[:, t_step, :]

        # Goal weighting
        speeds = np.linalg.norm(target_velocity, axis=1)
        is_parked = speeds < 0.01
        parked_ratio = np.mean(is_parked)
        weights = np.ones(target_velocity.shape[0], dtype=np.float32)
        moving_weight = 1.0 / (1.0 - parked_ratio + 1e-3)
        parked_weight = 1.0 / (parked_ratio + 1e-3)
        weights[~is_parked] = moving_weight
        weights[is_parked] = parked_weight
        weights = weights * (len(weights) / weights.sum())

        # BD loading
        scen_name = filename.replace('.npz', '').rsplit('_', 1)[0]
        bd_file_path = os.path.join(BD_DIR, f"{scen_name}_bds.npz")
        bd_key = f"{map_name}-random-{scen_name.split('-random-')[-1]}"
        with np.load(bd_file_path) as bd_data:
            bd_grid = bd_data[bd_key][:cur_locs.shape[0]].astype(np.float32)
        bd_grid = np.pad(bd_grid, ((0, 0), (k, k), (k, k)),
                         'constant', constant_values=10000)

        dummy_goals = np.zeros_like(cur_locs_discrete)

        graph_data = create_data_object(cur_locs_discrete, bd_grid, grid_map, k, m, dummy_goals)
        graph_data = normalize_graph_data(graph_data, k)
        graph_data.y = torch.tensor(target_velocity, dtype=torch.float32)
        graph_data.node_weights = torch.tensor(weights, dtype=torch.float32)

        torch.save(graph_data, out_path)
        return out_path
    except Exception:
        return None


def _pp_worker_init(maps_dict, k_val, m_val, out_dir_val):
    global _maps, _k, _m, _out_dir
    _maps = maps_dict
    _k = k_val
    _m = m_val
    _out_dir = out_dir_val


def _pp_worker_fn(args):
    return process_sample(args, _maps, _k, _m, _out_dir)


# ============================================================
# Main pipeline
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate EECBS data, preprocess, and clean up")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers (default: all CPU cores)")
    parser.add_argument("--out", default=PREPROCESSED_DIR,
                        help="Output dir for preprocessed .pt files")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip EECBS generation (only preprocess existing data)")
    parser.add_argument("--skip-cleanup", action="store_true",
                        help="Keep raw trajectory files after preprocessing")
    parser.add_argument("--custom-only", action="store_true",
                        help="Only process maps with 'custom_' in the name to prevent re-running old maps.")
    parser.add_argument("--include-heldout", action="store_true",
                        help="Also generate training trajectories for held-out test maps "
                             "(using scenarios 2..N, reserving scenario 1 for evaluation)")
    parser.add_argument("--heldout-train-limit", type=int, default=8,
                        help="Number of scenarios per held-out map for training (default: 8)")
    args = parser.parse_args()

    num_workers = args.workers if args.workers > 0 else cpu_count()

    # Ensure directories exist
    for d in [TRAJ_DIR, BD_DIR, args.out]:
        os.makedirs(d, exist_ok=True)

    if args.include_heldout:
        print(f"** --include-heldout: generating training trajectories for held-out maps")
        print(f"   Maps: {', '.join(sorted(HELD_OUT_TEST))}")
        print(f"   Scenarios: 2 through {1 + args.heldout_train_limit} "
              f"(scenario 1 reserved for eval)")
        print()

    # ==========================================================
    # Phase 0: Show current dataset distribution
    # ==========================================================
    existing_trajs = glob.glob(os.path.join(TRAJ_DIR, "*.npz"))
    agent_dist = defaultdict(int)
    for f in existing_trajs:
        try:
            n = int(os.path.basename(f).replace('.npz', '').rsplit('_', 1)[1])
            agent_dist[n] += 1
        except (ValueError, IndexError):
            pass

    manifest = load_manifest()

    print("=" * 60)
    print("CURRENT DATASET DISTRIBUTION")
    print("=" * 60)
    if existing_trajs:
        print("  Raw trajectory .npz files:")
        for k_count in sorted(agent_dist.keys()):
            print(f"    {k_count:>6} agents: {agent_dist[k_count]:>5} files")
        print(f"    {'Total':>6}: {len(existing_trajs):>5} files")
    else:
        print("  No raw .npz files (cleaned up after previous preprocessing)")
    if manifest:
        manifest_dist = defaultdict(int)
        for entry in manifest:
            try:
                n = int(entry.replace('.npz', '').rsplit('_', 1)[1])
                manifest_dist[n] += 1
            except (ValueError, IndexError):
                pass
        print(f"  Preprocessed manifest: {len(manifest):,} trajectories already done")
        for k_count in sorted(manifest_dist.keys()):
            print(f"    {k_count:>6} agents: {manifest_dist[k_count]:>5} preprocessed")
    existing_pt = len(glob.glob(os.path.join(args.out, "sample_*.pt")))
    print(f"  Preprocessed .pt samples: {existing_pt:,}")
    print()

    if not args.skip_generation:
        # Check EECBS binary
        if not os.path.exists(EECBS_BIN):
            print(f"WARNING: EECBS binary not found at {EECBS_BIN}")
            print("Skipping generation. Build EECBS first or use --skip-generation")
            args.skip_generation = True

    if not args.skip_generation:
        # ==================================================
        # Phase 1: Generate missing BD heuristics
        # ==================================================
        map_files = glob.glob(os.path.join(MAP_DIR, "*.map"))
        
        if args.custom_only:
            map_files = [m for m in map_files if "custom_" in os.path.basename(m)]
            
        bd_jobs = []
        traj_jobs = []

        for map_path in map_files:
            map_name = os.path.basename(map_path).replace(".map", "")
            if map_name in OMITTED_MAPS:
                continue

            scen_files = sorted(glob.glob(
                os.path.join(SCEN_DIR, f"{map_name}-random-*.scen")))
            limit = 25 if map_name in HELD_OUT_TEST else 128

            for scen_path in scen_files[:limit]:
                bd_jobs.append((map_path, scen_path))

                scen_num = int(os.path.basename(scen_path).rsplit("-", 1)[1].replace(".scen", ""))
                generate_traj = False

                if map_name not in HELD_OUT_TEST:
                    generate_traj = True
                elif args.include_heldout and 2 <= scen_num <= (1 + args.heldout_train_limit):
                    # Scenario 1 is reserved for evaluation; use 2..N for training
                    generate_traj = True

                if generate_traj:
                    for n in AGENT_COUNTS:
                        if n > 200 and ("32-32" in map_name or "32_32" in map_name):
                            continue
                        if n > 20 and "corridor" in map_name:
                            continue
                        traj_jobs.append((map_path, scen_path, n))

        # Filter to only missing jobs (check raw .npz AND manifest of preprocessed)
        already_preprocessed = load_manifest()

        # Auto-seed manifest on first run after adding manifest tracking
        if not already_preprocessed and existing_trajs:
            seed_entries = set(os.path.basename(f) for f in existing_trajs)
            save_manifest(sorted(seed_entries))
            already_preprocessed = seed_entries
            print(f"  Auto-seeded manifest from {len(seed_entries)} existing raw .npz files")
        elif not already_preprocessed and not existing_trajs and existing_pt > 0:
            print("  No manifest and raw .npz cleaned up -- reconstructing from pipeline state...")
            already_preprocessed = seed_manifest_from_preprocessed_dir(
                args.out, MAP_DIR, SCEN_DIR)
            print(f"  Reconstructed manifest with {len(already_preprocessed):,} entries "
                  f"(previous agent counts). Only new counts will be generated.")

        traj_jobs_needed = []
        for map_path, scen_path, n in traj_jobs:
            scen_name = os.path.basename(scen_path).replace(".scen", "")
            npz_basename = f"{scen_name}_{n}.npz"
            out_file = os.path.join(TRAJ_DIR, npz_basename)
            if not os.path.exists(out_file) and npz_basename not in already_preprocessed:
                traj_jobs_needed.append((map_path, scen_path, n))

        print(f"Phase 1: BD Heuristics ({len(bd_jobs)} scenarios to check)")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(generate_scenario_bd, m, s)
                       for m, s in bd_jobs]
            generated = 0
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="BD Heuristics"):
                result = future.result()
                if result != "skip":
                    generated += 1
                    if generated <= 5:
                        print(f"  {result}")
        print(f"  BD generation done ({generated} new files)\n")

        print(f"Phase 2: EECBS Trajectories ({len(traj_jobs_needed)} new to generate, "
              f"{len(traj_jobs) - len(traj_jobs_needed)} already exist)")

        if traj_jobs_needed:
            # Show what we're generating
            new_dist = defaultdict(int)
            for _, _, n in traj_jobs_needed:
                new_dist[n] += 1
            for k_count in sorted(new_dist.keys()):
                print(f"  Will generate {new_dist[k_count]} files for {k_count} agents")

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(generate_scenario_trajectory, m, s, a): (m, s, a)
                           for m, s, a in traj_jobs_needed}
                done = 0
                failed = 0
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc="EECBS Trajectories"):
                    result = future.result()
                    if "Fail" in result:
                        failed += 1
                    elif result != "skip":
                        done += 1
            print(f"  Trajectory generation done ({done} new, {failed} failed)\n")
        else:
            print("  All trajectories already exist!\n")

    # ==========================================================
    # Phase 3: Preprocess into PyG .pt files
    # ==========================================================
    print("Phase 3: Preprocessing into PyG .pt files")

    # Load maps
    print("  Loading maps...")
    maps = {}
    for map_path in glob.glob(os.path.join(MAP_DIR, "*.map")):
        map_name = os.path.basename(map_path).replace(".map", "")
        maps[map_name] = read_map_padded(map_path, K)
    print(f"  Loaded {len(maps)} maps")

    # Build sample index
    print("  Building sample index...")
    npz_files = sorted(glob.glob(os.path.join(TRAJ_DIR, "*.npz")))
    
    # --- NEW OFFSET LOGIC ---
    existing_pt_files = glob.glob(os.path.join(args.out, "sample_*.pt"))
    start_idx = 0
    if existing_pt_files:
        # Find the highest existing sample index so we don't overwrite old data
        indices = [int(os.path.basename(f).replace('sample_', '').replace('.pt', '')) for f in existing_pt_files]
        start_idx = max(indices) + 1
    # ------------------------

    index = []
    for f in tqdm(npz_files, desc="  Scanning"):
        try:
            with np.load(f) as data:
                T = data['discrete_positions'].shape[1]
                for t in range(T):
                    # USE start_idx HERE:
                    index.append((f, t, start_idx + len(index)))
        except Exception:
            pass
    print(f"  Total samples: {len(index):,}")

    # Count how many already exist
    existing_pt = len(glob.glob(os.path.join(args.out, "sample_*.pt")))
    print(f"  Already preprocessed: {existing_pt:,}")
    print(f"  To process: up to {len(index):,}")

    # Process
    print(f"  Processing with {num_workers} workers -> {args.out}/")
    with Pool(num_workers, initializer=_pp_worker_init,
              initargs=(maps, K, M, args.out)) as pool:
        results = list(tqdm(
            pool.imap_unordered(_pp_worker_fn, index, chunksize=64),
            total=len(index),
            desc="  Preprocessing"
        ))

    succeeded = sum(1 for r in results if r is not None)
    failed = sum(1 for r in results if r is None)
    print(f"  Done! {succeeded:,} samples OK, {failed:,} failed.\n")

    # Update manifest with all .npz files that were successfully preprocessed
    newly_preprocessed = set(os.path.basename(f) for f, _, _ in index)
    already_in_manifest = load_manifest()
    new_entries = newly_preprocessed - already_in_manifest
    if new_entries:
        save_manifest(sorted(new_entries))
        print(f"  Manifest updated: {len(new_entries):,} new entries "
              f"(total: {len(already_in_manifest) + len(new_entries):,})")

    # Show final distribution
    final_trajs = glob.glob(os.path.join(TRAJ_DIR, "*.npz"))
    final_dist = defaultdict(int)
    for f in final_trajs:
        try:
            n = int(os.path.basename(f).replace('.npz', '').rsplit('_', 1)[1])
            final_dist[n] += 1
        except (ValueError, IndexError):
            pass

    print("=" * 60)
    print("FINAL DATASET DISTRIBUTION")
    print("=" * 60)
    for k_count in sorted(final_dist.keys()):
        print(f"  {k_count:>6} agents: {final_dist[k_count]:>5} files")
    print(f"  {'Total':>6}: {len(final_trajs):>5} files")
    print(f"  Preprocessed .pt files: {succeeded:,}")
    print()

    # ==========================================================
    # Phase 4: Clean up raw data to free disk space
    # ==========================================================
    if not args.skip_cleanup:
        print("Phase 4: Cleaning up to free disk space")

        # Delete raw trajectory NPZs (now redundant — data is in .pt files)
        traj_size = sum(os.path.getsize(f) for f in glob.glob(
            os.path.join(TRAJ_DIR, "*.npz")))
        print(f"  Deleting raw trajectories ({traj_size / 1e9:.1f} GB)...")
        shutil.rmtree(TRAJ_DIR, ignore_errors=True)

        # Delete tmpFolder if it exists
        tmp_folder = os.path.join(DATA_DIR, "tmpFolder")
        if os.path.exists(tmp_folder):
            tmp_size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, dn, filenames in os.walk(tmp_folder)
                for f in filenames
            )
            print(f"  Deleting tmpFolder ({tmp_size / 1e9:.1f} GB)...")
            shutil.rmtree(tmp_folder, ignore_errors=True)

        # Delete wandb logs
        wandb_dir = "wandb"
        if os.path.exists(wandb_dir):
            wandb_size = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, dn, filenames in os.walk(wandb_dir)
                for f in filenames
            )
            print(f"  Deleting wandb logs ({wandb_size / 1e6:.0f} MB)...")
            shutil.rmtree(wandb_dir, ignore_errors=True)

        print("  Cleanup done!")
        print()
        print("  Remaining data structure:")
        print("    data/all_maps.npz         (evaluation)")
        print("    data/mapf-map/            (map files)")
        print("    data/bd_npzs/             (BD heuristics)")
        print("    data/scen-random/         (scenario files)")
        print(f"    {args.out}/  (preprocessed training data)")
    else:
        print("Skipping cleanup (--skip-cleanup)")

    print("\nAll done!")


if __name__ == "__main__":
    main()
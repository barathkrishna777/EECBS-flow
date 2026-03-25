import os
import random
import glob
import numpy as np

MAP_DIR = "../Flow-CS-PIBT/data/mapf-map"
SCEN_OUT_DIR = "../Flow-CS-PIBT/data/scen-random"
SCENES_PER_MAP = 25
MAX_AGENTS = 1000

def get_free_cells(map_file):
    with open(map_file, 'r') as f:
        f.readline()
        height = int(f.readline().split()[1])
        width = int(f.readline().split()[1])
        f.readline()
        free_cells = []
        for r in range(height):
            line = f.readline().strip()
            for c in range(width):
                if line[c] not in ['@', 'T', 'O']:
                    free_cells.append((r, c))
    return free_cells, width, height

os.makedirs(SCEN_OUT_DIR, exist_ok=True)
custom_maps = glob.glob(os.path.join(MAP_DIR, "custom_*.map"))

for map_file in custom_maps:
    map_basename = os.path.basename(map_file)
    map_name = map_basename.replace(".map", "")
    free_cells, width, height = get_free_cells(map_file)
    
    # Dynamically scale agent count for smaller maps
    agents_for_this_map = min(MAX_AGENTS, len(free_cells) // 2)

    for scene_id in range(1, SCENES_PER_MAP + 1):
        scen_path = os.path.join(SCEN_OUT_DIR, f"{map_name}-random-{scene_id}.scen")
        
        with open(scen_path, 'w') as f:
            f.write("version 1\n")
            starts = random.sample(free_cells, agents_for_this_map)
            goals = random.sample(free_cells, agents_for_this_map)
            
            for i in range(agents_for_this_map):
                while goals[i] == starts[i]:
                    goals[i] = random.choice(free_cells)
                
                sx, sy = starts[i][1], starts[i][0]
                gx, gy = goals[i][1], goals[i][0]
                f.write(f"0\t{map_basename}\t{width}\t{height}\t{sx}\t{sy}\t{gx}\t{gy}\t0\n")
                
    print(f"Generated {SCENES_PER_MAP} scenarios for {map_name} (Max Agents: {agents_for_this_map})")
import random
import os

# Create a directory to store the custom maps
output_dir = "custom_maps"
os.makedirs(output_dir, exist_ok=True)

def save_moving_ai_map(filename, grid):
    """Saves a 2D list grid in the Moving AI .map format."""
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    
    with open(os.path.join(output_dir, filename), 'w') as f:
        f.write("type octile\n")
        f.write(f"height {height}\n")
        f.write(f"width {width}\n")
        f.write("map\n")
        for row in grid:
            f.write("".join(row) + "\n")
    print(f"Saved {filename}")

def generate_random_map(width, height, obstacle_pct=0.10):
    """Generates a random map with a specific obstacle percentage."""
    grid = []
    for _ in range(height):
        row = []
        for _ in range(width):
            # 10% chance of being an obstacle ('@'), 90% chance of being free space ('.')
            if random.random() < obstacle_pct:
                row.append('@')
            else:
                row.append('.')
        grid.append(row)
    return grid

def generate_corridor_map(length, width=3):
    """Generates a simple narrow corridor map."""
    grid = []
    for _ in range(length):
        row = ['.'] * width
        grid.append(row)
    return grid

# 1. Generate 6 random maps of varying sizes (10% obstacle density)
sizes = [(32, 32), (32, 32), (64, 64), (64, 64), (128, 128), (128, 128)]
for i, (w, h) in enumerate(sizes):
    map_grid = generate_random_map(w, h, obstacle_pct=0.10)
    save_moving_ai_map(f"custom_random_{w}_{h}_{i+1}.map", map_grid)

# 2. Generate 1 small corridor map
# For example, a 20x3 corridor
corridor_grid = generate_corridor_map(length=20, width=3)
save_moving_ai_map("custom_corridor_20_3.map", corridor_grid)
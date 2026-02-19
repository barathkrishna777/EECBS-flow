import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def read_map(map_file):
    """Reads the .map file and returns a 2D numpy array (0 for empty, 1 for obstacle)."""
    with open(map_file, 'r') as f:
        f.readline() # type octile
        height_line = f.readline()
        height = int(height_line.split()[1])
        width_line = f.readline()
        width = int(width_line.split()[1])
        f.readline() # map
        
        map_data = np.zeros((height, width), dtype=int)
        for r in range(height):
            line = f.readline().strip()
            for c in range(width):
                if line[c] in ['@', 'T', 'O']:
                    map_data[r, c] = 1
    return map_data

def visualize_npz(map_file, npz_file, num_agents_to_plot=3):
    """Plots the discrete path, smoothed path, and velocity vectors over the map."""
    # Load Data
    map_data = read_map(map_file)
    data = np.load(npz_file)
    
    discrete_pos = data['discrete_positions']
    smoothed_pos = data['smoothed_positions']
    expert_vel = data['expert_velocities']
    
    total_agents = discrete_pos.shape[0]
    agents_to_plot = min(num_agents_to_plot, total_agents)
    
    # Setup Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    # Display map (Greys colormap: 0=white, 1=black)
    ax.imshow(map_data, cmap='Greys', origin='upper')
    
    colors = plt.cm.get_cmap('tab10', agents_to_plot)
    
    for i in range(agents_to_plot):
        color = colors(i)
        
        # Paths are stored as (row, col) which translates to (y, x) in matplotlib
        disc_y, disc_x = discrete_pos[i, :, 0], discrete_pos[i, :, 1]
        smooth_y, smooth_x = smoothed_pos[i, :, 0], smoothed_pos[i, :, 1]
        vel_y, vel_x = expert_vel[i, :, 0], expert_vel[i, :, 1]
        
        # 1. Plot Discrete Path (Faint dotted lines with markers)
        ax.plot(disc_x, disc_y, marker='o', linestyle=':', color=color, 
                alpha=0.4, label=f'Agent {i} (Discrete)' if i==0 else "")
        
        # 2. Plot Smoothed Path (Solid line)
        ax.plot(smooth_x, smooth_y, linestyle='-', color=color, linewidth=2, 
                label=f'Agent {i} (Smoothed)' if i==0 else "")
        
        # 3. Plot Velocity Vectors
        # We skip every few frames if the path is long to avoid cluttering the arrows
        step = max(1, len(smooth_x) // 15) 
        ax.quiver(smooth_x[::step], smooth_y[::step], 
                  vel_x[::step], vel_y[::step], color=color, angles='xy', scale_units='xy', scale=1, width=0.005, alpha=0.8)
        
        # Mark Start and Goal
        ax.plot(disc_x[0], disc_y[0], marker='s', color=color, markersize=8, markeredgecolor='black')
        ax.plot(disc_x[-1], disc_y[-1], marker='*', color=color, markersize=12, markeredgecolor='black')

    ax.set_title(f"Flow Matching Data:\n{os.path.basename(npz_file)}")
    ax.set_xlabel("Columns (X)")
    ax.set_ylabel("Rows (Y)")
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', marker='o', linestyle=':', alpha=0.4, label='Discrete Grid Path'),
        Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='Smoothed Flow Path'),
        Line2D([0], [0], color='black', marker='s', linestyle='None', label='Start'),
        Line2D([0], [0], color='black', marker='*', linestyle='None', markersize=10, label='Goal')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize smoothed flow data.")
    parser.add_argument("--map", type=str, required=True, help="Path to the .map file")
    parser.add_argument("--npz", type=str, required=True, help="Path to the .npz file")
    parser.add_argument("--agents", type=int, default=3, help="Number of agents to plot")
    
    args = parser.parse_args()
    visualize_npz(args.map, args.npz, args.agents)
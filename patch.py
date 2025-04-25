import matplotlib.pyplot as plt
import numpy as np

# Parameters
grid_size = 9
highlight_rows = [3, 4, 5]
cell_size = 1

# Create figure and axes
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.axis('off')

# Draw grid with colors
for y in range(grid_size):
    for x in range(grid_size):
        color = 'lightgreen' if y in highlight_rows else 'lightgray'
        rect = plt.Rectangle((x, y), cell_size, cell_size, facecolor=color, edgecolor='gray')
        ax.add_patch(rect)

# Draw orange dot at the center (middle cell of the grid)
center_x = grid_size // 2 + 0.5
center_y = grid_size // 2 + 0.5
ax.plot(center_x - 0.5, center_y - 0.5, 'o', color='orangered', markersize=10)

# Set limits and save image
ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
plt.gca().invert_yaxis()  # Match the top-left origin look
plt.savefig("2d_grid_with_dot.png", transparent=True, bbox_inches='tight')
plt.close()



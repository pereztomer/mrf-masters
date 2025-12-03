import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio
import MRzeroCore as mr0
import torch

# ---------------- Load Sequence & Phantom ----------------
seq_file = r"gre_epi_96_half.seq"#
seq_file  =r"C:\Users\perez\PycharmProjects\epi_gre_seq\epi_gre\epi_gre_flip_angle_90_slice_thickness_0.003_matrix_size_126\epi_gre_flip_angle_90_slice_thickness_0.003_matrix_size_126_pe_1.seq"
gif_path = r"C:\Users\perez\PycharmProjects\epi_gre_seq\epi_gre\epi_gre_flip_angle_90_slice_thickness_0.003_matrix_size_126\seq.gif"

seq0 = mr0.Sequence.import_file(seq_file)
phantom_path = r"C:\Users\perez\Desktop\mrf_runs_laptop\numerical_brain_cropped.mat"
obj_p = mr0.VoxelGridPhantom.load_mat(phantom_path)
obj_p = obj_p.build()

# Simulate sequence
graph = mr0.compute_graph(seq0.cuda(), obj_p.cuda(), 200, 1e-3)
# signal = mr0.execute_graph(graph, seq0.cuda(), obj_p.cuda(), print_progress=True)

# Get k-space trajectory
kspace = seq0.cuda().get_kspace()
kx_ky = kspace[:, :2].cpu().numpy()

# ---------------- Figure Setup ----------------
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(kx_ky[:, 0].min() - 0.1, kx_ky[:, 0].max() + 0.1)
ax.set_ylim(kx_ky[:, 1].min() - 0.1, kx_ky[:, 1].max() + 0.1)
ax.set_xlabel('kx')
ax.set_ylabel('ky')
ax.set_title('k-space Trajectory Acquisition')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)

scatter = ax.scatter([], [], c=[], cmap='viridis', s=10, alpha=0.6)
current_point = ax.scatter([], [], c='red', s=100, marker='o', zorder=5)

# ---------------- Animation Function ----------------
def animate(frame):
    points_so_far = kx_ky[:frame + 1]
    colors = np.arange(frame + 1)
    scatter.set_offsets(points_so_far)
    scatter.set_array(colors)
    current_point.set_offsets([kx_ky[frame]])
    ax.set_title(f'k-space Trajectory - Step {frame + 1}/{len(kx_ky)}')
    return scatter, current_point

# ---------------- Create Animation ----------------
anim = animation.FuncAnimation(
    fig, animate, frames=len(kx_ky),
    interval=50, blit=False  # blit=False fixes saving issues
)

# ---------------- Save as GIF ----------------
# gif_path = 'gre_epi_96_half.gif'
anim.save(gif_path, writer='pillow', fps=20)
print(f"GIF saved to {gif_path}")


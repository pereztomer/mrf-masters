import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import MRzeroCore as mr0
import torch

# ---------------- Load Sequence & Phantom ----------------
seq_file = r"C:\Users\perez\PycharmProjects\epi_gre_seq\epi_gre\epi_gre_flip_angle_90_slice_thickness_0.003_matrix_size_36_v3\epi_gre_flip_angle_90_slice_thickness_0.003_matrix_size_36_pe_1.seq"
video_path = r"C:\Users\perez\PycharmProjects\epi_gre_seq\epi_gre\epi_gre_flip_angle_90_slice_thickness_0.003_matrix_size_36_v3\seq.mp4"

seq0 = mr0.Sequence.import_file(seq_file)

# Get k-space trajectory
kspace = seq0.cuda().get_kspace()
kx_ky = kspace[:, :2].cpu().numpy()

# ---------------- Figure Setup ----------------
fig, ax = plt.subplots(figsize=(8, 8))

# Add relative padding based on data range
kx_range = kx_ky[:, 0].max() - kx_ky[:, 0].min()
ky_range = kx_ky[:, 1].max() - kx_ky[:, 1].min()
padding = 0.1  # 10% padding

ax.set_xlim(kx_ky[:, 0].min() - padding * kx_range,
            kx_ky[:, 0].max() + padding * kx_range)
ax.set_ylim(kx_ky[:, 1].min() - padding * ky_range,
            kx_ky[:, 1].max() + padding * ky_range)

# Increase tick density
num_ticks = 25  # More ticks = denser numbers
ax.set_xticks(np.linspace(kx_ky[:, 0].min() - padding * kx_range,
                          kx_ky[:, 0].max() + padding * kx_range,
                          num_ticks))
ax.set_yticks(np.linspace(kx_ky[:, 1].min() - padding * ky_range,
                          kx_ky[:, 1].max() + padding * ky_range,
                          num_ticks))

ax.set_xlabel('kx')
ax.set_ylabel('ky')
ax.set_title('k-space Trajectory Acquisition')
ax.set_aspect('equal', adjustable='box')
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
    interval=25, blit=True  # blit=True for faster rendering
)

# ---------------- Save as Video with NVIDIA GPU encoding ----------------
Writer = animation.writers['ffmpeg']
writer = Writer(
    fps=40,
    metadata=dict(artist='Me'),
    bitrate=1800,
    extra_args=['-c:v', 'h264_nvenc']  # Use NVIDIA GPU encoder
)
anim.save(video_path, writer=writer)
print(f"Video saved to {video_path}")
import numpy as np
import cv2
import MRzeroCore as mr0
import torch
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path
import subprocess

# ---------------- Load Sequence ----------------
seq_file = r"C:\Users\perez\OneDrive - Technion\masters\scans\4.12.2025\epi_gre\single_shot\epi_gre_flip_angle_90_slice_thickness_0.003_matrix_size_48\epi_gre_flip_angle_90_slice_thickness_0.003_res_48_pe_enabled.seq"
video_path = seq_file.replace('.seq', '.mp4')
seq0 = mr0.Sequence.import_file(seq_file)

# Get k-space trajectory
kspace = seq0.cuda().get_kspace()
kx_ky = kspace[:, :2].cpu().numpy()

print(f"Total k-space points: {len(kx_ky)}")

# ---------------- Canvas Setup ----------------
width, height = 1280, 1280
padding = 0.1

# Calculate k-space bounds with padding
kx_min, kx_max = kx_ky[:, 0].min(), kx_ky[:, 0].max()
ky_min, ky_max = kx_ky[:, 1].min(), kx_ky[:, 1].max()
kx_range = kx_max - kx_min
ky_range = ky_max - ky_min

# Handle case where ky is constant (no phase encoding)
if ky_range == 0:
    ky_range = kx_range if kx_range > 0 else 1.0  # Use kx_range or default to 1
    ky_min -= ky_range / 2
    ky_max += ky_range / 2

# Handle case where kx is constant (unlikely but safe)
if kx_range == 0:
    kx_range = ky_range if ky_range > 0 else 1.0
    kx_min -= kx_range / 2
    kx_max += kx_range / 2

kx_min -= padding * kx_range
kx_max += padding * kx_range
ky_min -= padding * ky_range
ky_max += padding * ky_range

kx_range = kx_max - kx_min
ky_range = ky_max - ky_min

# Margins for axes and labels
margin_left, margin_right = 150, 50
margin_top, margin_bottom = 100, 150
plot_width = width - margin_left - margin_right
plot_height = height - margin_top - margin_bottom


# Coordinate transformation functions
def kspace_to_pixel(kx, ky):
    """Convert k-space coordinates to pixel coordinates"""
    px = margin_left + int((kx - kx_min) / kx_range * plot_width)
    py = margin_top + int((ky_max - ky) / ky_range * plot_height)
    return px, py


# ---------------- Create Base Canvas with Grid and Axes ----------------
def create_base_canvas():
    """Create the base canvas with grid, axes, and labels"""
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Draw grid lines
    num_grid_lines = 10
    for i in range(num_grid_lines + 1):
        x = margin_left + int(i * plot_width / num_grid_lines)
        cv2.line(canvas, (x, margin_top), (x, margin_top + plot_height), (220, 220, 220), 1)

        y = margin_top + int(i * plot_height / num_grid_lines)
        cv2.line(canvas, (margin_left, y), (margin_left + plot_width, y), (220, 220, 220), 1)

    # Draw axes
    zero_x, zero_y = kspace_to_pixel(0, 0)
    cv2.line(canvas, (zero_x, margin_top), (zero_x, margin_top + plot_height), (100, 100, 100), 2)
    cv2.line(canvas, (margin_left, zero_y), (margin_left + plot_width, zero_y), (100, 100, 100), 2)

    # Draw border
    cv2.rectangle(canvas, (margin_left, margin_top),
                  (margin_left + plot_width, margin_top + plot_height), (0, 0, 0), 2)

    # Add title
    cv2.putText(canvas, 'k-space Trajectory Acquisition', (width // 2 - 300, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    # Add axis labels
    cv2.putText(canvas, 'kx', (width // 2 - 20, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(canvas, 'ky', (30, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

    # Add tick labels (with actual numbers, not scientific notation)
    num_ticks = 30
    for i in range(num_ticks + 1):
        # X-axis ticks
        kx_val = kx_min + i * kx_range / num_ticks
        x = margin_left + int(i * plot_width / num_ticks)
        if abs(kx_val) >= 100:
            label = f"{kx_val:.0f}"
        elif abs(kx_val) >= 1:
            label = f"{kx_val:.1f}"
        else:
            label = f"{kx_val:.2f}"
        cv2.putText(canvas, label, (x - 40, height - margin_bottom + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Y-axis ticks
        ky_val = ky_max - i * ky_range / num_ticks
        y = margin_top + int(i * plot_height / num_ticks)
        if abs(ky_val) >= 100:
            label = f"{ky_val:.0f}"
        elif abs(ky_val) >= 1:
            label = f"{ky_val:.1f}"
        else:
            label = f"{ky_val:.2f}"
        cv2.putText(canvas, label, (10, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return canvas


# ---------------- Color Map ----------------
def get_color(idx, total):
    """Get color for point based on position in trajectory"""
    t = idx / total
    if t < 0.5:
        b = int(255 * (1 - 2 * t))
        g = int(255 * 2 * t)
        r = int(128 * t)
    else:
        b = 0
        g = int(255)
        r = int(255 * 2 * (t - 0.5))
    return (b, g, r)


# ---------------- Render Chunk Function ----------------
def render_chunk(args):
    """Render a chunk of frames in parallel"""
    chunk_id, start_idx, end_idx, pixel_coords, base_canvas_with_points, output_path, total_points = args

    fps = 40
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not open video writer for chunk {chunk_id}")
        return False

    # Start with the base canvas (already has points 0 to start_idx-1 drawn)
    current_frame = base_canvas_with_points.copy()

    # Render frames for this chunk
    for frame_idx in range(start_idx, end_idx):
        # Draw new point
        px, py = pixel_coords[frame_idx]
        color = get_color(frame_idx, total_points)
        cv2.circle(current_frame, (px, py), 3, color, -1)

        # Create frame with marker
        frame_with_marker = current_frame.copy()
        cv2.circle(frame_with_marker, (px, py), 8, (0, 0, 255), 2)

        # Add frame counter
        text = f"Step {frame_idx + 1}/{total_points}"
        cv2.putText(frame_with_marker, text, (width - 300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        out.write(frame_with_marker)

    out.release()
    return True


# ---------------- Main Parallel Processing ----------------
if __name__ == '__main__':
    # Determine number of processes
    num_processes = mp.cpu_count() - 1
    print(f"Using {num_processes} parallel processes")

    # Create base canvas
    print("Creating base canvas...")
    base_canvas = create_base_canvas()

    # Pre-compute pixel coordinates
    print("Computing pixel coordinates...")
    pixel_coords = np.array([kspace_to_pixel(kx, ky) for kx, ky in kx_ky])

    # Divide work into chunks
    total_frames = len(kx_ky)
    chunk_size = total_frames // num_processes
    chunks = []

    temp_dir = Path(video_path).parent / "temp_chunks"
    temp_dir.mkdir(exist_ok=True)

    print("Preparing chunks...")
    for i in range(num_processes):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_processes - 1 else total_frames

        # Create starting canvas for this chunk (base + all points up to start_idx)
        canvas_with_points = base_canvas.copy()
        for j in range(start_idx):
            px, py = pixel_coords[j]
            color = get_color(j, total_frames)
            cv2.circle(canvas_with_points, (px, py), 3, color, -1)

        output_path = str(temp_dir / f"chunk_{i}.mp4")
        chunks.append((i, start_idx, end_idx, pixel_coords, canvas_with_points, output_path, total_frames))

    # Render chunks in parallel
    print("Rendering chunks in parallel...")
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(render_chunk, chunks), total=len(chunks), desc="Rendering chunks"))

    if not all(results):
        print("Error: Some chunks failed to render")
        exit(1)

    # Concatenate videos using ffmpeg
    print("Concatenating video chunks...")
    concat_file = temp_dir / "concat_list.txt"
    with open(concat_file, 'w') as f:
        for i in range(num_processes):
            f.write(f"file 'chunk_{i}.mp4'\n")

    # Use ffmpeg to concatenate
    cmd = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', str(concat_file),
        '-c', 'copy',
        video_path
    ]

    subprocess.run(cmd, check=True)

    # Cleanup temp files
    print("Cleaning up temporary files...")
    for i in range(num_processes):
        (temp_dir / f"chunk_{i}.mp4").unlink()
    concat_file.unlink()
    temp_dir.rmdir()

    print(f"Video saved to {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"Video duration: {total_frames / 40:.2f} seconds")
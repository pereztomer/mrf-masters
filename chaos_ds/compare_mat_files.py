from scipy.io import loadmat
import numpy as np

# Load both files
working = loadmat(r"C:\Users\perez\Desktop\mrf_runs\numerical_brain_cropped.mat")
mine = loadmat(r"C:\Users\perez\Desktop\phantom\abdominal_phantom.mat")

print("Working file keys:", list(working.keys()))
print("My file keys:", list(mine.keys()))
print()

# Compare each key
for key in working.keys():
    if key.startswith('__'):  # Skip MATLAB metadata
        continue

    if key not in mine:
        print(f"Missing key: {key}")
        print(working[key])
        continue

    w_shape = working[key].shape
    m_shape = mine[key].shape
    w_dtype = working[key].dtype
    m_dtype = mine[key].dtype

    print(f"{key}:")
    print(f"  Working: shape={w_shape}, dtype={w_dtype}")
    print(f"  Mine:    shape={m_shape}, dtype={m_dtype}")

    if w_shape != m_shape:
        print(f"  ❌ Shape mismatch!")
    if w_dtype != m_dtype:
        print(f"  ❌ Dtype mismatch!")

    print()
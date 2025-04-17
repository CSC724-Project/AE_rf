#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C

# -----------------------------
# Step 1: Load the CSV data
# -----------------------------
datafile = 'beegfs_test_results4.csv'
df = pd.read_csv(datafile)

# Suppose your CSV columns are named exactly as follows:
#   'file_size_KB'    - file size in KB
#   'chunk_size_KB'   - chunk size in KB
#   'throughput_KBps' - throughput in KB/s

print("Data Head:\n", df.head())

# --------------------------------------------------------------------
# Scale the 'file_size_KB' to MB (if desired). 
# If you already have the correct file size unit (e.g., in bytes), 
# just remove or change this step accordingly.
# --------------------------------------------------------------------
df['file_size_MB'] = df['file_size_KB'] / 1024.0  # 1 MB = 1024 KB

# -----------------------------
# Step 2: Basic Data Visualization
# -----------------------------
# 1) Scatter plot: File Size vs Throughput, colored by Chunk Size
plt.figure(figsize=(8, 6))
# Use file_size_MB instead of file_size_KB
sc = plt.scatter(df['file_size_MB'], df['throughput_KBps'], 
                 c=df['chunk_size_KB'], cmap='viridis', alpha=0.7)

plt.xlabel('File Size (MB)')
plt.ylabel('Throughput (KB/s)')
plt.title('File Size vs Throughput (colored by Chunk Size)')
cb = plt.colorbar(sc)
cb.set_label('Chunk Size (KB)')
plt.grid(True)

# Remove scientific notation and set tick spacing to 2000
plt.ticklabel_format(style='plain', axis='both')
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(5000))
ax.yaxis.set_major_locator(ticker.MultipleLocator(5000))
plt.tight_layout()
plt.show()


# 2) Scatter plot: Chunk Size vs Throughput, colored by File Size (in MB)
plt.figure(figsize=(8, 6))
sc = plt.scatter(df['chunk_size_KB'], df['throughput_KBps'], 
                 c=df['file_size_MB'], cmap='plasma', alpha=0.7)

plt.xlabel('Chunk Size (KB)')
plt.ylabel('Throughput (KB/s)')
plt.title('Chunk Size vs Throughput (colored by File Size)')
cb = plt.colorbar(sc)
cb.set_label('File Size (MB)')
plt.grid(True)

# Remove scientific notation and set tick spacing to 2000
plt.ticklabel_format(style='plain', axis='both')
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(5000))
ax.yaxis.set_major_locator(ticker.MultipleLocator(5000))
plt.tight_layout()
plt.show()


# 3) 3D Scatter Plot: File Size, Chunk Size, Throughput
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

p = ax.scatter(df['file_size_MB'],       # X: now in MB
               df['chunk_size_KB'],      # Y
               df['throughput_KBps'],    # Z
               c=df['throughput_KBps'],  # color scale
               cmap='coolwarm', alpha=0.8)

ax.set_xlabel('File Size (MB)')
ax.set_ylabel('Chunk Size (KB)')
ax.set_zlabel('Throughput (KB/s)')
ax.set_title('3D Scatter Plot of Performance Data')

# Turn off scientific notation and set tick spacing to 2000 on all three axes
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))
ax.zaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))
ax.xaxis.set_major_locator(ticker.MultipleLocator(5000))
ax.yaxis.set_major_locator(ticker.MultipleLocator(5000))
ax.zaxis.set_major_locator(ticker.MultipleLocator(5000))

fig.colorbar(p, ax=ax, label='Throughput (KB/s)')
plt.tight_layout()
plt.show()


# -----------------------------
# Step 3: Gaussian Process Regression
# -----------------------------
X = df[['file_size_MB', 'chunk_size_KB']].values
y = df['throughput_KBps'].values

kernel = (
    C(1.0, (1e-3, 1e3))
    * Matern(length_scale=1.0, nu=2.5)
    + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
)
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
gpr.fit(X, y)

print("Optimized kernel:", gpr.kernel_)

# -----------------------------
# Step 4: Find Optimal Chunk Size for a Chosen File Size
# -----------------------------
# For demonstration, pick the median of file_size_MB
target_filesize = np.median(df['file_size_MB'])
print("Target file size (median):", target_filesize, "MB")

chunk_min = df['chunk_size_KB'].min()
chunk_max = df['chunk_size_KB'].max()
chunk_range = np.linspace(chunk_min, chunk_max, 200)

X_pred = np.column_stack((np.full_like(chunk_range, target_filesize), chunk_range))
y_pred, sigma = gpr.predict(X_pred, return_std=True)

opt_index = np.argmax(y_pred)
optimal_chunk_size = chunk_range[opt_index]
optimal_throughput = y_pred[opt_index]

print(f"Optimal chunk size for file size {target_filesize:.2f} MB: {optimal_chunk_size:.2f} KB")
print(f"Predicted maximum throughput: {optimal_throughput:.2f} KB/s")

# Plot the prediction vs. chunk size
plt.figure(figsize=(8, 6))
plt.plot(chunk_range, y_pred, label='Predicted Throughput')
plt.fill_between(chunk_range, y_pred - sigma, y_pred + sigma, alpha=0.2, label='Std. Dev.')
plt.scatter(optimal_chunk_size, optimal_throughput, color='red', zorder=5, label='Optimal Chunk Size')

plt.xlabel('Chunk Size (KB)')
plt.ylabel('Predicted Throughput (KB/s)')
plt.title(f'Predicted Throughput vs. Chunk Size\n(File Size = {target_filesize:.2f} MB)')
plt.grid(True)
plt.legend()
plt.ticklabel_format(style='plain', axis='both')
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(5000))
ax.yaxis.set_major_locator(ticker.MultipleLocator(5000))
plt.tight_layout()
plt.show()


# -----------------------------
# Optional: 2D Contour Plot over File Size (MB) and Chunk Size (KB)
# -----------------------------
file_min, file_max = df['file_size_MB'].min(), df['file_size_MB'].max()
file_range = np.linspace(file_min, file_max, 100)
chunk_range_grid = np.linspace(chunk_min, chunk_max, 100)
F, C_grid = np.meshgrid(file_range, chunk_range_grid)

grid_points = np.column_stack([F.ravel(), C_grid.ravel()])
throughput_pred, _ = gpr.predict(grid_points, return_std=True)
Throughput_grid = throughput_pred.reshape(F.shape)

plt.figure(figsize=(8, 6))
cp = plt.contourf(F, C_grid, Throughput_grid, cmap='viridis', levels=20)

plt.xlabel('File Size (MB)')
plt.ylabel('Chunk Size (KB)')
plt.title('Contour Plot of Predicted Throughput')
plt.colorbar(cp, label='Predicted Throughput (KB/s)')
plt.scatter(target_filesize, optimal_chunk_size, color='red', s=100, label='Optimal (Target File Size)')
plt.legend()
plt.grid(True)
plt.ticklabel_format(style='plain', axis='both')
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(5000))
ax.yaxis.set_major_locator(ticker.MultipleLocator(5000))
plt.tight_layout()
plt.show()




# #!/usr/bin/env python3

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C

# # -----------------------------
# # Step 1: Load the CSV data
# # -----------------------------
# datafile = 'beegfs_test_results4.csv'
# df = pd.read_csv(datafile)

# # Suppose your CSV columns are named exactly as follows:
# #   'file_size_KB'    - file size in KB
# #   'chunk_size_KB'   - chunk size in KB
# #   'throughput_KBps' - throughput in KB/s

# print("Data Head:\n", df.head())

# # --------------------------------------------------------------------
# # Scale the 'file_size_KB' to MB (if desired). 
# # If you already have the correct file size unit (e.g., in bytes), 
# # just remove or change this step accordingly.
# # --------------------------------------------------------------------
# df['file_size_MB'] = df['file_size_KB'] / 1024.0  # 1 MB = 1024 KB
# # If you want GB instead, use:
# # df['file_size_GB'] = df['file_size_KB'] / (1024.0 * 1024.0)

# # -----------------------------
# # Step 2: Basic Data Visualization
# # -----------------------------
# # 1) Scatter plot: File Size vs Throughput, colored by Chunk Size
# plt.figure(figsize=(8, 6))
# # Use file_size_MB instead of file_size_KB
# sc = plt.scatter(df['file_size_MB'], df['throughput_KBps'], 
#                  c=df['chunk_size_KB'], cmap='viridis', alpha=0.7)

# plt.xlabel('File Size (MB)')
# plt.ylabel('Throughput (KB/s)')
# plt.title('File Size vs Throughput (colored by Chunk Size)')
# cb = plt.colorbar(sc)
# cb.set_label('Chunk Size (KB)')
# plt.grid(True)

# # Optional: remove scientific notation on both axes
# plt.ticklabel_format(style='plain', axis='both')
# plt.tight_layout()
# plt.show()


# # 2) Scatter plot: Chunk Size vs Throughput, colored by File Size (in MB)
# plt.figure(figsize=(8, 6))
# sc = plt.scatter(df['chunk_size_KB'], df['throughput_KBps'], 
#                  c=df['file_size_MB'], cmap='plasma', alpha=0.7)

# plt.xlabel('Chunk Size (KB)')
# plt.ylabel('Throughput (KB/s)')
# plt.title('Chunk Size vs Throughput (colored by File Size)')
# cb = plt.colorbar(sc)
# cb.set_label('File Size (MB)')
# plt.grid(True)

# # Optional: remove scientific notation on both axes
# plt.ticklabel_format(style='plain', axis='both')
# plt.tight_layout()
# plt.show()


# # 3) 3D Scatter Plot: File Size, Chunk Size, Throughput
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# p = ax.scatter(df['file_size_MB'],       # X: now in MB
#                df['chunk_size_KB'],      # Y
#                df['throughput_KBps'],    # Z
#                c=df['throughput_KBps'],  # color scale
#                cmap='coolwarm', alpha=0.8)

# ax.set_xlabel('File Size (MB)')
# ax.set_ylabel('Chunk Size (KB)')
# ax.set_zlabel('Throughput (KB/s)')
# ax.set_title('3D Scatter Plot of Performance Data')

# # Optional: turn off scientific notation on 3D axes
# ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))
# ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))
# ax.zaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))

# fig.colorbar(p, ax=ax, label='Throughput (KB/s)')
# plt.tight_layout()
# plt.show()


# # -----------------------------
# # Step 3: Gaussian Process Regression
# # -----------------------------
# X = df[['file_size_MB', 'chunk_size_KB']].values
# y = df['throughput_KBps'].values

# kernel = (
#     C(1.0, (1e-3, 1e3))
#     * Matern(length_scale=1.0, nu=2.5)
#     + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
# )
# gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
# gpr.fit(X, y)

# print("Optimized kernel:", gpr.kernel_)

# # -----------------------------
# # Step 4: Find Optimal Chunk Size for a Chosen File Size
# # -----------------------------
# # For demonstration, pick the median of file_size_MB
# target_filesize = np.median(df['file_size_MB'])
# print("Target file size (median):", target_filesize, "MB")

# chunk_min = df['chunk_size_KB'].min()
# chunk_max = df['chunk_size_KB'].max()
# chunk_range = np.linspace(chunk_min, chunk_max, 200)

# X_pred = np.column_stack((np.full_like(chunk_range, target_filesize), chunk_range))
# y_pred, sigma = gpr.predict(X_pred, return_std=True)

# opt_index = np.argmax(y_pred)
# optimal_chunk_size = chunk_range[opt_index]
# optimal_throughput = y_pred[opt_index]

# print(f"Optimal chunk size for file size {target_filesize:.2f} MB: {optimal_chunk_size:.2f} KB")
# print(f"Predicted maximum throughput: {optimal_throughput:.2f} KB/s")

# # Plot the prediction vs. chunk size
# plt.figure(figsize=(8, 6))
# plt.plot(chunk_range, y_pred, label='Predicted Throughput')
# plt.fill_between(chunk_range, y_pred - sigma, y_pred + sigma, alpha=0.2, label='Std. Dev.')
# plt.scatter(optimal_chunk_size, optimal_throughput, color='red', zorder=5, label='Optimal Chunk Size')

# plt.xlabel('Chunk Size (KB)')
# plt.ylabel('Predicted Throughput (KB/s)')
# plt.title(f'Predicted Throughput vs. Chunk Size\n(File Size = {target_filesize:.2f} MB)')
# plt.grid(True)
# plt.legend()
# plt.ticklabel_format(style='plain', axis='both')
# plt.tight_layout()
# plt.show()


# # -----------------------------
# # Optional: 2D Contour Plot over File Size (MB) and Chunk Size (KB)
# # -----------------------------
# file_min, file_max = df['file_size_MB'].min(), df['file_size_MB'].max()
# file_range = np.linspace(file_min, file_max, 100)
# chunk_range_grid = np.linspace(chunk_min, chunk_max, 100)
# F, C_grid = np.meshgrid(file_range, chunk_range_grid)

# grid_points = np.column_stack([F.ravel(), C_grid.ravel()])
# throughput_pred, _ = gpr.predict(grid_points, return_std=True)
# Throughput_grid = throughput_pred.reshape(F.shape)

# plt.figure(figsize=(8, 6))
# cp = plt.contourf(F, C_grid, Throughput_grid, cmap='viridis', levels=20)

# plt.xlabel('File Size (MB)')
# plt.ylabel('Chunk Size (KB)')
# plt.title('Contour Plot of Predicted Throughput')
# plt.colorbar(cp, label='Predicted Throughput (KB/s)')
# plt.scatter(target_filesize, optimal_chunk_size, color='red', s=100, label='Optimal (Target File Size)')
# plt.legend()
# plt.grid(True)
# plt.ticklabel_format(style='plain', axis='both')
# plt.tight_layout()
# plt.show()
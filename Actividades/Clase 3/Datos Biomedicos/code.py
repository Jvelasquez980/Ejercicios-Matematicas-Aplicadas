from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


csv_path = Path(__file__).parent / "Data" / "datos_Biomedicos.csv"
data = pd.read_csv(csv_path)

# Leer los datos como matriz NumPy (filas x columnas)
data_matrix = data.to_numpy(dtype=float)

new_base = np.array([
    [0.7, 0.7, 0.0],
    [0.5, -0.5, 0.7],
    [-0.5, 0.5, 0.7],
])

new_base_inv = np.linalg.inv(new_base)

data_transposed = data_matrix.T
data_in_new_base = new_base_inv @ data_transposed
data_in_new_base = data_in_new_base.T

print("Matriz original:", data_matrix.shape)
print(data_matrix[:5])
print("\nMatriz en nueva base:", data_in_new_base.shape)
print(data_in_new_base[:5])

# DataFrame con columnas en la nueva base
new_columns = ["Base_nueva_1", "Base_nueva_2", "Base_nueva_3"]
data_new_base_df = pd.DataFrame(data_in_new_base, columns=new_columns)

# Histogramas comparativos: datos originales vs datos en nueva base
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

all_series = [data[col].to_numpy() for col in data.columns] + [
    data_new_base_df[col].to_numpy() for col in new_columns
]
all_values = np.concatenate(all_series)
x_min, x_max = all_values.min(), all_values.max()
bins = np.linspace(x_min, x_max, 31)

max_count = max(np.histogram(series, bins=bins)[0].max() for series in all_series)

for i, col in enumerate(data.columns):
    axes[0, i].hist(data[col], bins=bins, edgecolor="black", alpha=0.75)
    axes[0, i].set_title(f"Original: {col}")
    axes[0, i].set_xlabel("Valor")
    axes[0, i].set_ylabel("Frecuencia")
    axes[0, i].set_xlim(x_min, x_max)
    axes[0, i].set_ylim(0, max_count * 1.05)

for i, col in enumerate(new_columns):
    axes[1, i].hist(data_new_base_df[col], bins=bins, edgecolor="black", alpha=0.75)
    axes[1, i].set_title(f"Nueva base: {col}")
    axes[1, i].set_xlabel("Valor")
    axes[1, i].set_ylabel("Frecuencia")
    axes[1, i].set_xlim(x_min, x_max)
    axes[1, i].set_ylim(0, max_count * 1.05)

fig.suptitle("Histogramas: datos originales vs datos en nueva base", fontsize=14)
plt.tight_layout()
plt.show()



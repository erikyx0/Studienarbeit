import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # nötig für 3D-Plots
import re
import numpy as np



def plot_runs_3d_simple(dfs, cmap='plasma', s=20, run_step=1, point_step=1):
    """
    3D-Plot aller Runs:
      x = Distance, y = Run-Nummer, z = Temperature, Farbe = z-Wert (global skaliert).
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 🔸 1. Globale Farbskala bestimmen
    all_z = []
    for df in dfs[::run_step]:
        zcol = next(c for c in df.columns if re.search('temperature', c, re.IGNORECASE))
        all_z.append(df[zcol].iloc[::point_step].to_numpy())
    all_z = np.concatenate(all_z)
    vmin, vmax = np.nanmin(all_z), np.nanmax(all_z)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # 🔸 2. Plotten
    for i, df in enumerate(dfs[::run_step], start=1):
        xcol = next(c for c in df.columns if re.search('distance', c, re.IGNORECASE))
        zcol = next(c for c in df.columns if re.search('temperature', c, re.IGNORECASE))
        df_reduced = df.iloc[::point_step, :]

        x = df_reduced[xcol].to_numpy()
        z = df_reduced[zcol].to_numpy()
        y = np.full_like(x, i * run_step)

        # nach z sortieren für saubere Sichttiefe
        order = np.argsort(z)

        # global normierte Farbskala verwenden
        ax.scatter(x[order], y[order], z[order],
                   c=z[order], cmap=cmap, norm=norm,
                   s=s, alpha=0.7, edgecolors='none')
    # 🔸 3. Achsen & Layout
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Run')
    ax.set_zlabel('Temperature (K)')
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=22, azim=135)
    ax.set_proj_type('ortho')

    # 🔸 4. Gemeinsame Farbskala
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label('Temperature (K)')

    plt.tight_layout()
    plt.show()


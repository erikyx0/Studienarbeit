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


def plot_runs_3d_bubble(dfs, cmap='viridis', run_step=1, point_step=1):
    """
    3D-Bubbleplot:
      x = Distance, y = Run, z = Temperatur
      Farbe = Temperatur, Blasengröße = |dT/dz| (lokaler Gradient)
    """
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')

    # globale Temperaturgrenzen
    all_T = []
    for df in dfs[::run_step]:
        tcol = next(c for c in df.columns if re.search('temperature', c, re.IGNORECASE))
        all_T.append(df[tcol].iloc[::point_step].to_numpy())
    all_T = np.concatenate(all_T)
    vmin, vmax = np.nanmin(all_T), np.nanmax(all_T)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for i, df in enumerate(dfs[::run_step], start=1):
        xcol = next(c for c in df.columns if re.search('distance', c, re.IGNORECASE))
        tcol = next(c for c in df.columns if re.search('temperature', c, re.IGNORECASE))
        df = df.iloc[::point_step, :]

        x = df[xcol].to_numpy()
        z = df[tcol].to_numpy()
        y = np.full_like(x, i*run_step)

        # lokale Temperaturgradienten → Größe
        gradT = np.abs(np.gradient(z))
        # robuster lineare Skalierung
        p5, p95 = np.nanpercentile(gradT, [5, 95])
        s = 30 + 470 * np.clip((gradT - p5) / (p95 - p5 + 1e-9), 0, 1)

        ax.scatter(x, y, z, c=z, s=s, cmap=cmap, norm=norm,
                   alpha=0.85, edgecolors='none')

    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Run')
    ax.set_zlabel('Temperature (K)')
    ax.view_init(elev=22, azim=135)
    ax.set_proj_type('ortho')
    ax.set_box_aspect([1, 1, 1])

    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
    cb.set_label('Temperature (K)')
    plt.tight_layout()
    plt.show()

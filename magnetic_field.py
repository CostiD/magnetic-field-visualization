"""
Calculul și vizualizarea câmpului magnetic al unei spire circulare
Versiune optimizată pentru Google Colab - vizualizări statice HD

INSTALARE ÎN COLAB:
!pip install pyvista numpy scipy matplotlib

PENTRU VIZUALIZARE STATICĂ:
import pyvista as pv
pv.set_jupyter_backend('static')
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate
import warnings
warnings.filterwarnings('ignore')

# Pentru Colab: configurare PyVista
try:
    import pyvista as pv
    pv.set_jupyter_backend('static')
    PYVISTA_AVAILABLE = True
    print("✓ PyVista disponibil - mod static")
except:
    PYVISTA_AVAILABLE = False
    print("⚠ PyVista indisponibil - doar Matplotlib")

# ============================================================================
# CONSTANTE FIZICE
# ============================================================================
MU_0 = 4 * np.pi * 1e-7  # Permeabilitatea vidului [H/m]

# ============================================================================
# PARAMETRI SPIRĂ
# ============================================================================
I = 10.0        # Curent [A]
R = 0.1         # Rază spiră [m]
N_LOOP = 200    # Număr puncte pe spiră

# ============================================================================
# PARAMETRI GRILĂ 2D
# ============================================================================
GRID_SIZE = 50
X_RANGE = (-0.3, 0.3)
Z_RANGE = (-0.3, 0.3)
Y_PLANE = 0.0

# ============================================================================
# FUNCȚII CALCUL
# ============================================================================

def generate_loop_points(radius, n_points):
    """Generează punctele pe spiră"""
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros_like(theta)
    return np.column_stack([x, y, z])

def biot_savart_loop(field_point, loop_points, current):
    """
    Legea Biot-Savart: dB = (μ₀/4π) * I * (dl × r) / |r|³
    """
    B = np.zeros(3)
    n_segments = len(loop_points)
    
    for i in range(n_segments):
        p1 = loop_points[i]
        p2 = loop_points[(i + 1) % n_segments]
        dl = p2 - p1
        p_mid = (p1 + p2) / 2
        r = field_point - p_mid
        r_mag = np.linalg.norm(r)
        
        if r_mag < R/20:
            continue
        
        dl_cross_r = np.cross(dl, r)
        dB = (MU_0 / (4 * np.pi)) * current * dl_cross_r / (r_mag**3)
        B += dB
    
    return B

def calculate_field_grid_2d(loop_points, current, x_range, z_range, grid_size, y_plane=0):
    """Calculează câmpul pe grilă 2D"""
    x = np.linspace(x_range[0], x_range[1], grid_size)
    z = np.linspace(z_range[0], z_range[1], grid_size)
    X, Z = np.meshgrid(x, z)
    
    B_field = np.zeros((grid_size, grid_size, 3))
    B_mag = np.zeros((grid_size, grid_size))
    
    print(f"Calculez câmpul magnetic pe grilă {grid_size}×{grid_size}...")
    
    for i in range(grid_size):
        if i % 10 == 0:
            print(f"  Progres: {i}/{grid_size}")
        for j in range(grid_size):
            point = np.array([X[i, j], y_plane, Z[i, j]])
            B = biot_savart_loop(point, loop_points, current)
            B_field[i, j] = B
            B_mag[i, j] = np.linalg.norm(B)
    
    print("✓ Calcul finalizat!")
    return X, Z, B_field, B_mag

def B_axial_analytical(z, R, I):
    """Formula analitică pe axa spirei"""
    return (MU_0 * I * R**2) / (2 * (R**2 + z**2)**(3/2))

# ============================================================================
# VIZUALIZĂRI MATPLOTLIB HD
# ============================================================================

def plot_comprehensive_2d(X, Z, B_field, B_mag, loop_points):
    """Vizualizare comprehensivă 2D - 4 panouri"""
    fig = plt.figure(figsize=(18, 14))
    plt.style.use('dark_background')
    
    theta = np.linspace(0, 2*np.pi, 100)
    
    # 1. HEATMAP MAGNITUDINE
    ax1 = plt.subplot(2, 2, 1)
    im1 = ax1.contourf(X, Z, B_mag, levels=100, cmap='inferno')
    contours = ax1.contour(X, Z, B_mag, levels=15, colors='white', 
                           linewidths=0.8, alpha=0.4)
    ax1.clabel(contours, inline=True, fontsize=8, fmt='%.1e')
    ax1.plot(R*np.cos(theta), np.zeros_like(theta), 
             color='gold', linewidth=4, label=f'Spiră (I={I:.1f}A)',
             marker='o', markersize=3, markevery=10)
    ax1.set_xlabel('x [m]', fontsize=12, fontweight='bold')
    ax1.set_ylabel('z [m]', fontsize=12, fontweight='bold')
    ax1.set_title('Magnitudine |B| [T]', fontsize=14, fontweight='bold', pad=15)
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.2, linestyle='--')
    cbar1 = plt.colorbar(im1, ax=ax1, pad=0.02)
    cbar1.set_label('|B| [T]', fontsize=11, fontweight='bold')
    
    # 2. STREAMPLOT
    ax2 = plt.subplot(2, 2, 2)
    Bx = B_field[:, :, 0]
    Bz = B_field[:, :, 2]
    B_mag_clip = np.clip(B_mag, 1e-12, None)
    strm = ax2.streamplot(X, Z, Bx, Bz, color=np.log10(B_mag_clip), 
                          cmap='plasma', density=2.0, linewidth=2,
                          arrowsize=2.0, arrowstyle='->', minlength=0.1)
    ax2.plot(R*np.cos(theta), np.zeros_like(theta),
             color='gold', linewidth=4, marker='o', markersize=3, markevery=10)
    ax2.set_xlabel('x [m]', fontsize=12, fontweight='bold')
    ax2.set_ylabel('z [m]', fontsize=12, fontweight='bold')
    ax2.set_title('Linii de câmp (streamlines)', fontsize=14, fontweight='bold', pad=15)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.2, linestyle='--')
    cbar2 = plt.colorbar(strm.lines, ax=ax2, pad=0.02)
    cbar2.set_label('log₁₀(|B|)', fontsize=11, fontweight='bold')
    
    # 3. QUIVER PLOT
    ax3 = plt.subplot(2, 2, 3)
    step = max(1, GRID_SIZE // 20)
    X_sub = X[::step, ::step]
    Z_sub = Z[::step, ::step]
    Bx_sub = Bx[::step, ::step]
    Bz_sub = Bz[::step, ::step]
    B_mag_sub = B_mag[::step, ::step]
    
    im3 = ax3.contourf(X, Z, B_mag, levels=50, cmap='viridis', alpha=0.6)
    quiv = ax3.quiver(X_sub, Z_sub, Bx_sub, Bz_sub, B_mag_sub,
                      cmap='hot', scale=np.max(B_mag)*40, width=0.004,
                      headwidth=4, headlength=5, alpha=0.9)
    ax3.plot(R*np.cos(theta), np.zeros_like(theta),
             color='cyan', linewidth=4, marker='o', markersize=3, markevery=10)
    ax3.set_xlabel('x [m]', fontsize=12, fontweight='bold')
    ax3.set_ylabel('z [m]', fontsize=12, fontweight='bold')
    ax3.set_title('Vectori câmp (quiver)', fontsize=14, fontweight='bold', pad=15)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.2, linestyle='--')
    cbar3 = plt.colorbar(quiv, ax=ax3, pad=0.02)
    cbar3.set_label('|B| [T]', fontsize=11, fontweight='bold')
    
    # 4. PROFILE PE AXE
    ax4 = plt.subplot(2, 2, 4)
    idx_z0 = GRID_SIZE // 2
    ax4.plot(X[idx_z0, :], B_mag[idx_z0, :], 'o-', 
             linewidth=2.5, markersize=5, label='Profil x (z=0)', color='cyan')
    idx_x0 = GRID_SIZE // 2
    z_axis = Z[:, idx_x0]
    B_z_axis = B_mag[:, idx_x0]
    ax4.plot(z_axis, B_z_axis, 's-', 
             linewidth=2.5, markersize=5, label='Profil z (x=0)', color='magenta')
    z_analytical = np.linspace(Z_RANGE[0], Z_RANGE[1], 100)
    B_analytical = B_axial_analytical(z_analytical, R, I)
    ax4.plot(z_analytical, B_analytical, '--', linewidth=2, 
             label='Formula analitică', color='lime', alpha=0.8)
    ax4.set_xlabel('Poziție [m]', fontsize=12, fontweight='bold')
    ax4.set_ylabel('|B| [T]', fontsize=12, fontweight='bold')
    ax4.set_title('Profile pe axe', fontsize=14, fontweight='bold', pad=15)
    ax4.legend(fontsize=10, loc='upper right')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_yscale('log')
    
    info_text = f'R={R:.3f}m\nI={I:.1f}A\nμ₀={MU_0:.2e}H/m\nGrid:{GRID_SIZE}×{GRID_SIZE}'
    ax4.text(0.98, 0.02, info_text, transform=ax4.transAxes,
             fontsize=9, va='bottom', ha='right',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, 
                      edgecolor='cyan', linewidth=2))
    
    plt.tight_layout()
    plt.savefig('magnetic_field_2d.png', dpi=300, bbox_inches='tight', facecolor='black')
    print("✓ Salvat: magnetic_field_2d.png")
    plt.show()

def plot_3d_surface(X, Z, B_mag, loop_points):
    """Vizualizare 3D cu 4 perspective"""
    fig = plt.figure(figsize=(16, 12))
    plt.style.use('dark_background')
    theta = np.linspace(0, 2*np.pi, 100)
    
    views = [
        (1, 25, 45, 'Vedere 3D standard', 'plasma'),
        (2, 90, 0, 'Vedere de sus', 'inferno'),
        (3, 0, 0, 'Vedere laterală', 'viridis'),
        (4, 30, 135, 'Wireframe + contururi', 'cool')
    ]
    
    for idx, elev, azim, title, cmap in views:
        ax = fig.add_subplot(2, 2, idx, projection='3d')
        
        if idx == 4:
            ax.plot_wireframe(X, Z, B_mag, color='cyan', linewidth=0.5, alpha=0.6)
            ax.contour(X, Z, B_mag, zdir='z', offset=B_mag.min(),
                      cmap='hot', levels=15, linewidths=2, alpha=0.8)
        else:
            surf = ax.plot_surface(X, Z, B_mag, cmap=cmap, 
                                  edgecolor='none', alpha=0.9,
                                  antialiased=True, shade=True)
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        
        ax.plot(R*np.cos(theta), np.zeros_like(theta), 
                np.full_like(theta, B_mag.min()),
                color='gold', linewidth=4)
        
        ax.set_xlabel('x [m]', fontsize=11, fontweight='bold', labelpad=10)
        ax.set_ylabel('z [m]', fontsize=11, fontweight='bold', labelpad=10)
        ax.set_zlabel('|B| [T]', fontsize=11, fontweight='bold', labelpad=10)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        ax.view_init(elev=elev, azim=azim)
    
    plt.tight_layout()
    plt.savefig('magnetic_field_3d.png', dpi=300, bbox_inches='tight', facecolor='black')
    print("✓ Salvat: magnetic_field_3d.png")
    plt.show()

def compare_with_analytical():
    """Comparație cu formula analitică"""
    z_axis = np.linspace(-0.3, 0.3, 100)
    B_numerical = []
    B_analytical = []
    
    loop_pts = generate_loop_points(R, N_LOOP)
    
    print("\nComparaț cu formula analitică...")
    for z in z_axis:
        point = np.array([0, 0, z])
        B_num = biot_savart_loop(point, loop_pts, I)
        B_numerical.append(B_num[2])
        B_analytical.append(B_axial_analytical(z, R, I))
    
    B_numerical = np.array(B_numerical)
    B_analytical = np.array(B_analytical)
    error = np.abs((B_numerical - B_analytical) / (B_analytical + 1e-12)) * 100
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))
    plt.style.use('dark_background')
    
    # Comparație
    ax1.plot(z_axis, B_numerical*1e6, 'o-', linewidth=2, markersize=4, 
             label='Numeric', color='cyan', markevery=5)
    ax1.plot(z_axis, B_analytical*1e6, '--', label='Analitic', 
             linewidth=3, color='lime', alpha=0.8)
    ax1.set_xlabel('z [m]', fontsize=12, fontweight='bold')
    ax1.set_ylabel('B_z [μT]', fontsize=12, fontweight='bold')
    ax1.set_title('Comparație Numeric vs Analitic', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Eroare relativă
    ax2.plot(z_axis, error, '-', linewidth=2.5, color='red')
    ax2.fill_between(z_axis, 0, error, alpha=0.3, color='red')
    ax2.set_xlabel('z [m]', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Eroare relativă [%]', fontsize=12, fontweight='bold')
    ax2.set_title('Eroare numerică', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    stats = f'Medie: {np.mean(error):.4f}%\nMax: {np.max(error):.4f}%\nStd: {np.std(error):.4f}%'
    ax2.text(0.02, 0.98, stats, transform=ax2.transAxes, fontsize=10, va='top',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.8,
                      edgecolor='red', linewidth=2))
    
    # Diferență absolută
    diff = np.abs(B_numerical - B_analytical)*1e9
    ax3.semilogy(z_axis, diff, '-', linewidth=2.5, color='magenta')
    ax3.fill_between(z_axis, 1e-5, diff, alpha=0.3, color='magenta')
    ax3.set_xlabel('z [m]', fontsize=12, fontweight='bold')
    ax3.set_ylabel('|B_num - B_anal| [nT]', fontsize=12, fontweight='bold')
    ax3.set_title('Diferență absolută (log)', fontsize=14, fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('analytical_comparison.png', dpi=300, bbox_inches='tight', facecolor='black')
    print(f"✓ Eroare medie: {np.mean(error):.4f}%")
    print(f"✓ Eroare max: {np.max(error):.4f}%")
    print("✓ Salvat: analytical_comparison.png")
    plt.show()

def plot_pyvista_static(X, Z, B_field, B_mag, loop_points):
    """PyVista static render HD"""
    if not PYVISTA_AVAILABLE:
        print("⚠ PyVista indisponibil")
        return
    
    print("Generez PyVista static...")
    plotter = pv.Plotter(window_size=[1920, 1080])
    plotter.set_background('black')
    
    # Spiră
    loop_poly = pv.PolyData(loop_points)
    loop_tube = loop_poly.tube(radius=R/30)
    plotter.add_mesh(loop_tube, color='gold', metallic=1.0)
    
    # Grid
    grid_points = np.column_stack([X.ravel(), np.full(X.size, Y_PLANE), Z.ravel()])
    grid_poly = pv.PolyData(grid_points)
    grid_poly['B_magnitude'] = B_mag.ravel()
    surf = grid_poly.delaunay_2d()
    plotter.add_mesh(surf, scalars='B_magnitude', cmap='plasma', opacity=0.8,
                     show_scalar_bar=True,
                     scalar_bar_args={'title': '|B| [T]', 'color': 'white'})
    
    # Vectori
    step = max(1, GRID_SIZE // 15)
    arrow_points = np.column_stack([X[::step, ::step].ravel(), 
                                    np.full(X[::step, ::step].size, Y_PLANE),
                                    Z[::step, ::step].ravel()])
    arrow_vectors = B_field[::step, ::step].reshape(-1, 3)
    arrow_mag = np.linalg.norm(arrow_vectors, axis=1, keepdims=True)
    arrow_mag[arrow_mag < 1e-12] = 1
    arrow_vectors_norm = arrow_vectors / arrow_mag
    
    arrow_poly = pv.PolyData(arrow_points)
    arrow_poly['vectors'] = arrow_vectors_norm
    arrows = arrow_poly.glyph(orient='vectors', scale=False, factor=0.02)
    plotter.add_mesh(arrows, color='cyan', opacity=0.9)
    
    plotter.camera_position = 'xz'
    plotter.camera.zoom(1.3)
    plotter.show()
    print("✓ PyVista vizualizare afișată")

def export_to_csv(X, Z, B_field, B_mag, filename='magnetic_field.csv'):
    """Export CSV"""
    with open(filename, 'w') as f:
        f.write('x,z,Bx,By,Bz,B_mag\n')
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                f.write(f'{X[i,j]},{Z[i,j]},{B_field[i,j,0]},{B_field[i,j,1]},'
                       f'{B_field[i,j,2]},{B_mag[i,j]}\n')
    print(f"✓ Salvat: {filename}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("CÂMP MAGNETIC - SPIRĂ CIRCULARĂ (VERSIUNE COLAB)")
    print("="*70)
    print(f"\nParametri: I={I}A | R={R}m | Grid={GRID_SIZE}×{GRID_SIZE}")
    
    L = MU_0 * R * (np.log(8*R/(R/100)) - 2)
    print(f"Inductanță estimată: {L*1e6:.2f} μH")
    
    loop_points = generate_loop_points(R, N_LOOP)
    X, Z, B_field, B_mag = calculate_field_grid_2d(
        loop_points, I, X_RANGE, Z_RANGE, GRID_SIZE, Y_PLANE)
    
    print(f"\nStatistici: B_max={np.max(B_mag):.4e}T | B_min={np.min(B_mag[B_mag>0]):.4e}T")
    
    print("\n" + "-"*70)
    compare_with_analytical()
    
    print("\n" + "-"*70)
    plot_comprehensive_2d(X, Z, B_field, B_mag, loop_points)
    
    print("\n" + "-"*70)
    plot_3d_surface(X, Z, B_mag, loop_points)
    
    if PYVISTA_AVAILABLE:
        print("\n" + "-"*70)
        plot_pyvista_static(X, Z, B_field, B_mag, loop_points)
    
    print("\n" + "-"*70)
    export_to_csv(X, Z, B_field, B_mag)
    
    print("\n" + "="*70)
    print("✓ FINALIZAT! Toate imaginile salvate.")
    print("="*70)

if __name__ == "__main__":
    main()

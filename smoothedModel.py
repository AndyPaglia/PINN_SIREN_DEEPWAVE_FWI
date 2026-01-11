import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import argparse
import os

def smooth_velocity_model(vp, sigma=10):
    return gaussian_filter(vp, sigma=sigma)

def main(args):
    if not os.path.exists(args.input_path):
        print(f"❌ Errore: Il file {args.input_path} non esiste.")
        return

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"🚀 Caricamento modello originale: {args.input_path}")
    data = np.load(args.input_path)
    vp = data["vp"].copy()
    spacing = data["spacing"]

    # --- CORREZIONE UNITA' DI MISURA ---
    # Se i valori sono piccoli (es. max < 10), convertiamo da km/s a m/s
    if vp.max() < 10:
        print("ℹ️ Rilevate velocità in km/s. Conversione in m/s...")
        vp *= 1000.0

    # Applicazione smoothing
    print(f"🔍 Applicando smoothing (Gaussian Filter, sigma={args.sigma})...")
    vp_smooth = smooth_velocity_model(vp, sigma=args.sigma)

    # Visualizzazione
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    extent = [0, vp.shape[0]*spacing[0], vp.shape[1]*spacing[1], 0]
    
    # Usiamo lo stesso vmin/vmax per entrambi per un confronto equo
    vmin, vmax = vp.min(), vp.max()

    # Plot Originale
    im0 = axes[0].imshow(vp.T, cmap='jet', extent=extent, aspect='auto', vmin=vmin, vmax=vmax)
    axes[0].set_title("Modello Originale (Marmousi)")
    axes[0].set_ylabel("Profondità (m)")
    axes[0].set_xlabel("Distanza (m)")
    plt.colorbar(im0, ax=axes[0], label="Velocità (m/s)", fraction=0.046, pad=0.04)
    
    # Plot Smoothato
    im1 = axes[1].imshow(vp_smooth.T, cmap='jet', extent=extent, aspect='auto', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Modello Iniziale per FWI (Sigma={args.sigma})")
    axes[1].set_xlabel("Distanza (m)")
    plt.colorbar(im1, ax=axes[1], label="Velocità (m/s)", fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    fig_path = args.output_path.replace('.npz', '.png')
    plt.savefig(fig_path, dpi=150)
    print(f"✅ Confronto salvato in: {fig_path}")

    # Salvataggio file NPZ (sempre in m/s per Deepwave)
    np.savez(args.output_path, vp=vp_smooth, spacing=spacing)
    print(f"💾 Modello smoothato salvato (m/s): {args.output_path}")
    
    if args.show:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--sigma', type=float, default=10)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    main(args)
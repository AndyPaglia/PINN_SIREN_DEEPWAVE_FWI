import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import deepwave
from deepwave import scalar

def forward_modeling_deepwave(vp, spacing, src_coordinates, rec_coordinates, 
                              f0, tn, dt, device, accuracy=8, pml_width=20, verbose=True):
    """
    Forward modeling ottimizzato: genera dati per FWI e visualizzazione.
    """
    # 1. GESTIONE INPUT (Tensore per FWI o Numpy per Forward)
    if torch.is_tensor(vp):
        v = vp
        is_fwi = True
    else:
        v = torch.from_numpy(vp).float().to(device)
        is_fwi = False

    should_print = verbose and (not is_fwi)

    if should_print:
        print(f"\n{'='*60}\nDEBUG - INFORMAZIONI MODELLO\n{'='*60}")
        print(f"  - Input type: {'Torch Tensor (FWI mode)' if is_fwi else 'Numpy Array'}")
        print(f"  - Min/Max velocità: {v.min():.2f} / {v.max():.2f} m/s")

    # 2. PARAMETRI TEMPORALI
    dt_sec = dt / 1000.0
    tn_sec = tn / 1000.0
    nt = int(tn_sec / dt_sec) + 1
    peak_time = 1.5 / f0
    dx = float(spacing[0])
    dz = float(spacing[1]) if len(spacing) > 1 else dx
    
    # 3. COORDINATE (Semplificazione posizionamento e indici)
    # Calcolo indici PURI (Deepwave gestisce il PML internamente)
    src_x_idx = torch.tensor(src_coordinates[:, 0] / dx, dtype=torch.long)
    src_z_idx = torch.tensor(src_coordinates[:, 1] / dz, dtype=torch.long)
    
    max_x_idx, max_z_idx = v.shape[0] - 1, v.shape[1] - 1
    
    # Validazione automatica per evitare crash "Out of model"
    valid_mask = (src_x_idx >= 0) & (src_x_idx <= max_x_idx) & \
                 (src_z_idx >= 0) & (src_z_idx <= max_z_idx)
    
    src_x_idx = src_x_idx[valid_mask]
    src_z_idx = src_z_idx[valid_mask]
    n_shots_valid = len(src_x_idx)

    # Creazione tensori locazioni (Batch x N_sorgenti_per_shot x Dim)
    source_locations = torch.zeros(n_shots_valid, 1, 2, dtype=torch.long, device=device)
    source_locations[:, 0, 0] = src_x_idx
    source_locations[:, 0, 1] = src_z_idx

    rec_x_idx = torch.tensor(rec_coordinates[:, 0] / dx, dtype=torch.long)
    rec_z_idx = torch.tensor(rec_coordinates[:, 1] / dz, dtype=torch.long)
    
    # Tutti gli shot condividono la stessa linea di ricevitori
    receiver_locations = torch.zeros(n_shots_valid, len(rec_x_idx), 2, dtype=torch.long, device=device)
    for shot_idx in range(n_shots_valid):
        receiver_locations[shot_idx, :, 0] = rec_x_idx
        receiver_locations[shot_idx, :, 1] = rec_z_idx

    # 4. WAVELET
    source_amplitudes = (
        deepwave.wavelets.ricker(f0, nt, dt_sec, peak_time)
        .repeat(n_shots_valid, 1, 1)
        .to(device)
    )

    # 5. ESECUZIONE
    if should_print:
        print(f"  - Esecuzione forward modeling per {n_shots_valid} shots...")

    out = scalar(
        v, dx, dt_sec,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        accuracy=accuracy,
        pml_width=pml_width,
        pml_freq=f0
    )
    
    receiver_amplitudes = out[-1]
    
    if should_print:
        print(f"  - Forward completato. Shape output: {receiver_amplitudes.shape}")

    return receiver_amplitudes, source_amplitudes

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}\nCONFIGURAZIONE SISTEMA\n{'='*60}")
    print(f"  - Device: {device}")
    
    # Caricamento Modello
    npzfile = np.load(args.vp_model_path)
    vp = npzfile["vp"]
    spacing = npzfile["spacing"]
    if vp.max() < 10: vp = vp * 1000.0
    
    # Calcolo dimensioni fisiche
    domain_size_x = (vp.shape[0] - 1) * spacing[0]
    
    # SEMPLIFICAZIONE: Generazione coordinate lineari pulite
    n_src = int(domain_size_x // args.src_spacing) + 1
    src_coordinates = np.zeros((n_src, 2))
    src_coordinates[:, 0] = np.linspace(0, domain_size_x, n_src)
    src_coordinates[:, 1] = args.src_depth
    
    n_rec = int(domain_size_x // args.rec_spacing) + 1
    rec_coordinates = np.zeros((n_rec, 2))
    rec_coordinates[:, 0] = np.linspace(0, domain_size_x, n_rec)
    rec_coordinates[:, 1] = args.rec_depth

    print(f"\n{'='*60}\nGEOMETRIA ACQUISIZIONE\n{'='*60}")
    print(f"  - Sorgenti attese: {n_src} (Spaziatura: {args.src_spacing}m)")
    print(f"  - Ricevitori attesi: {n_rec} (Spaziatura: {args.rec_spacing}m)")

    # Esecuzione
    receiver_amplitudes, _ = forward_modeling_deepwave(
        vp=vp, spacing=spacing, src_coordinates=src_coordinates, 
        rec_coordinates=rec_coordinates, f0=args.f0, tn=args.tn, dt=args.dt, 
        device=device, accuracy=args.accuracy, pml_width=args.pml_width
    )
    
    # --- SALVATAGGIO DATI .NPZ PER FWI ---
    receiver_data = receiver_amplitudes.cpu().numpy()
    out_dir = os.path.join("data", "shots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(args.vp_model_path))[0] + "_deepwave.npz")
    
    np.savez(
        out_path,
        d_obs_list=receiver_data,
        src_coordinates=src_coordinates,
        rec_coordinates=rec_coordinates,
        spacing=spacing,
        f0=args.f0,
        dt=args.dt,
        tn=args.tn,
        domain=vp.shape,
        nbl=args.pml_width
    )
    print(f"\n{'='*60}\nDATI SALVATI: {out_path}\n{'='*60}")

    # --- VISUALIZZAZIONE GRAFICA OTTIMIZZATA ---
    os.makedirs("data/plots", exist_ok=True)
    shot_idx = len(receiver_data) // 2
    time_axis = np.linspace(0, args.tn/1000, receiver_data.shape[2]) # Tempo in secondi
    dist_axis = np.linspace(0, domain_size_x, receiver_data.shape[1]) # Distanza in metri

    plt.figure(figsize=(14, 10)) # Figura più alta per ospitare i due plot
    
    # Sotto-grafico 1: Modello di velocità
    ax1 = plt.subplot(2, 1, 1)
    # aspect='equal' mantiene le proporzioni reali 1m:1m
    im1 = ax1.imshow(vp.T, cmap='jet', extent=[0, domain_size_x, vp.shape[1]*spacing[1], 0], aspect='auto')
    ax1.scatter(src_coordinates[:, 0], src_coordinates[:, 1], c='white', s=5, edgecolors='black', label='Sorgenti')
    ax1.set_title("Modello di Velocità Marmousi (m/s)")
    ax1.set_ylabel("Profondità (m)")
    ax1.set_xlabel("Distanza (m)")
    # Fissiamo la colorbar sui valori reali
    plt.colorbar(im1, ax=ax1, label="Velocità (m/s)", fraction=0.046, pad=0.04)
    ax1.legend(loc='lower right')

    # Sotto-grafico 2: Shot Gather (il sismogramma)
    ax2 = plt.subplot(2, 1, 2)
    v_lim = np.percentile(np.abs(receiver_data[shot_idx]), 98)
    # Usiamo extent per mettere metri e secondi sugli assi
    im2 = ax2.imshow(receiver_data[shot_idx].T, cmap='gray', aspect='auto', 
                     vmin=-v_lim, vmax=v_lim,
                     extent=[0, domain_size_x, time_axis[-1], 0])
    ax2.set_title(f"Shot Gather Centrale (Sorgente a X={src_coordinates[shot_idx, 0]}m)")
    ax2.set_xlabel("Posizione Ricevitori (m)")
    ax2.set_ylabel("Tempo (s)")
    
    plt.tight_layout()
    plot_path = f"data/plots/shot_gather_fixed_{os.path.basename(args.vp_model_path)}.png"
    plt.savefig(plot_path, dpi=200)
    print(f"  - Grafico corretto salvato in: {plot_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deepwave Forward Modeling per FWI')
    parser.add_argument('--vp_model_path', type=str, default="./data/v_models/marmousi_sp25.npz")
    parser.add_argument('--src_spacing', type=float, default=100)
    parser.add_argument('--rec_spacing', type=float, default=25)
    parser.add_argument('--src_depth', type=float, default=20)
    parser.add_argument('--rec_depth', type=float, default=20)
    parser.add_argument('--f0', type=float, default=10)
    parser.add_argument('--tn', type=float, default=3000)
    parser.add_argument('--dt', type=float, default=2)
    parser.add_argument('--pml_width', type=int, default=20)
    parser.add_argument('--accuracy', type=int, default=8)
    main(args = parser.parse_args())
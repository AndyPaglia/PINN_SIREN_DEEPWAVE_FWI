"""
Full Waveform Inversion using SIREN PINN with Deepwave

PERTURBATION-BASED APPROACH:
- SIREN learns Δv (perturbation) instead of absolute velocity
- Multi-scale frequency strategy (low → high)
- Proper shape handling for Deepwave (nz, nx)

Image and Sound Processing Lab - Politecnico di Milano
"""

import argparse
import os
import gc
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import deepwave
from deepwave import scalar

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.pinn_utils import Siren

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


# ═══════════════════════════════════════════════════════════════════════════
# DEEPWAVE FORWARD MODELING
# ═══════════════════════════════════════════════════════════════════════════

def deepwave_forward(vp, spacing, src_coordinates, rec_coordinates, 
                     f0, tn, dt, device, accuracy=8, pml_width=20):
    """
    Forward modeling with Deepwave
    
    Args:
        vp: Tensor (nz, nx) in m/s  ← IMPORTANT: nz first!
        spacing: tuple (dx, dz) in meters
        src_coordinates: Tensor (n_shots, 2) physical coords [x, z]
        rec_coordinates: Tensor (n_rec, 2) physical coords [x, z]
        f0: Dominant frequency (Hz)
        tn: Total time (ms)
        dt: Time step (ms)
        device: torch.device
        accuracy: Spatial accuracy order
        pml_width: PML width
        
    Returns:
        receiver_amplitudes: Tensor (n_shots, n_rec, nt)
    """
    
    dt_sec = dt / 1000.0
    tn_sec = tn / 1000.0
    nt = int(tn_sec / dt_sec) + 1
    peak_time = 1.5 / f0
    
    dx = float(spacing[0])
    dz = float(spacing[1]) if len(spacing) > 1 else dx
    
    n_shots = src_coordinates.shape[0]
    n_rec = rec_coordinates.shape[0]
    nz, nx = vp.shape  # ← Note: nz, nx order!
    
    # Physical coords → grid indices
    src_x_idx = (src_coordinates[:, 0] / dx).long()
    src_z_idx = (src_coordinates[:, 1] / dz).long()
    src_x_idx = torch.clamp(src_x_idx, 0, nx - 1)
    src_z_idx = torch.clamp(src_z_idx, 0, nz - 1)
    
    rec_x_idx = (rec_coordinates[:, 0] / dx).long()
    rec_z_idx = (rec_coordinates[:, 1] / dz).long()
    rec_x_idx = torch.clamp(rec_x_idx, 0, nx - 1)
    rec_z_idx = torch.clamp(rec_z_idx, 0, nz - 1)
    
    # Locations for Deepwave (nz, nx convention)
    source_locations = torch.zeros(n_shots, 1, 2, dtype=torch.long, device=device)
    source_locations[:, 0, 0] = src_z_idx  # ← z first!
    source_locations[:, 0, 1] = src_x_idx
    
    receiver_locations = torch.zeros(n_shots, n_rec, 2, dtype=torch.long, device=device)
    receiver_locations[:, :, 0] = rec_z_idx.unsqueeze(0).expand(n_shots, -1)  # ← z first!
    receiver_locations[:, :, 1] = rec_x_idx.unsqueeze(0).expand(n_shots, -1)
    
    # Wavelet
    source_amplitudes = (
        deepwave.wavelets.ricker(f0, nt, dt_sec, peak_time)
        .repeat(n_shots, 1, 1)
        .to(device)
    )
    
    # Forward modeling
    out = scalar(
        vp,
        dx,
        dt_sec,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        accuracy=accuracy,
        pml_width=pml_width,
        pml_freq=f0
    )
    
    return out[-1]


# ═══════════════════════════════════════════════════════════════════════════
# MAIN FWI FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def main(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    # -------------------------------------------------------------------------
    # LOAD OBSERVED DATA
    # -------------------------------------------------------------------------
    print(f"\n📂 Loading observed data from: {args.obs_data_path}")
    npzfile = np.load(args.obs_data_path, allow_pickle=True)
    
    d_obs_np = npzfile["d_obs_list"]  # (n_shots, n_rec, nt)
    spacing_npz = npzfile["spacing"]
    src_coordinates_np = npzfile["src_coordinates"]
    rec_coordinates_np = npzfile["rec_coordinates"]
    
    f0_max = float(npzfile["f0"])
    dt = float(npzfile["dt"])
    tn = float(npzfile["tn"])
    pml_width = int(npzfile["nbl"]) if "nbl" in npzfile else args.nbl
    
    if isinstance(spacing_npz, np.ndarray):
        dx = float(spacing_npz[0])
        dz = float(spacing_npz[1]) if len(spacing_npz) > 1 else dx
    else:
        dx = dz = float(spacing_npz)
    
    spacing = (dx, dz)
    
    n_shots_total, n_rec, nt = d_obs_np.shape
    print(f"📊 Data shape: {d_obs_np.shape} (shots × receivers × time)")
    print(f"   Spacing: dx={dx}m, dz={dz}m | f0_max: {f0_max}Hz | dt: {dt}ms | tn: {tn}ms")
    
    # -------------------------------------------------------------------------
    # LOAD SMOOTH MODEL (INITIAL MODEL)
    # -------------------------------------------------------------------------
    print(f"\n📂 Loading smooth initial model from: {args.smooth_model_path}")
    smooth_npz = np.load(args.smooth_model_path)
    vp_smooth_np = smooth_npz["vp"]  # (nx, nz) as loaded
    
    # CRITICAL: Transpose to (nz, nx) for Deepwave!
    vp_smooth = torch.from_numpy(vp_smooth_np).float().T.contiguous().to(device)
    nz, nx = vp_smooth.shape
    
    print(f"   Smooth model shape: {vp_smooth.shape} (nz × nx)")
    print(f"   Velocity range: [{vp_smooth.min().item():.1f}, {vp_smooth.max().item():.1f}] m/s")
    
    # Extract model name
    data_filename = os.path.basename(args.obs_data_path)
    if "marmousi" in data_filename.lower():
        model_name = "marmousi"
    else:
        model_name = data_filename.split("_deepwave")[0]
    
    print(f"🏔️  Model: {model_name}")
    
    # -------------------------------------------------------------------------
    # OUTPUT DIRECTORY
    # -------------------------------------------------------------------------
    out_dir = os.path.join(
        args.out_dir,
        f"{model_name}_perturbation_lr{args.lr}_alpha{args.alpha}"
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"💾 Output: {out_dir}")
    
    # -------------------------------------------------------------------------
    # INITIALIZE SIREN (for perturbation Δv)
    # -------------------------------------------------------------------------
    print(f"\n🧠 Initializing SIREN for perturbation learning...")
    
    siren = Siren(
        in_features=2,
        hidden_features=128,
        hidden_layers=4,
        out_features=1,
        outermost_linear=True,
        domain_shape=(nz, nx),  # ← Match Deepwave convention
        dh=None
    ).to(device)
    
    # Optionally load pretrained weights (but NOT required for perturbation)
    if args.siren_path and os.path.exists(args.siren_path):
        print(f"   Loading pretrained weights from: {args.siren_path}")
        checkpoint = torch.load(args.siren_path, map_location=device, weights_only=False)
        siren.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        print("   Starting from random initialization")
    
    siren.train()
    
    # Prepare normalized coordinates
    z_norm = torch.linspace(-1.0, 1.0, nz, device=device)
    x_norm = torch.linspace(-1.0, 1.0, nx, device=device)
    Z, X = torch.meshgrid(z_norm, x_norm, indexing="ij")
    coords = torch.stack([Z.reshape(-1), X.reshape(-1)], dim=-1)  # (nz*nx, 2)
    
    print(f"✅ SIREN initialized | Shape: (nz={nz}, nx={nx})")
    
    # -------------------------------------------------------------------------
    # CONVERT DATA TO TENSORS
    # -------------------------------------------------------------------------
    d_obs = torch.from_numpy(d_obs_np).float().to(device)
    src_coordinates_all = torch.from_numpy(src_coordinates_np).float().to(device)
    rec_coordinates = torch.from_numpy(rec_coordinates_np).float().to(device)
    
    # -------------------------------------------------------------------------
    # OPTIMIZER
    # -------------------------------------------------------------------------
    optimizer = torch.optim.Adam(siren.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    
    # -------------------------------------------------------------------------
    # MULTI-SCALE FWI LOOP
    # -------------------------------------------------------------------------
    LOSS_HISTORY = []
    global_iter = 0
    
    # Multi-scale frequency strategy
    freq_schedule = args.freq_schedule.split(',')
    freq_schedule = [float(f) for f in freq_schedule]
    
    print(f"\n🔥 Starting Multi-Scale FWI")
    print(f"   Frequency schedule: {freq_schedule} Hz")
    print(f"   Total iterations: {args.fwi_iterations}")
    print(f"   Shots per batch: {args.shots_per_epoch}")
    print(f"   Perturbation scale α: {args.alpha}")
    
    for f0 in freq_schedule:
        
        print(f"\n{'='*60}")
        print(f"🎵 FREQUENCY = {f0} Hz")
        print(f"{'='*60}")
        
        iters_this_freq = args.fwi_iterations // len(freq_schedule)
        
        for local_iter in tqdm(range(iters_this_freq), desc=f"FWI @ {f0}Hz"):
            
            optimizer.zero_grad()
            
            # Random shot selection
            shot_indices = np.random.choice(n_shots_total, args.shots_per_epoch, replace=False)
            src_coords_batch = src_coordinates_all[shot_indices]
            d_obs_batch = d_obs[shot_indices]
            
            # ---------------------------------------------------------------
            # PERTURBATION APPROACH
            # ---------------------------------------------------------------
            # 1. SIREN predicts Δv (perturbation)
            delta_v = siren(coords)[0].reshape(nz, nx)
            
            # 2. Add scaled perturbation to smooth model
            vp_pred = vp_smooth + args.alpha * delta_v
            
            # 3. Clamp to physical bounds
            vp_pred = torch.clamp(vp_pred, min=1500.0, max=5000.0)
            
            # ---------------------------------------------------------------
            # FORWARD MODELING
            # ---------------------------------------------------------------
            d_syn = deepwave_forward(
                vp=vp_pred,
                spacing=spacing,
                src_coordinates=src_coords_batch,
                rec_coordinates=rec_coordinates,
                f0=f0,
                tn=tn,
                dt=dt,
                device=device,
                accuracy=8,
                pml_width=pml_width
            )
            
            # ---------------------------------------------------------------
            # LOSS COMPUTATION
            # ---------------------------------------------------------------
            # Normalize by RMS for stability
            d_syn_flat = d_syn.reshape(d_syn.shape[0], -1)
            d_obs_flat = d_obs_batch.reshape(d_obs_batch.shape[0], -1)
            
            syn_rms = torch.sqrt((d_syn_flat ** 2).mean(dim=1, keepdim=True)) + 1e-10
            obs_rms = torch.sqrt((d_obs_flat ** 2).mean(dim=1, keepdim=True)) + 1e-10
            
            d_syn_norm = (d_syn_flat / syn_rms).reshape(d_syn.shape)
            d_obs_norm = (d_obs_flat / obs_rms).reshape(d_obs_batch.shape)
            
            loss = loss_fn(d_syn_norm, d_obs_norm)
            
            # ---------------------------------------------------------------
            # BACKWARD & UPDATE
            # ---------------------------------------------------------------
            loss.backward()
            
            # Gradient clipping
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(siren.parameters(), max_norm=args.clip_grad)
            
            optimizer.step()
            
            # Logging
            LOSS_HISTORY.append(loss.item())
            global_iter += 1
            
            # Debug output
            if global_iter % 50 == 0:
                print(f"\n   Iter {global_iter:05d} | Loss: {loss.item():.4e} | "
                      f"vp range: [{vp_pred.min().item():.0f}, {vp_pred.max().item():.0f}] m/s")
                
                # Check gradient norms
                grad_norms = [p.grad.norm().item() for p in siren.parameters() if p.grad is not None]
                print(f"   Grad norms: min={min(grad_norms):.2e}, max={max(grad_norms):.2e}")
            
            # ---------------------------------------------------------------
            # PERIODIC VISUALIZATION
            # ---------------------------------------------------------------
            if global_iter % args.plot_interval == 0:
                
                with torch.no_grad():
                    delta_v_plot = siren(coords)[0].reshape(nz, nx)
                    vp_plot = torch.clamp(vp_smooth + args.alpha * delta_v_plot, 1500, 5000)
                    
                    vp_plot_np = vp_plot.cpu().numpy()
                    delta_v_np = delta_v_plot.cpu().numpy()
                
                fig, axes = plt.subplots(2, 2, figsize=(16, 10))
                
                # Inverted model
                im0 = axes[0, 0].imshow(vp_plot_np, cmap='jet', aspect='auto',
                                        extent=[0, nx*dx, nz*dz, 0], vmin=1500, vmax=4500)
                axes[0, 0].set_title(f"Inverted Model (Iter {global_iter}, f={f0}Hz)")
                axes[0, 0].set_xlabel("Distance (m)")
                axes[0, 0].set_ylabel("Depth (m)")
                plt.colorbar(im0, ax=axes[0, 0], label="Velocity (m/s)")
                
                # Perturbation Δv
                vmax_dv = max(abs(delta_v_np.min()), abs(delta_v_np.max()))
                im1 = axes[0, 1].imshow(delta_v_np, cmap='seismic', aspect='auto',
                                        extent=[0, nx*dx, nz*dz, 0], 
                                        vmin=-vmax_dv, vmax=vmax_dv)
                axes[0, 1].set_title(f"Perturbation Δv (α={args.alpha})")
                axes[0, 1].set_xlabel("Distance (m)")
                axes[0, 1].set_ylabel("Depth (m)")
                plt.colorbar(im1, ax=axes[0, 1], label="Δv (m/s)")
                
                # Loss curve
                axes[1, 0].plot(LOSS_HISTORY, 'b-', linewidth=1, alpha=0.7)
                axes[1, 0].set_yscale('log')
                axes[1, 0].set_xlabel("Iteration")
                axes[1, 0].set_ylabel("MSE Loss")
                axes[1, 0].set_title("Training Loss")
                axes[1, 0].grid(True, alpha=0.3)
                
                # Velocity histogram
                axes[1, 1].hist(vp_plot_np.flatten(), bins=50, alpha=0.7, edgecolor='black')
                axes[1, 1].axvline(vp_smooth.mean().item(), color='r', 
                                   linestyle='--', label='Initial mean')
                axes[1, 1].set_xlabel("Velocity (m/s)")
                axes[1, 1].set_ylabel("Count")
                axes[1, 1].set_title("Velocity Distribution")
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                png_dir = os.path.join(out_dir, "png")
                os.makedirs(png_dir, exist_ok=True)
                plt.savefig(f"{png_dir}/iter_{global_iter:05d}_f{f0:.0f}Hz.png", 
                           dpi=150, bbox_inches='tight')
                
                if args.plot:
                    plt.show()
                plt.close()
                
                # Save velocity
                npy_dir = os.path.join(out_dir, "npy")
                os.makedirs(npy_dir, exist_ok=True)
                np.save(f"{npy_dir}/vp_iter_{global_iter:05d}.npy", vp_plot_np)
            
            # Memory cleanup
            torch.cuda.empty_cache()
            gc.collect()
    
    # -------------------------------------------------------------------------
    # SAVE FINAL RESULTS
    # -------------------------------------------------------------------------
    print("\n💾 Saving final results...")
    
    with torch.no_grad():
        delta_v_final = siren(coords)[0].reshape(nz, nx)
        vp_final = torch.clamp(vp_smooth + args.alpha * delta_v_final, 1500, 5000)
        vp_final_np = vp_final.cpu().numpy()
    
    # Save results
    out_path = os.path.join(out_dir, f"{model_name}_fwi_perturbation.npz")
    np.savez(
        out_path,
        vp=vp_final_np,
        vp_smooth=vp_smooth.cpu().numpy(),
        delta_v=delta_v_final.cpu().numpy(),
        loss_history=np.array(LOSS_HISTORY),
        alpha=args.alpha,
        freq_schedule=freq_schedule
    )
    
    # Save SIREN
    model_path = os.path.join(out_dir, f"{model_name}_siren_perturbation.pth")
    torch.save({
        "state_dict": siren.state_dict(),
        "domain_shape": (nz, nx),
        "alpha": args.alpha,
        "final_loss": LOSS_HISTORY[-1]
    }, model_path)
    
    print(f"✅ FWI Complete!")
    print(f"   Results: {out_path}")
    print(f"   Model: {model_path}")
    print(f"   Final loss: {LOSS_HISTORY[-1]:.6e}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description='FWI with SIREN Perturbation Learning + Deepwave'
    )
    
    # Data paths
    parser.add_argument('--obs_data_path', type=str, required=True,
                       help='Path to observed data NPZ')
    parser.add_argument('--smooth_model_path', type=str, required=True,
                       help='Path to smooth initial model NPZ')
    parser.add_argument('--siren_path', type=str, default=None,
                       help='Optional: pretrained SIREN checkpoint')
    parser.add_argument('--out_dir', type=str, 
                       default="./data/output2/deepwave_fwi_perturbation")
    
    # Training params
    parser.add_argument('--fwi_iterations', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--shots_per_epoch', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='Perturbation scaling factor')
    
    # Multi-scale
    parser.add_argument('--freq_schedule', type=str, default='5.0,7.0,10.0',
                       help='Comma-separated frequency schedule')
    
    # Physics
    parser.add_argument('--nbl', type=int, default=20)
    
    # Visualization
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--plot_interval', type=int, default=100)
    
    # Stability
    parser.add_argument('--clip_grad', type=float, default=1.0)
    
    args = parser.parse_args()
    
    main(args)
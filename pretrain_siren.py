"""
Pretrain SIREN for FWI - Version 2 (Deepwave compatible)

La SIREN impara DIRETTAMENTE le velocità in m/s usando sigmoid per limitare il range.
Nessuna normalizzazione/denormalizzazione = codice FWI molto più semplice!

Image and Sound Processing Lab - Politecnico di Milano
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.pinn_utils_deepwave import Siren, set_gpu


def main(args):
    # --------------------------------------------------
    # DEVICE SETUP
    # --------------------------------------------------
    set_gpu(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [PRETRAIN V2] Using device: {device}")

    # --------------------------------------------------
    # LOAD SMOOTH MODEL (m/s)
    # --------------------------------------------------
    if not os.path.exists(args.vp_model_path):
        print(f"❌ Error: {args.vp_model_path} not found.")
        return

    npz = np.load(args.vp_model_path)
    vp = npz["vp"]  # (nx, nz) in m/s
    spacing = npz["spacing"]
    
    vp_tensor = torch.from_numpy(vp).float().to(device)
    
    print(f"📊 Model Shape: {vp.shape}")
    print(f"   Velocity range: {vp.min():.1f} - {vp.max():.1f} m/s")

    # --------------------------------------------------
    # PHYSICAL BOUNDS (per sigmoid)
    # --------------------------------------------------
    # Usiamo i valori del modello smoothato con un po' di margine
    vmin = max(1500.0, float(vp.min()) - 100.0)
    vmax = min(5000.0, float(vp.max()) + 100.0)
    
    print(f"   Target bounds: [{vmin:.1f}, {vmax:.1f}] m/s")

    # --------------------------------------------------
    # SIREN ARCHITECTURE
    # --------------------------------------------------
    model = Siren(
        in_features=2,
        out_features=1,
        hidden_features=128,
        hidden_layers=4,
        outermost_linear=True,
        domain_shape=vp.shape
    ).to(device)

    coords = model.coords.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    
    # Mixed precision training
    scaler = torch.amp.GradScaler("cuda")

    # --------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------
    losses = []
    pbar = tqdm(range(args.epochs), desc="Training SIREN (V2)")
    
    for epoch in pbar:
        optimizer.zero_grad()
        
        with torch.amp.autocast("cuda"):
            # 1. SIREN output raw (unbounded)
            vp_pred_raw = model(coords)[0].reshape(vp.shape)
            
            # 2. Sigmoid per limitare a [vmin, vmax]
            # sigmoid(x) ∈ [0, 1] → scala a [vmin, vmax]
            vp_pred = vmin + (vmax - vmin) * torch.sigmoid(vp_pred_raw)
            
            # 3. Loss DIRETTAMENTE in m/s (no normalizzazione!)
            loss = loss_fn(vp_pred, vp_tensor)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        losses.append(loss.item())
        pbar.set_postfix({"Loss": f"{loss.item():.2e}", 
                         "vp_range": f"[{vp_pred.min().item():.0f}, {vp_pred.max().item():.0f}]"})

        # Plot periodico
        if args.plot and epoch % 500 == 0:
            plt.figure(figsize=(14, 5))
            
            # Subplot 1: SIREN reconstruction
            plt.subplot(1, 3, 1)
            curr_vp = vp_pred.detach().cpu().numpy()
            plt.imshow(curr_vp.T, cmap="jet", aspect="auto", vmin=vmin, vmax=vmax)
            plt.title(f"SIREN Output (Epoch {epoch})")
            plt.colorbar(label="Velocity (m/s)")
            plt.xlabel("X")
            plt.ylabel("Z")
            
            # Subplot 2: Target (smooth model)
            plt.subplot(1, 3, 2)
            plt.imshow(vp.T, cmap="jet", aspect="auto", vmin=vmin, vmax=vmax)
            plt.title("Target (Smooth Model)")
            plt.colorbar(label="Velocity (m/s)")
            plt.xlabel("X")
            plt.ylabel("Z")
            
            # Subplot 3: Loss curve
            plt.subplot(1, 3, 3)
            plt.semilogy(losses)
            plt.title("Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("MSE Loss")
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()

    # --------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------
    print("\n✅ Training Complete! Final validation...")
    with torch.no_grad():
        vp_final_raw = model(coords)[0].reshape(vp.shape)
        vp_final = vmin + (vmax - vmin) * torch.sigmoid(vp_final_raw)
        
        final_loss = loss_fn(vp_final, vp_tensor).item()
        relative_error = torch.abs(vp_final - vp_tensor).mean() / vp_tensor.mean() * 100
        
        print(f"   Final MSE Loss: {final_loss:.4e}")
        print(f"   Relative Error: {relative_error.item():.2f}%")
        print(f"   Output range: [{vp_final.min().item():.1f}, {vp_final.max().item():.1f}] m/s")

    # --------------------------------------------------
    # SAVE MODEL & PARAMETERS
    # --------------------------------------------------
    out_dir = "./data/siren"
    os.makedirs(out_dir, exist_ok=True)
    
    # Usa nome con suffisso _v2 per distinguere dalla vecchia versione
    file_name = os.path.basename(args.vp_model_path).replace(".npz", "_new.pth")
    save_path = os.path.join(out_dir, file_name)
    
    # Salva SOLO i parametri necessari (no mean/std!)
    torch.save({
        "state_dict": model.state_dict(),
        "vmin": vmin,
        "vmax": vmax,
        "domain_shape": vp.shape,
        "spacing": spacing,
        "final_loss": final_loss,
        "version": "v2_sigmoid"
    }, save_path)
    
    print(f"\n💾 Model saved to: {save_path}")
    print(f"   vmin: {vmin:.1f} m/s")
    print(f"   vmax: {vmax:.1f} m/s")
    print(f"   domain: {vp.shape}")
    print("\n🎯 Ready for FWI with fwi_deepwave_pinn.py!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Pretrain SIREN V2 for FWI (Deepwave compatible)'
    )
    parser.add_argument("--vp_model_path", type=str, required=True,
                       help="Path to smoothed velocity model (.npz)")
    parser.add_argument("--epochs", type=int, default=2000,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--plot", action="store_true",
                       help="Show plots during training")
    
    args = parser.parse_args()
    main(args)
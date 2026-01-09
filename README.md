# PINN_SIREN_DEEPWAVE_FWI
FWI with SIREN architecture and DEEPWAVE wave propagator

1) Parto dal modello di velocità vero.
2) Faccio il forward modeling.
3) Uso la funzione smoothedModel.py per creare il modello di velocità smoothed.
4) Faccio un check dei dati npz e pth per le dimensioni: (venv_TOTEST)
5) (base) apaglialunga@ispl-dirac:/nas/home/apaglialunga/SIRENdeepwave_TOTEST$ python check_data.py
--- CHECK NPZ ---
Shape d_obs: (121, 481, 1501) -> (Shots, Rec, Time)
Spacing: 25 Max X nelle coordinate: 12000.0
Max Z nelle coordinate: 20.0
 --- CHECK VELOCITY MODEL (SMOOTHED) ---
Shape vp: (481, 121) -> (NX, NZ)
Spacing: [25 25] Dimensioni calcolate: X=12025m, Z=3025m
--- CHECK SIREN ---
Domain Shape nel checkpoint (pth): (481, 121)
Dimensione fisica asse 0: 12000 metri Dimensione fisica asse 1: 3000 metri
Conclusione: L'asse 0 della SIREN è la X (Larghezza)

5) Faccio il pretrain della siren.
6) FWI

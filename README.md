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
6) FWI:
Questo script implementa una full waveform inversion (FWI) usando:
- deepwave: risolve equazione onde
- SIREN PINN: una rete neurale con attivazioni delle sinusoidi
- approccio usato: perturbazione. La rete apprende delta_v, quindi una perturbazione dal modello smoothed iniziale, quindi va a perturabre i pesi della SIREN per andare a modificare e migliorare il modello di velocità.

Il codice si divide in 4 parti:
1) forward modeling con deepwave --> simula i dati sismici sintetici a partire da un modello di velocità.
2) caricamento dei dati e dei modelli --> carico i dati sismici osservati e il modello smoothed che sarebbe la conoscenza a priori (low frequency content).
3) inversion FWI con la SIREN --> partiamo da un modello di velocità smooth. La SIREN impara una perturbazione. Aggiornando i pesi della SIREN, miglioriamo progressivamente il modello finale. Quindi, perturbiamo il modello smooth attraverso i  pesi della SIREN. Quindi noi non stiamo perturbando direttamente il modello, ma stiamo ottimizzando i pesi della SIREN, che parametrizzano una funzione continua.

v(x,z) = vsmooth(x,z) + alpha * delta_v_theta(x,z).
Dove vsmooth è fisso. delta_v_theta(x,z) è la SIREN. Theta sono i pesi della rete. L'ottimizzazione viene fatta su theta. 

IMPORTANTE: Partiamo da un modello di velocità smooth fissato e ottimizziamo una rete SIREN che parametrizza una perturbazione continua del modello. L’ottimizzazione avviene sui pesi della rete, che vengono aggiornati in modo da produrre una perturbazione tale che il modello risultante riproduca i dati osservati.

Questo riduce il cycle skipping: Il cycle skipping è un fenomeno per cui la funzione di costo presenta minimi locali quando il dato sintetico è sfasato di più di mezza lunghezza d’onda rispetto a quello osservato, portando l’inversione a convergere verso modelli errati. In piu, ntroduce regolarizzazione implicita: La regolarizzazione è implicita, in quanto la parametrizzazione del modello tramite una rete SIREN e l’uso di un approccio a perturbazione limitano lo spazio delle soluzioni ammissibili, senza introdurre termini di regolarizzazione espliciti nella funzione di costo.

4) salvataggio e visualizzazione dei risultati.


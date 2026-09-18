import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


# Definizione del Modello Generatore
# Il suo compito e' trasformare numeri casuali (spazio latente) in coordinate (x, y).
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Definiamo la struttura della rete neurale: una sequenza di strati lineari.
        self.net = nn.Sequential(
            # Primo strato: riceve il rumore (dimensione input_dim) e lo espande a 64 neuroni.
            nn.Linear(input_dim, 64),
            # Attivazione ReLU: introduce non-linearita' per permettere l'apprendimento di forme complesse.
            nn.ReLU(),
            # Secondo strato: mantiene la complessita' a 64 neuroni.
            nn.Linear(64, 64),
            nn.ReLU(),
            # Strato finale: riduce la dimensione a output_dim (che sara' 2, ovvero X e Y).
            nn.Linear(64, output_dim)
        )
    
    def forward(self, z):
        # La funzione forward definisce il passaggio del dato attraverso la rete.
        # Prende 'z' (rumore casuale) e restituisce una coppia di coordinate predetta.
        return self.net(z)

# Definizione del Modello Discriminatore
# Agisce come un poliziotto o un critico d'arte che deve smascherare i falsi.
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # La struttura e' simile al generatore ma termina con un singolo valore (probabilita').
        self.net = nn.Sequential(
            # Riceve coordinate (x, y) e inizia l'analisi.
            nn.Linear(input_dim, 64),
            # LeakyReLU previene il problema dei neuroni "morti", lasciando passare un piccolo segnale negativo.
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            # Strato finale: produce un unico numero.
            nn.Linear(32, 1),
            # Sigmoide: trasforma l'output in un valore tra 0 (sicuramente falso) e 1 (sicuramente vero).
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Riceve punti reali o generati e restituisce la probabilita' che siano reali.
        return self.net(x)

# Funzione per ottenere campioni reali (Il Target)
# Generiamo matematicamente dei punti che formano un cerchio con raggio 1.
def get_real_samples(n):
    # Generiamo 'n' angoli casuali tra 0 e 2*Pi greco.
    theta = torch.rand(n) * 2 * np.pi
    # Calcoliamo le coordinate X e Y usando le funzioni trigonometriche.
    x = torch.cos(theta)
    y = torch.sin(theta)
    # Uniamo i tensori e aggiungiamo un leggero rumore bianco per rendere i dati piu' naturali.
    data = torch.stack([x, y], dim=1) + torch.randn(n, 2) * 0.05
    return data

# --- FASE DI CONFIGURAZIONE ---

# Dimensione del vettore di rumore (spazio latente)
latent_dim = 10 
# Istanza dei due modelli
G = Generator(latent_dim, 2) # Il generatore accetta 10 numeri casuali e restituisce 2 coordinate.
D = Discriminator(2)        # Il discriminatore accetta 2 coordinate e restituisce un giudizio.

# Ottimizzatori: Adam e' un algoritmo efficiente per aggiornare i pesi delle reti.
# Gestiamo gli ottimizzatori separatamente perche' i modelli imparano in tempi diversi.
opt_G = optim.Adam(G.parameters(), lr=0.001)
opt_D = optim.Adam(D.parameters(), lr=0.001)

# Funzione di perdita: Binary Cross Entropy (BCE).
# Ideale per problemi di classificazione binaria (Vero/Falso).
criterion = nn.BCELoss()

# Liste per monitorare i progressi durante l'addestramento.
history = []
# Momenti prestabiliti in cui scattare una "foto" dello stato del generatore.
plot_snapshots = list(range(0, 10000, 1000))

print("Inizio addestramento competitivo...")
print("-" * 65)

# --- CICLO DI ADDESTRAMENTO ---

for epoch in range(10000):
    
    # --- FASE 1: ADDESTRAMENTO DEL DISCRIMINATORE ---
    # In questa fase vogliamo che D impari a distinguere bene il vero dal falso.
    opt_D.zero_grad() # Resetta i gradienti accumulati.
    
    # 1.1 Allenamento sui dati REALI
    real_data = get_real_samples(64) # Preleviamo un gruppo (batch) di 64 punti reali.
    d_real_decision = D(real_data) # D valuta i punti.
    # L'obiettivo e' che D risponda 1 (Real) per questi punti.
    d_loss_real = criterion(d_real_decision, torch.ones(64, 1))
    
    # 1.2 Allenamento sui dati FALSI (Generati)
    noise = torch.randn(64, latent_dim) # Generiamo rumore casuale.
    # G crea dei punti partendo dal rumore.
    # .detach() serve a non calcolare i gradienti per G in questa fase, poiche' stiamo allenando solo D.
    fake_data = G(noise).detach()
    d_fake_decision = D(fake_data) # D valuta i punti creati da G.
    # L'obiettivo e' che D risponda 0 (Fake) per questi punti.
    d_loss_fake = criterion(d_fake_decision, torch.zeros(64, 1))
    
    # Calcolo dell'errore totale di D e aggiornamento dei pesi.
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward() # Calcola come cambiare i pesi per ridurre l'errore.
    opt_D.step()      # Applica il cambiamento dei pesi.
    
    # --- FASE 2: ADDESTRAMENTO DEL GENERATORE ---
    # In questa fase vogliamo che G impari a ingannare D.
    opt_G.zero_grad() # Resetta i gradienti.
    
    noise = torch.randn(64, latent_dim) # Generiamo nuovo rumore.
    gen_data = G(noise) # G crea nuovi punti.
    
    # G passa i suoi punti a D.
    # IMPORTANTE: Qui l'obiettivo di G e' che D risponda 1 (ovvero che creda che i falsi siano veri).
    g_loss = criterion(D(gen_data), torch.ones(64, 1))
    
    g_loss.backward() # Calcola come migliorare G per ingannare meglio D.
    opt_G.step()      # Applica il miglioramento a G.
    
    # Stampa dei progressi ogni 500 iterazioni.
    if epoch % 500 == 0:
        print(f"Iterazione {epoch:4d} | Perdita Discriminatore: {d_loss.item():.4f} | Perdita Generatore: {g_loss.item():.4f}")

    # Salvataggio dei punti generati per la visualizzazione finale.
    if epoch in plot_snapshots:
        with torch.no_grad(): # Non calcoliamo gradienti durante la generazione dei test.
            test_noise = torch.randn(500, latent_dim)
            generated_points = G(test_noise).numpy()
            history.append((epoch, generated_points))

print("-" * 65)
print("Addestramento completato. Preparazione dei grafici...")

# --- VISUALIZZAZIONE DEI RISULTATI ---

# Creiamo una griglia 2x5 per mostrare l'evoluzione in 10 passaggi.
fig, axes = plt.subplots(2, 5, figsize=(20, 10))
axes = axes.flatten()

# Otteniamo punti reali come riferimento visivo (Target).
real_p = get_real_samples(500).numpy()

for i, (ep, points) in enumerate(history):
    # Rappresentiamo i punti del cerchio reale in blu semitrasparente.
    axes[i].scatter(real_p[:, 0], real_p[:, 1], c='blue', alpha=0.1, label='Target Reale')
    # Rappresentiamo i punti creati dal Generatore in rosso.
    axes[i].scatter(points[:, 0], points[:, 1], c='red', alpha=0.5, label='Punti Generati')
    
    axes[i].set_title(f"Evoluzione a {ep} step")
    axes[i].set_xlim(-2, 2)
    axes[i].set_ylim(-2, 2)
    axes[i].set_aspect('equal') # Mantiene le proporzioni corrette (il cerchio non sembra un'ellisse).
    if i == 0: 
        axes[i].legend(loc='upper right', fontsize='small')

# Titolo generale e ottimizzazione del layout.
plt.suptitle("EVOLUZIONE DELLA RETE GAN: DALLO STATO CASUALE ALLA FORMA CIRCOLARE", fontsize=18, fontweight='bold')
plt.tight_layout()
plt.show()
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Inizializzazione e preparazione dei dati
# Il dataset Iris contiene 4 caratteristiche per ogni fiore. L'obiettivo e' ridurre queste 4 dimensioni a 2.
iris = load_iris()
X = iris.data  # Matrice delle caratteristiche (150 campioni, 4 feature)
y = iris.target  # Etichette delle classi (Setosa, Versicolor, Virginica)
target_names = iris.target_names

# La standardizzazione e' necessaria perche' sia la PCA che l'Autoencoder sono sensibili alla scala dei dati.
# Trasformiamo i dati in modo che abbiano media 0 e deviazione standard 1.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convertiamo i dati in un tensore PyTorch per darli in pasto alla rete neurale dell'Autoencoder.
X_tensor = torch.FloatTensor(X_scaled)

# --- FASE A: PCA (Principal Component Analysis) ---
# La PCA e' una tecnica lineare. Cerca le direzioni (componenti principali) che mantengono la massima varianza.
# Qui riduciamo direttamente le 4 dimensioni originali a 2 componenti principali.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# --- FASE B: Autoencoder (Rete Neurale per Riduzione Non Lineare) ---
# Definiamo la struttura dell'autoencoder ereditando da nn.Module di PyTorch.
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # L'Encoder comprime l'input.
        # Passiamo da 4 feature a 8 intermedie, poi a 2 (spazio latente o bottleneck).
        # L'obiettivo e' forzare la rete a imparare una rappresentazione compressa ma significativa.
        self.encoder = nn.Sequential(
            nn.Linear(4, 8),   # Strato lineare: input 4 -> output 8
            nn.ReLU(),         # Funzione di attivazione non lineare per catturare relazioni complesse
            nn.Linear(8, 2)    # Strato finale dell'encoder: output 2 (le nostre nuove coordinate)
        )
        
        # Il Decoder tenta di ricostruire i dati originali partendo dalle 2 dimensioni compresse.
        # Se la ricostruzione e' accurata, significa che le 2 dimensioni contengono le info essenziali.
        self.decoder = nn.Sequential(
            nn.Linear(2, 8),   # Espansione: 2 -> 8
            nn.ReLU(),         # Attivazione
            nn.Linear(8, 4)    # Ritorno alle 4 dimensioni originali
        )

    def forward(self, x):
        # Il metodo forward definisce il flusso dei dati attraverso la rete.
        encoded = self.encoder(x)   # Fase di compressione
        decoded = self.decoder(encoded)  # Fase di ricostruzione
        return encoded, decoded

# Inizializzazione del modello e degli strumenti di ottimizzazione
model = Autoencoder()
# MSELoss (Mean Squared Error) misura la differenza tra l'input originale e la ricostruzione del decoder.
criterion = nn.MSELoss()
# Adam e' l'algoritmo che modifica i pesi della rete per minimizzare l'errore (loss).
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Ciclo di addestramento (Training Loop)
print("Inizio addestramento Autoencoder...")
for epoch in range(5000):
    # Passaggio in avanti (Forward pass): otteniamo la rappresentazione compressa e quella ricostruita.
    encoded, decoded = model(X_tensor)
    
    # Calcolo dell'errore confrontando la ricostruzione con i dati originali standardizzati.
    loss = criterion(decoded, X_tensor)
    
    # Fase di ottimizzazione:
    optimizer.zero_grad()  # Reset dei gradienti dai cicli precedenti
    loss.backward()        # Calcolo del gradiente tramite backpropagation
    optimizer.step()       # Aggiornamento dei pesi della rete
    
    # Stampa del progresso ogni 50 epoche
    if (epoch + 1) % 50 == 0:
        print(f'Epoca [{epoch+1}], Loss: {loss.item():.4f}')

# Estrazione dello spazio latente (le coordinate ridotte a 2 dimensioni)
# Usiamo torch.inference_mode() per disabilitare il calcolo del gradiente e risparmiare memoria.
with torch.inference_mode():
    X_ae, _ = model(X_tensor)
    X_ae = X_ae.numpy()  # Convertiamo il tensore in array NumPy per la visualizzazione

# --- FASE C: Visualizzazione e Confronto ---
# Creiamo due grafici affiancati per confrontare i risultati della PCA e dell'Autoencoder.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Cicliamo sulle tre classi di Iris per assegnare colori diversi nel plot.
for i, color in zip(range(len(target_names)), colors):
    # Visualizzazione PCA: proiezione lineare classica.
    mask = (y == i)
    ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], color=color, label=target_names[i], alpha=0.7)
    
    # Visualizzazione Autoencoder: proiezione basata sul bottleneck della rete neurale.
    ax2.scatter(X_ae[mask, 0], X_ae[mask, 1], color=color, label=target_names[i], alpha=0.7)

# Configurazione estetica del grafico PCA
ax1.set_title("PCA (Riduzione Lineare delle Dimensioni)", fontsize=14)
ax1.set_xlabel("Prima Componente Principale")
ax1.set_ylabel("Seconda Componente Principale")
ax1.legend()

# Configurazione estetica del grafico Autoencoder
ax2.set_title("Autoencoder (Riduzione Non Lineare delle Dimensioni)", fontsize=14)
ax2.set_xlabel("Dimensione Latente 1")
ax2.set_ylabel("Dimensione Latente 2")
ax2.legend()

plt.tight_layout()
plt.show()
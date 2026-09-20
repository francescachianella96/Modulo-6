import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configurazione dei parametri principali per l'esecuzione del modello
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128    # Numero di campioni analizzati per ogni iterazione di gradient descent
LR = 1e-3           # Learning Rate: velocità di apprendimento dell'ottimizzatore
EPOCHS = 5         # Numero di cicli completi attraverso l'intero set di dati
LATENT_DIM = 2     # Dimensione dello spazio latente (2D per permettere la visualizzazione diretta)

class LatentModel(nn.Module):
    """
    Classe che implementa sia un Autoencoder (AE) che un Variational Autoencoder (VAE).
    La struttura è composta da un Encoder, un collo di bottiglia (Bottleneck) e un Decoder.
    """
    def __init__(self, is_vae=True):
        super().__init__()
        self.is_vae = is_vae
        
        # L'Encoder riduce gradualmente la dimensione dell'immagine (28x28 = 784 pixel)
        # a una rappresentazione compressa più astratta.
        self.encoder_fc = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU()
        )
        
        # Definizione dello spazio latente. 
        # In un VAE, l'encoder non produce un singolo punto, ma i parametri di una distribuzione:
        # mu rappresenta la media, logvar il logaritmo della varianza.
        self.fc_mu = nn.Linear(200, LATENT_DIM)
        self.fc_logvar = nn.Linear(200, LATENT_DIM)
        
        # Il Decoder riceve un punto dallo spazio latente e cerca di ricostruire l'immagine originale.
        # L'obiettivo è ripristinare la dimensione 784 partendo da LATENT_DIM.
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid() # Sigmoid garantisce che i pixel siano nell'intervallo [0, 1]
        )

    def reparameterize(self, mu, logvar):
        """
        Applica il 'Reparameterization Trick' fondamentale per l'addestramento dei VAE.
        Permette la backpropagation attraverso un campionamento stocastico.
        Se il modello è un AE standard, restituisce direttamente la media senza campionamento.
        """
        if not self.is_vae: 
            return mu
            
        # Calcoliamo la deviazione standard dalla varianza logaritmica
        std = torch.exp(0.5 * logvar)
        # Generiamo un rumore casuale epsilon da una distribuzione normale standard
        eps = torch.randn_like(std)
        # Il punto latente z è calcolato come: media + rumore * deviazione_standard
        return mu + eps * std

    def forward(self, x):
        """
        Definisce il flusso dei dati: Immagine -> Encoder -> Spazio Latente -> Decoder -> Ricostruzione.
        """
        # Flatten dell'immagine da (1, 28, 28) a (784)
        h = self.encoder_fc(x.view(-1, 784))
        
        # Calcolo dei parametri della distribuzione latente
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Generazione del vettore latente z tramite riparametrizzazione
        z = self.reparameterize(mu, logvar)
        
        # Restituisce l'immagine ricostruita insieme ai parametri latenti per il calcolo della loss
        return self.decoder(z), mu, logvar

def loss_fn(recon_x, x, mu, logvar, beta, is_vae):
    """
    Funzione di perdita che guida l'apprendimento del modello.
    In un VAE, è composta da due termini: ricostruzione e regolarizzazione (KL Divergence).
    """
    # L'errore di ricostruzione (Binary Cross Entropy) misura quanto la copia sia simile all'originale.
    # Si somma l'errore su tutti i 784 pixel per ogni immagine del batch.
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    # Se è un Autoencoder standard, usiamo solo l'errore di ricostruzione.
    if not is_vae: 
        return BCE
    
    # La Divergenza di Kullback-Leibler (KLD) forza lo spazio latente a seguire una distribuzione normale standard.
    # Questo termine "ordina" lo spazio latente, creando cluster coerenti e aree continue.
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # La perdita totale è il bilanciamento tra fedeltà (BCE) e struttura dello spazio (KLD).
    # Il parametro beta controlla quanto peso dare alla regolarizzazione KL.
    return BCE + beta * KLD

# Caricamento del dataset MNIST (cifre scritte a mano)
transform = transforms.Compose([transforms.ToTensor()])
train_loader = DataLoader(datasets.MNIST('.', train=True, download=True, transform=transform), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(datasets.MNIST('.', train=False, transform=transform), batch_size=BATCH_SIZE, shuffle=False)

def run_experiment(is_vae, beta):
    """
    Funzione di coordinamento per l'addestramento e la successiva estrazione dei dati latenti.
    Interagisce con la classe LatentModel per il training e la funzione loss_fn per l'ottimizzazione.
    """
    model = LatentModel(is_vae=is_vae).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        model.train()
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad() # Reset dei gradienti per la nuova iterazione
            
            # Passaggio avanti (forward pass)
            recon, mu, logvar = model(data)
            
            # Calcolo della perdita globale
            loss = loss_fn(recon, data, mu, logvar, beta, is_vae)
            
            # Passaggio all'indietro (backpropagation)
            loss.backward()
            optimizer.step()
            
    # Fase di valutazione: estraiamo le coordinate dei punti nello spazio latente
    model.eval()
    z_points, labels = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            h = model.encoder_fc(data.view(-1, 784))
            # Utilizziamo la media (mu) come punto rappresentativo per la visualizzazione
            z = model.fc_mu(h) 
            z_points.append(z.cpu())
            labels.append(target) # Conserviamo l'etichetta originale (0-9) per colorare i punti
            
    return torch.cat(z_points), torch.cat(labels)

# Main script: esegue il confronto tra diverse configurazioni di modelli
print("Avvio degli esperimenti di confronto tra Autoencoder e VAE...")

# Definizione dei casi di test: tipo di modello, valore di beta e etichetta per il grafico
configs = [
    (False, 0, "Autoencoder (Senza KL)"),
    (True, 1, "VAE (Beta=1)"),
    (True, 3, "VAE (Beta=3)"),
    (True, 10, "VAE (Beta=10)")
]

results = []
for is_vae, beta, title in configs:
    print(f"Addestramento in corso: {title}...")
    z, y = run_experiment(is_vae, beta)
    results.append((z, y, title))

# Creazione della visualizzazione finale per confrontare gli spazi latenti
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
for i, (z, y, title) in enumerate(results):
    # Ogni cifra (0-9) è rappresentata da un colore diverso nello scatter plot
    sc = axes[i].scatter(z[:, 0], z[:, 1], c=y, cmap='tab10', s=2, alpha=0.6)
    axes[i].set_title(title)
    # Scale fisse per rendere il confronto visivo oggettivo tra i grafici
    axes[i].set_xlim(-5, 5)
    axes[i].set_ylim(-5, 5) 
    axes[i].grid(True, linestyle='--', alpha=0.5)

# Aggiunta di una barra dei colori per indicare a quale cifra corrisponde ogni colore
plt.colorbar(sc, ax=axes[-1], label='Etichetta Digitale')
plt.tight_layout()
plt.show()
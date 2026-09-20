import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# --- CONFIGURAZIONE E PARAMETRI DI ADDESTRAMENTO ---

# Selezione automatica del dispositivo: usa la GPU (cuda) se disponibile, altrimenti la CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Numero di immagini elaborate contemporaneamente durante l'addestramento.
BATCH_SIZE = 128

# Numero di variabili che compongono la "chiave" di compressione (spazio latente).
# Rappresenta le caratteristiche astratte dell'immagine.
LATENT_DIM = 20

# Quante volte il modello vedrà l'intero dataset durante l'addestramento.
EPOCHS = 10

# Tasso di apprendimento: controlla quanto velocemente il modello aggiorna i suoi parametri.
LR = 1e-3

# --- PREPARAZIONE DEI DATI ---

# Trasformiamo i dati in tensori (formato numerico per PyTorch).
# Questo normalizza i pixel tra 0 e 1, fondamentale per l'interpretazione probabilistica del VAE.
transform = transforms.ToTensor()

# Scarichiamo e carichiamo il dataset MNIST (immagini di cifre scritte a mano).
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Il DataLoader si occupa di dividere il dataset in batch e di mescolare i dati a ogni epoca.
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- ARCHITETTURA DEL VARIATIONAL AUTOENCODER (VAE) ---

class VAE(nn.Module):
    """
    Classe principale che definisce la struttura del Variational Autoencoder.
    Il VAE non si limita a comprimere i dati, ma impara a modellare la loro distribuzione.
    """
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        
        # L'Encoder comprime l'immagine di input (784 pixel) in una rappresentazione intermedia.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU() # Funzione di attivazione per introdurre non-linearità.
        )
        
        # Invece di produrre un singolo punto, l'encoder produce i parametri di una distribuzione:
        # 1. Media (mu): il valore centrale della rappresentazione latente.
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        # 2. Log-varianza (logvar): quanto è "diffusa" o incerta questa rappresentazione.
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Il Decoder riceve un punto dallo spazio latente e cerca di ricostruire l'immagine originale.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            # Sigmoid assicura che i pixel dell'uscita siano tra 0 e 1 (interpretazione probabilistica).
            nn.Sigmoid() 
        )

    def reparameterize(self, mu, logvar):
        """
        Il Reparameterization Trick permette di campionare dalla distribuzione in modo derivabile.
        Senza questo trucco, il gradiente non potrebbe passare attraverso il campionamento casuale.
        """
        # Calcoliamo la deviazione standard dalla log-varianza.
        std = torch.exp(0.5 * logvar)
        # Generiamo rumore casuale da una distribuzione normale standard.
        eps = torch.randn_like(std)
        # Il punto nello spazio latente è: media + (rumore * deviazione standard).
        return mu + eps * std

    def forward(self, x):
        """
        Definisce il flusso dei dati: Immagine -> Encoder -> Spazio Latente -> Decoder -> Ricostruzione.
        """
        # Appiattiamo l'immagine 28x28 in un vettore di 784 elementi.
        h = self.encoder(x.view(-1, 784))
        
        # Otteniamo i parametri della distribuzione latente.
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        
        # Campioniamo un punto 'z' usando il trucco della riparametrizzazione.
        z = self.reparameterize(mu, logvar)
        
        # Restituiamo l'immagine ricostruita, la media e la log-varianza (necessarie per la loss).
        return self.decoder(z), mu, logvar

# --- FUNZIONE DI PERDITA (VAE LOSS / ELBO) ---

def vae_loss_function(recon_x, x, mu, logvar):
    """
    La loss del VAE (ELBO) bilancia due obiettivi opposti:
    1. Fedeltà della ricostruzione.
    2. Regolarità dello spazio latente.
    """
    
    # 1. Perdita di Ricostruzione (Binary Cross Entropy):
    # Confronta pixel per pixel l'immagine originale con quella generata dal decoder.
    # Sommiamo tutte le discrepanze per l'intero batch.
    recon_loss = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # 2. Divergenza KL (Regolarizzazione):
    # Forza lo spazio latente a seguire una distribuzione normale standard N(0, 1).
    # Evita che il modello memorizzi semplicemente gli input anziché imparare le caratteristiche.
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # La perdita totale è la somma dei due termini.
    return recon_loss + kl_loss

# --- CICLO DI ADDESTRAMENTO ---

# Inizializziamo il modello e lo spostiamo sulla GPU/CPU.
model = VAE(latent_dim=LATENT_DIM).to(device)

# L'ottimizzatore Adam aggiornerà i pesi del modello per minimizzare la loss.
optimizer = optim.Adam(model.parameters(), lr=LR)

print(f"Inizio addestramento su {device}...")

for epoch in range(EPOCHS):
    model.train() # Imposta il modello in modalità addestramento.
    train_loss = 0
    
    for data, _ in train_loader:
        # Spostiamo i dati sul dispositivo corretto.
        data = data.to(device)
        
        # Resettiamo i gradienti accumulati per evitare interferenze tra i batch.
        optimizer.zero_grad()
        
        # Passaggio in avanti (Forward Pass): otteniamo la ricostruzione e i parametri latenti.
        recon_batch, mu, logvar = model(data)
        
        # Calcoliamo quanto errore ha commesso il modello (ELBO loss).
        loss = vae_loss_function(recon_batch, data, mu, logvar)
        
        # Calcoliamo i gradienti (Backpropagation).
        loss.backward()
        
        # Accumuliamo la loss per le statistiche dell'epoca.
        train_loss += loss.item()
        
        # Aggiorniamo i pesi del modello.
        optimizer.step()
        
    # Calcolo e stampa della perdita media per immagine in questa epoca.
    avg_loss = train_loss / len(train_loader.dataset)
    print(f"Epoca [{epoch+1}/{EPOCHS}] - Perdita Media: {avg_loss:.2f}")

# --- VALUTAZIONE E VISUALIZZAZIONE DEI RISULTATI ---

# Passiamo alla modalità valutazione (disabilita calcoli non necessari come il dropout se presente).
model.eval()

# Disabilitiamo il calcolo dei gradienti per risparmiare memoria durante l'interferenza.
with torch.inference_mode():
    # Prendiamo un batch di immagini reali per confrontarle con le ricostruzioni del VAE.
    data, _ = next(iter(train_loader))
    data = data.to(device)
    
    # Chiediamo al modello di ricostruire queste immagini.
    recon, _, _ = model(data)

    # Visualizziamo le prime 10 immagini originali e le loro relative ricostruzioni.
    n = 10 
    plt.figure(figsize=(20, 6))
    for i in range(n):
        # Prima riga: Immagini Originali.
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(data[i].cpu().view(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0: ax.set_title("Originali", fontsize=15)

        # Seconda riga: Immagini Ricostruite dal VAE.
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(recon[i].cpu().view(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0: ax.set_title("Ricostruzioni VAE", fontsize=15)
        
    plt.tight_layout()
    plt.show()
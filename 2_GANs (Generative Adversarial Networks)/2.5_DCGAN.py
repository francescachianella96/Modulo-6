import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# 1. CONFIGURAZIONE E IPERPARAMETRI
# -------------------------------------------------------------------------
# Selezione del dispositivo di calcolo: GPU se disponibile, altrimenti CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dimensione del vettore di rumore casuale (spazio latente) in ingresso al generatore.
latent_dim = 100

# Parametri di ottimizzazione standard per DCGAN (Deep Convolutional GAN).
# Un tasso di apprendimento di 0.0002 e un beta1 di 0.5 sono consigliati per la stabilita'.
lr = 0.0002
beta1 = 0.5 
batch_size = 128
epochs = 15

# Trasformazioni per il dataset:
# 1. ToTensor: Converte l'immagine PIL in un tensore PyTorch [0, 1].
# 2. Normalize: Sposta i valori nel range [-1, 1]. Questo e' fondamentale perche' 
# lo strato finale del Generatore usa una funzione Tanh che produce valori tra -1 e 1.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Caricamento del dataset FashionMNIST.
# Viene scaricato localmente e preparato per il caricamento a blocchi (batch).
train_loader = DataLoader(
    datasets.FashionMNIST('.', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)

# -------------------------------------------------------------------------
# 2. INIZIALIZZAZIONE DEI PESI
# -------------------------------------------------------------------------
# Per le architetture DCGAN, gli autori suggeriscono di inizializzare i pesi 
# seguendo una distribuzione normale con media 0 e deviazione standard 0.02.
# Questo aiuta a prevenire la saturazione delle funzioni di attivazione e 
# rende l'inizio dell'addestramento piu' fluido.
def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        # Pesi delle convoluzioni inizializzati intorno allo zero.
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        # BatchNorm inizializzato con media 1 e deviazione standard 0.02.
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# -------------------------------------------------------------------------
# 3. ARCHITETTURA: GENERATORE
# -------------------------------------------------------------------------
# Il compito della classe Generator e' trasformare un vettore di rumore (spazio latente)
# in un'immagine che sembri reale (28x28 pixel, in scala di grigi).
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Utilizziamo ConvTranspose2d (convoluzioni trasposte o de-convoluzioni) 
        # per "espandere" il vettore piatto iniziale in una griglia spaziale.
        self.main = nn.Sequential(
            # Input: (100) x 1 x 1 -> Trasformato in 256 mappe di caratteristiche 7x7.
            nn.ConvTranspose2d(latent_dim, 256, 7, 1, 0, bias=False),
            nn.BatchNorm2d(256), # Stabilizza l'apprendimento normalizzando le attivazioni.
            nn.ReLU(True),       # ReLU e' l'attivazione standard per gli strati intermedi del Generatore.
            
            # Espansione: 256 x 7 x 7 -> 128 x 14 x 14.
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # Output finale: 128 x 14 x 14 -> 1 x 28 x 28 (dimensione FashionMNIST).
            nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),
            nn.Tanh() # Restituisce valori in [-1, 1] per matchare la normalizzazione dei dati.
        )
    
    def forward(self, x):
        # Il flusso dei dati riceve il rumore e lo passa attraverso la sequenza definita sopra.
        return self.main(x)

# -------------------------------------------------------------------------
# 4. ARCHITETTURA: DISCRIMINATORE
# -------------------------------------------------------------------------
# Il Discriminatore e' un classificatore binario che deve distinguere tra immagini
# reali (provenienti dal dataset) e immagini fake (create dal Generatore).
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Al contrario del Generatore, usiamo Convoluzioni standard per ridurre i dati spaziali
        # in un singolo valore di probabilita'.
        self.main = nn.Sequential(
            # Input: 1 x 28 x 28 (Immagine) -> 128 x 14 x 14.
            nn.Conv2d(1, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True), # LeakyReLU aiuta a evitare gradienti nulli nel Discriminatore.
            
            # Riduzione ulteriore: 128 x 14 x 14 -> 256 x 7 x 7.
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output finale: 256 x 7 x 7 -> 1 x 1 x 1 (Probabilita').
            nn.Conv2d(256, 1, 7, 1, 0, bias=False),
            nn.Sigmoid() # Restituisce un valore tra 0 (Fake) e 1 (Real).
        )
    
    def forward(self, x):
        # Elabora l'immagine e restituisce un valore piatto (squeezed) per il calcolo della loss.
        return self.main(x).view(-1, 1)

# Istanza dei modelli e spostamento sulla memoria del dispositivo (GPU/CPU).
netG = Generator().to(device)
netD = Discriminator().to(device)

# Applicazione del metodo di inizializzazione dei pesi definito al punto 2.
netG.apply(weights_init)
netD.apply(weights_init)

# Definizione dei due ottimizzatori separati, uno per ogni rete.
# Anche se lavorano insieme, le reti hanno obiettivi opposti e devono aggiornare i pesi in modo indipendente.
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Binary Cross Entropy Loss: funzione di errore standard per compiti di classificazione binaria.
criterion = nn.BCELoss()

# -------------------------------------------------------------------------
# 5. CICLO DI ADDESTRAMENTO AVVERSARIALE
# -------------------------------------------------------------------------
# In questa fase le due reti interagiscono in un gioco "min-max": il Discriminatore 
# cerca di massimizzare la precisione, mentre il Generatore cerca di ingannarlo.

print(f"Inizio addestramento DCGAN su {device}...")

for epoch in range(epochs):
    for i, (real_images, _) in enumerate(train_loader):
        b_size = real_images.size(0)
        real_images = real_images.to(device)

        # -----------------------------------------------------------------
        # (A) AGGIORNAMENTO DEL DISCRIMINATORE
        # Obiettivo: Massimizzare log(D(x)) + log(1 - D(G(z)))
        # -----------------------------------------------------------------
        netD.zero_grad()
        
        # 1. Fase immagini reali: usiamo label 0.9 invece di 1.0 (Label Smoothing).
        # Questa tecnica aiuta a prevenire che il Discriminatore diventi troppo "sicuro"
        # e domini il Generatore rendendo i gradienti troppo deboli.
        label_real = torch.full((b_size, 1), 0.9, device=device)
        output_real = netD(real_images)
        lossD_real = criterion(output_real, label_real)
        
        # 2. Fase immagini fake: creiamo rumore e lo passiamo al Generatore.
        noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
        fake_images = netG(noise)
        label_fake = torch.zeros(b_size, 1, device=device)
        
        # Passiamo le immagini fake a D. Usiamo .detach() perche' in questa fase
        # vogliamo solo aggiornare D, quindi non serve calcolare i gradienti per G.
        output_fake = netD(fake_images.detach())
        lossD_fake = criterion(output_fake, label_fake)
        
        # Unione delle perdite ed esecuzione della backpropagation.
        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # -----------------------------------------------------------------
        # (B) AGGIORNAMENTO DEL GENERATORE
        # Obiettivo: Massimizzare log(D(G(z))) -> Far credere a D che l'immagine sia reale.
        # -----------------------------------------------------------------
        netG.zero_grad()
        
        # Ripassiamo le immagini fake appena create a D, ma questa volta senza detach.
        # In questo modo, i gradienti fluiranno attraverso D fino a G.
        output_g = netD(fake_images)
        
        # Per il Generatore, l'errore si calcola basandosi su quanto D si sbaglia.
        # Puntiamo alla label "reale" (0.9) per spronare G a migliorare la qualità dei dettagli.
        lossG = criterion(output_g, label_real)
        
        lossG.backward()
        optimizerG.step()

    # Log dei progressi alla fine di ogni epoca.
    print(f"Epoca [{epoch+1}/{epochs}] | Loss_D: {lossD.item():.4f} | Loss_G: {lossG.item():.4f}")

# -------------------------------------------------------------------------
# 6. GENERAZIONE E VISUALIZZAZIONE DEI RISULTATI
# -------------------------------------------------------------------------
# Dopo l'addestramento, utilizziamo il Generatore per creare campioni di test 
# partendo da un rumore fisso per vedere i progressi qualitativi.
with torch.no_grad():
    fixed_noise = torch.randn(16, latent_dim, 1, 1, device=device)
    fake_samples = netG(fixed_noise).cpu()

# Creazione di una griglia 4x4 per mostrare gli abiti/scarpe generati dal modello.
plt.figure(figsize=(8, 8))
for i in range(16):
    plt.subplot(4, 4, i+1)
    # Mostriamo solo la prima dimensione (canale grigio) del tensore (1, 28, 28).
    plt.imshow(fake_samples[i][0], cmap='gray')
    plt.axis('off')

plt.suptitle("Fashion MNIST: Campioni Generati tramite DCGAN")
plt.show()
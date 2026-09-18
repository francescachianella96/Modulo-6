import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# --- 1. CONFIGURAZIONE E PREPARAZIONE DATI ---
# Iperparametri del modello
latent_dim = 10     # Dimensione del vettore di rumore casuale in ingresso al generatore
lr = 0.0002         # Learning rate, valore basso per stabilizzare l'addestramento della GAN
batch_size = 64     # Numero di immagini processate in ogni iterazione

# Definizione delle trasformazioni per le immagini (normalizzazione e conversione in vettore pytorch).
transform = transforms.Compose([
    transforms.ToTensor(),                # Converte l'immagine in un tensore PyTorch
    transforms.Normalize((0.5,), (0.5,))  # Normalizza da [0, 1] a [-1, 1]: (valore - 0.5) / 0.5
])


# --- 2. DEFINIZIONE DELL'ARCHITETTURA ---

# Generatore (G)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # Struttura: Una serie di strati lineari (Fully Connected) che aumentano la dimensione
        # dal vettore latente (100) fino alla dimensione dell'immagine appiattita (784 = 28x28).
        self.main = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),  # LeakyReLU permette un piccolo gradiente anche per valori negativi, evitando neuroni morti
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784), # 784 corrisponde ai pixel totali di un'immagine 28x28
            nn.Tanh() # Tanh schiaccia l'output tra -1 e 1, allineandolo alla normalizzazione dei dati reali
        )

    def forward(self, x):
        # Passa il rumore attraverso la rete e rimodella l'uscita in formato immagine (Batch, Canali, Altezza, Larghezza)
        return self.main(x).view(-1, 1, 28, 28)

# Discriminatore (D)
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Struttura: Comprime l'immagine appiattita (784) in un singolo valore di probabilità.
        self.main = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid() # Sigmoid produce un valore tra 0 e 1 (0 = Falso, 1 = Reale)
        )

    def forward(self, x):
        # Appiattisce l'immagine in un vettore 1D e la passa attraverso la rete
        return self.main(x.view(-1, 784))


if __name__ == '__main__':
    # Selezioniamo la GPU se disponibile per velocizzare il calcolo, altrimenti usiamo la CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Caricamento del dataset FashionMNIST.
    train_loader = DataLoader(
        datasets.FashionMNIST('.', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True, num_workers=0
    )

    # Inizializzazione delle reti e spostamento sul dispositivo (CPU o GPU)
    netG = Generator().to(device)
    netD = Discriminator().to(device)

    # Ottimizzatori: Adam è standard per le GAN.
    # Betas (0.5, 0.999) sono valori specifici che aiutano a stabilizzare l'addestramento adversarial.
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

    # Funzione di perdita: Binary Cross Entropy Loss.
    criterion = nn.BCELoss()

    # --- 3. LOOP DI ADDESTRAMENTO ---
    # In questa fase, G e D competono in un gioco a somma zero.
    epochs = 20
    print(f"Training su {device} iniziato...")

    for epoch in range(epochs):
        # Iteriamo su ogni batch di immagini reali dal dataset
        for i, (real_images, _) in enumerate(train_loader):
            b_size = real_images.size(0) # Dimensione del batch corrente
            real_images = real_images.to(device)

            # -----------------------------------------------------------
            # FASE A: ADDESTRAMENTO DEL DISCRIMINATORE
            # Obiettivo: Massimizzare la capacità di distinguere Reale da Falso
            # -----------------------------------------------------------
            netD.zero_grad() # Azzeriamo i gradienti accumulati nel passo precedente
            
            # 1. Addestramento con immagini REALI con Label Smoothing (0.9 invece di 1.0)
            label_real = torch.full((b_size, 1), 0.9, device=device)
            
            output_real = netD(real_images)              # D analizza le immagini reali
            lossD_real = criterion(output_real, label_real) # Calcoliamo l'errore rispetto all'etichetta "Vero" (0.9)
            lossD_real.backward()                        # Calcoliamo i gradienti per la parte reale

            # 2. Addestramento con immagini FALSE
            noise = torch.randn(b_size, latent_dim, device=device)  # Generazione rumore casuale
            fake_images = netG(noise)                    # G genera immagini false dal rumore
            label_fake = torch.full((b_size, 1), 0.0, device=device) # Etichette per il "Falso" (0.0)
            
            # Passaggio delle immagini false
            output_fake = netD(fake_images.detach())
            lossD_fake = criterion(output_fake, label_fake) # Calcoliamo l'errore rispetto all'etichetta "Falso" (0.0)
            lossD_fake.backward()                        # Calcoliamo i gradienti per la parte falsa

            # Aggiorniamo i pesi del Discriminatore
            lossD = lossD_real + lossD_fake              # Somma delle perdite (solo per log)
            optimizerD.step()

            # -----------------------------------------------------------
            # FASE B: ADDESTRAMENTO DEL GENERATORE
            # Obiettivo: Massimizzare la probabilità che D sbagli (classifichi Falso come Reale)
            # -----------------------------------------------------------
            netG.zero_grad() # Azzeriamo i gradienti
            
            # Rigenerazione etichette senza Label Smoothing (da 0.9 a 1.0 float per evitare conflitti con BCELoss)
            label_real_for_g = torch.full((b_size, 1), 1.0, device=device)
            
            # Passiamo di nuovo le immagini false a D (questa volta senza detach)
            output_g = netD(fake_images) 
            
            lossG = criterion(output_g, label_real_for_g) # Errore: quanto le immagini false sono lontane dall'essere "Reali"
            lossG.backward() # Calcoliamo i gradienti (che arriveranno fino ai pesi di G)
            optimizerG.step() # Aggiorniamo i pesi del Generatore

        # Stampiamo le perdite alla fine di ogni epoca per monitorare il progresso
        print(f"Epoca [{epoch+1}/{epochs}] Loss D: {lossD.item():.4f} | Loss G: {lossG.item():.4f}")

    # --- 4. VISUALIZZAZIONE DEI RISULTATI ---
    # Disabilitiamo il calcolo dei gradienti per l'inferenza (risparmia memoria e calcolo)
    with torch.no_grad():
        sample_noise = torch.randn(16, latent_dim, device=device) # Generiamo 16 vettori di rumore
        generated = netG(sample_noise).cpu()                      # Generiamo le immagini e le portiamo in CPU per mostrarle

    # Creiamo una griglia 4x4 per visualizzare le immagini generate
    plt.figure(figsize=(8, 8))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        # Mostriamo l'immagine (canale 0 perché è grayscale)
        plt.imshow(generated[i][0], cmap='gray')
        plt.axis('off') # Nascondiamo gli assi per pulizia
    plt.suptitle("Fashion MNIST: Campioni Generati (Architettura Dense)")
    plt.show()

# Commento finale: le immagini sono molto più sfocate e la diversità è diminuita
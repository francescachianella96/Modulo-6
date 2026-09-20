# Esercizio
# Addestrare una rete con VAE e con GAN prima a 10 epoche e poi a 20 e vedere come cambia il risultato

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# --- 1. PARAMETRI DI CONFIGURAZIONE ---
# Identificazione del dispositivo di calcolo: GPU (cuda) se disponibile, altrimenti CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Dimensione dello spazio latente: rappresenta il numero di variabili che sintetizzano le immagini.
latent_dim = 64
# Velocità di apprendimento degli ottimizzatori.
lr = 2e-4
# Numero di immagini analizzate contemporaneamente durante l'addestramento.
batch_size = 128
# Numero totale di cicli di addestramento su tutto il dataset.
epochs = 20

# Prepariamo le immagini del dataset: 
# transforms.ToTensor() converte le immagini in tensori PyTorch (valori tra 0 e 1).
# transforms.Normalize((0.5,), (0.5,)) sposta il range dei pixel a [-1, 1]. 
# Questo è fondamentale perché sia il VAE che la GAN terminano con la funzione Tanh.
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))
])

# Caricamento del dataset FashionMNIST: immagini in bianco e nero 28x28 di capi d'abbigliamento.
train_loader = DataLoader(
    datasets.FashionMNIST('.', train=True, download=True, transform=transform), 
    batch_size=batch_size, 
    shuffle=True
)

# --- 2. ARCHITETTURA VAE (Variational Autoencoder) ---
# Il VAE lavora cercando di comprimere l'input in una distribuzione statistica
# per poi ricostruirlo. L'obiettivo è la coerenza e la continuità dello spazio latente.
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        # L'Encoder riceve l'immagine (784 pixel) e produce due vettori:
        # la media (mu) e il logaritmo della varianza (logvar).
        # Questo serve per definire una distribuzione gaussiana nello spazio latente.
        self.enc = nn.Sequential(
            nn.Linear(784, 400), 
            nn.ReLU(), 
            nn.Linear(400, latent_dim * 2) # Produciamo 2 vettori da latent_dim ciascuno
        )
        # Il Decoder riceve un punto dello spazio latente e cerca di ricostruire l'immagine originale.
        # Caratteristica chiave: l'output passa per una Tanh per coincidere con i dati normalizzati [-1, 1].
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 400), 
            nn.ReLU(), 
            nn.Linear(400, 784), 
            nn.Tanh()
        )

    def forward(self, x):
        # Appiattiamo l'immagine 28x28 in un vettore di 784 elementi.
        h = self.enc(x.view(-1, 784))
        # Dividiamo l'output a metà per ottenere la media e la varianza separatamente.
        mu, logvar = h.chunk(2, dim=1)
        
        # Reparameterization Trick: per poter usare la backpropagation su una variabile casuale,
        # non campioniamo direttamente, ma usiamo la formula: z = mu + sigma * epsilon.
        # Epsilon è rumore casuale puro (gaussiana standard).
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # Restituiamo l'immagine ricostruita, la media e la varianza (necessarie per la loss function).
        return self.dec(z), mu, logvar

# --- 3. ARCHITETTURA GAN (Generative Adversarial Network) ---
# La GAN si basa sulla competizione tra due reti: il Generatore crea immagini dal nulla,
# il Discriminatore cerca di distinguere le immagini vere da quelle create dal Generatore.

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # Il Generatore mappa un vettore di rumore casuale (z) nello spazio delle immagini (784 pixel).
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256), 
            nn.LeakyReLU(0.2), # LeakyReLU evita il problema dei neuroni "morti" durante la competizione.
            nn.Linear(256, 512), 
            nn.LeakyReLU(0.2),
            nn.Linear(512, 784), 
            nn.Tanh() # Tanh obbliga i pixel tra [-1, 1], come le immagini originali.
        )
    def forward(self, z): 
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Il Discriminatore è un classificatore binario: risponde 1 (Vero) o 0 (Falso/Generato).
        self.model = nn.Sequential(
            nn.Linear(784, 512), 
            nn.LeakyReLU(0.2), 
            nn.Linear(512, 256), 
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1), 
            nn.Sigmoid() # Sigmoid trasforma l'output in una probabilità tra 0 e 1.
        )
    def forward(self, x): 
        # Riceve l'immagine appiattita e restituisce la probabilità di autenticità.
        return self.model(x.view(-1, 784))

# --- 4. VISUALIZZAZIONE E CONFRONTO ---
def plot_comparison(vae, gen, epoch):
    # Passiamo i modelli in modalità valutazione (disabilitando dropout se presente).
    vae.eval()
    gen.eval()
    with torch.no_grad():
        # Generiamo nuove immagini partendo da punti casuali dello spazio latente.
        z = torch.randn(8, latent_dim).to(device)
        
        # Per il VAE, decodifichiamo i punti casuali z per simulare la generazione di nuovi campioni.
        sample_vae = vae.dec(z) 
        # Per il Generatore della GAN, passiamo lo stesso rumore z.
        sample_gan = gen(z)
        
        # Uniamo le immagini in una griglia: prima riga VAE, seconda riga GAN.
        imgs = torch.cat([sample_vae.view(-1, 1, 28, 28), sample_gan.view(-1, 1, 28, 28)])
        grid = utils.make_grid(imgs, nrow=8, normalize=True)
        
        plt.figure(figsize=(10, 4))
        plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.title(f"Confronto Epoca {epoch}: VAE (Sopra) vs GAN (Sotto)")
        plt.axis('off')
        plt.show()

# --- 5. LOGICA DI ADDESTRAMENTO DOPPIA ---
# Inizializzazione dei modelli e spostamento sulla GPU/CPU.
vae = VAE().to(device)
gen = Generator().to(device)
disc = Discriminator().to(device)

# Utilizziamo Adam per tutti i modelli, un ottimizzatore robusto per queste architetture.
opt_vae = optim.Adam(vae.parameters(), lr=lr)
opt_g = optim.Adam(gen.parameters(), lr=lr)
opt_d = optim.Adam(disc.parameters(), lr=lr)

# Funzione di perdita per la GAN: Binary Cross Entropy Loss.
criterion = nn.BCELoss()

print("Inizio fase di addestramento...")
for epoch in range(1, epochs + 1):
    for i, (real_imgs, _) in enumerate(train_loader):
        real_imgs = real_imgs.to(device)
        b_size = real_imgs.size(0)
        
        # --- ADDESTRAMENTO VAE ---
        # L'obiettivo è minimizzare la distanza tra l'immagine originale e quella ricostruita.
        opt_vae.zero_grad()
        recon, mu, logvar = vae(real_imgs)
        # 1. Loss di Ricostruzione: misura l'errore tra i pixel originali e quelli prodotti dal decoder.
        recon_loss = F.mse_loss(recon, real_imgs.view(-1, 784), reduction='sum')
        # 2. Divergenza KL: forza la distribuzione latente a essere simile a una gaussiana standard.
        # Senza questa, il VAE memorizzerebbe semplicemente le immagini senza imparare a generarne nuove.
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss_vae = recon_loss + kld_loss
        loss_vae.backward()
        opt_vae.step()

        # --- ADDESTRAMENTO GAN ---
        # FASE 1: Addestramento del Discriminatore.
        opt_d.zero_grad()
        # Creiamo rumore e passiamolo al Generatore per ottenere immagini "false".
        z = torch.randn(b_size, latent_dim).to(device)
        fake_imgs = gen(z)
        
        # Etichette: 1 per le immagini vere, 0 per le immagini create dal generatore.
        label_real = torch.ones(b_size, 1).to(device)
        label_fake = torch.zeros(b_size, 1).to(device)
        
        # Il discriminatore deve imparare a dare 1 al reale e 0 al falso.
        loss_real = criterion(disc(real_imgs), label_real)
        loss_fake = criterion(disc(fake_imgs.detach()), label_fake) # Usiamo detach per non aggiornare il generatore.
        loss_d = loss_real + loss_fake
        loss_d.backward()
        opt_d.step()
        
        # FASE 2: Addestramento del Generatore.
        opt_g.zero_grad()
        # Il generatore "vince" se il discriminatore sbaglia e assegna 1 all'immagine falsa.
        loss_g = criterion(disc(fake_imgs), label_real)
        loss_g.backward()
        opt_g.step()

    # Visualizziamo i progressi a metà percorso e alla fine dell'addestramento.
    if epoch == 10 or epoch == 20:
        plot_comparison(vae, gen, epoch)
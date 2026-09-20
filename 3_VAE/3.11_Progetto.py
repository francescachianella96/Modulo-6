# 0 _ Importazione delle librerie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# --- FASE 1 - Data Strategy e Preprocessing ---

# Selezione del dispositivo di calcolo: GPU se disponibile, altrimenti CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configurazioni iperparametri
latent_dim = 32
epochs = 20
lr = 1e-3
batch_size = 64
img_size = 64
channels_img = 3

# Transformation Pipeline: ridimensiona a 64x64 e porta i pixel nel range [-1, 1]
data_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)]
    ),
])

# Caricamento del dataset Flowers102
train_dataset = datasets.Flowers102(
    root='./data', 
    split='train', 
    download=True, 
    transform=data_transforms
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size, 
    shuffle=True,
    num_workers=2 if torch.cuda.is_available() else 0,
    pin_memory=True if torch.cuda.is_available() else False
)

# --- FASE 2 - Architettura del Variational Autoencoder ---

class VAE(nn.Module):
    def __init__(self, channels_img=3, latent_dim=latent_dim):
        super().__init__()
        
        # Encoder: trasforma l'immagine (3 x 64 x 64) in una rappresentazione compressa
        self.encoder = nn.Sequential(
            nn.Conv2d(channels_img, 32, kernel_size=4, stride=2, padding=1),  # -> 32 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),            # -> 64 x 16 x 16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),           # -> 128 x 8 x 8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),          # -> 256 x 4 x 4
            nn.ReLU(),
            nn.Flatten(),                                                     # -> 256 * 4 * 4 = 4096
            nn.Linear(256 * 4 * 4, latent_dim * 2)                            # Output per mu e logvar
        )
        
        # Decoder: ricostruisce l'immagine (3 x 64 x 64) partendo dal vettore latente z
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),                                    # -> 256 x 4 x 4
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # -> 128 x 8 x 8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> 64 x 16 x 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # -> 32 x 32 x 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, channels_img, kernel_size=4, stride=2, padding=1), # -> 3 x 64 x 64
            nn.Tanh()  # Tanh per mappare i pixel nell'intervallo [-1, 1]
        )

    def encode(self, x):
        h = self.encoder(x)
        # Suddividiamo l'output in due parti uguali: mu e logvar
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar) # Calcolo della deviazione standard dal logaritmo della varianza
        eps = torch.randn_like(std)   # Campionamento di rumore casuale normale
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        # Restituiamo la ricostruzione, la media e la varianza (necessarie per la loss function)
        return self.decoder(z), mu, logvar

# --- FASE 3 - Definizione della Loss Function (ELBO) ---

def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    # 1. Perdita di Ricostruzione (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # 2. Divergenza KL
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Loss totale ponderata da beta
    total_loss = recon_loss + beta * kl_loss

    # Calcolo della loss totale
    return total_loss, recon_loss, kl_loss

# --- FASE 4 - Training Loop e KL Annealing ---

# Inizializzazione modello e ottimizzatore
model = VAE(channels_img=channels_img, latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Liste per il tracciamento delle metriche
history_loss = []
history_mse = []
history_kld = []

print("Inizio addestramento del VAE...")

for epoch in range(1, epochs + 1):
    model.train()
    total_epoch_loss = 0.0
    total_epoch_mse = 0.0
    total_epoch_kld = 0.0
    
    # Beta Annealing: incrementa beta da 0.0 a 1.0 nelle prime metà delle epoche
    beta = min(1.0, (epoch - 1) / (epochs * 0.5))
    
    pbar = tqdm(train_loader, desc=f"Epoca {epoch}/{epochs}")
    for data, _ in pbar:
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = model(data)
        loss, mse, kld = vae_loss_function(recon_batch, data, mu, logvar, beta)
        
        loss.backward()
        optimizer.step()
        
        # Accumulo per le metriche medie
        total_epoch_loss += loss.item()
        total_epoch_mse += mse.item()
        total_epoch_kld += kld.item()
        
        pbar.set_postfix(
            MSE=f"{mse.item() / len(data):.2f}", 
            KLD=f"{kld.item() / len(data):.2f}", 
            Beta=f"{beta:.2f}"
        )
    
    # Calcolo delle medie per campione nell'intera epoca
    num_samples = len(train_dataset)
    history_loss.append(total_epoch_loss / num_samples)
    history_mse.append(total_epoch_mse / num_samples)
    history_kld.append(total_epoch_kld / num_samples)

# --- Grafico delle Loss ---
plt.figure(figsize=(10, 4))
plt.plot(range(1, epochs + 1), history_loss, label='Total Loss (ELBO)')
plt.plot(range(1, epochs + 1), history_mse, label='Reconstruction Loss (MSE)')
plt.plot(range(1, epochs + 1), history_kld, label='KL Divergence')
plt.xlabel('Epoche')
plt.ylabel('Loss per Campione')
plt.title('Andamento del Training VAE e KL Annealing')
plt.legend()
plt.grid(True)
plt.show()

# --- FASE 5 - Analisi dello Spazio Latente e Generazione ---

# Funzione helper per riportare i pixel da [-1, 1] a [0, 1]
def denormalize(tensor):
    return (tensor + 1.0) / 2.0

model.eval()

# Ricostruzione (Originali vs Ricostruiti)
print("Visualizzazione delle ricostruzioni...")
with torch.no_grad():
    # Estraiamo un batch dal DataLoader
    real_imgs, _ = next(iter(train_loader))
    real_imgs = real_imgs[:8].to(device)  # Prendiamo le prime 8 immagini
    
    # Passaggio nel modello VAE
    recon_imgs, _, _ = model(real_imgs)
    
    # Portiamo in CPU e denormalizziamo
    real_imgs = denormalize(real_imgs).cpu()
    recon_imgs = denormalize(recon_imgs).cpu()

fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i in range(8):
    # Immagine Originale
    axes[0, i].imshow(real_imgs[i].permute(1, 2, 0).clamp(0, 1))
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_title("Originali", fontsize=12, fontweight='bold', loc='left')
        
    # Immagine Ricostruita
    axes[1, i].imshow(recon_imgs[i].permute(1, 2, 0).clamp(0, 1))
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_title("Ricostruiti", fontsize=12, fontweight='bold', loc='left')

plt.suptitle("Fase 5.1: Confronto Ricostruzione VAE", fontsize=14)
plt.tight_layout()
plt.show()

# Campionamento da Spazio Latente (Generazione Nuovi Fiori)
print("Generazione di nuovi campioni casuali...")
num_samples = 16
with torch.no_grad():
    # Campioniamo vettori z dalla distribuzione normale standard N(0, I)
    z_random = torch.randn(num_samples, latent_dim).to(device)
    
    # Generazione tramite il Decoder
    generated_imgs = model.decoder(z_random)
    generated_imgs = denormalize(generated_imgs).cpu()

fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i in range(num_samples):
    ax = axes[i // 8, i % 8]
    ax.imshow(generated_imgs[i].permute(1, 2, 0).clamp(0, 1))
    ax.axis('off')

plt.suptitle("Fase 5.2: Generazione di Nuovi Fiori da N(0, I)", fontsize=14)
plt.tight_layout()
plt.show()

# Interpolazione Latente (Latent Walk)
print("Esecuzione dell'interpolazione latente (Walk)...")
num_steps = 10

with torch.no_grad():
    # Selezioniamo due immagini distinte dal dataset
    data_iter = iter(train_loader)
    batch, _ = next(data_iter)
    img1, img2 = batch[0:1].to(device), batch[1:2].to(device)
    
    # Otteniamo i rispettivi vettori latenti z1 e z2 (usiamo la media mu)
    mu1, _ = model.encode(img1)
    mu2, _ = model.encode(img2)
    
    # Generiamo i vettori interpolati z_alpha = (1 - alpha) * z1 + alpha * z2
    alphas = torch.linspace(0, 1, steps=num_steps).to(device)
    interpolated_imgs = []
    
    for alpha in alphas:
        z_interp = (1 - alpha) * mu1 + alpha * mu2
        recon_interp = model.decoder(z_interp)
        interpolated_imgs.append(denormalize(recon_interp).cpu().squeeze(0))

fig, axes = plt.subplots(1, num_steps, figsize=(18, 2.5))
for i in range(num_steps):
    axes[i].imshow(interpolated_imgs[i].permute(1, 2, 0).clamp(0, 1))
    axes[i].axis('off')
    axes[i].set_title(f"α={alphas[i].item():.1f}", fontsize=9)

plt.suptitle("Fase 5.3: Interpolazione Latente Tra Due Fiori (Walk)", fontsize=14)
plt.tight_layout()
plt.show()
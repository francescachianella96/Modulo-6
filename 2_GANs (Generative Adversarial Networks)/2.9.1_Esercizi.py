import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# 1. IMPOSTAZIONI E IPERPARAMETRI
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 100
n_classes = 10
embed_size = 100  # Aumentato per matchare le dimensioni convoluzionali
image_size = 32
channels = 3

# Funzione per l'inizializzazione dei pesi (Standard DCGAN)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# ==============================================================================
# 2. GENERATORE CONDIZIONALE (CONVOLUZIONALE)
# ==============================================================================
class ConditionalGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Embedding(n_classes, latent_dim)
        
        # Input: (latent_dim + latent_dim) x 1 x 1
        self.model = nn.Sequential(
            # Output: 512 x 4 x 4
            nn.ConvTranspose2d(latent_dim * 2, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # Output: 256 x 8 x 8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # Output: 128 x 16 x 16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # Output: 3 x 32 x 32
            nn.ConvTranspose2d(128, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concateniamo il rumore e l'embedding della classe
        # noise: (batch, latent_dim) -> (batch, latent_dim, 1, 1)
        # label: (batch, latent_dim) -> (batch, latent_dim, 1, 1)
        c = self.label_embed(labels).view(-1, latent_dim, 1, 1)
        z = noise.view(-1, latent_dim, 1, 1)
        x = torch.cat([z, c], dim=1)
        return self.model(x)

# ==============================================================================
# 3. DISCRIMINATORE CONDIZIONALE (CONVOLUZIONALE)
# ==============================================================================
class ConditionalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Embedding(n_classes, image_size * image_size)
        
        self.model = nn.Sequential(
            # Input: (channels + 1) x 32 x 32
            nn.Conv2d(channels + 1, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output: 256 x 8 x 8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output: 512 x 4 x 4
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output finale: 1x1
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Condizionamento spaziale: creiamo un canale extra con l'etichetta
        c = self.label_embed(labels).view(-1, 1, image_size, image_size)
        x = torch.cat([img, c], dim=1)
        return self.model(x).view(-1, 1)

# ==============================================================================
# 4. INIZIALIZZAZIONE E PREPARAZIONE
# ==============================================================================
gen = ConditionalGenerator().to(device)
disc = ConditionalDiscriminator().to(device)

# Applichiamo l'inizializzazione pesi
gen.apply(weights_init)
disc.apply(weights_init)

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Spesso per le DCGAN si usa un learning rate leggermente più basso (es. 0.0002)
opt_gen = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# ==============================================================================
# 5. TRAINING LOOP (Identico alla logica precedente ma con dati 4D)
# ==============================================================================
def train_model(num_epochs=100):
    for epoch in range(num_epochs):
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), leave=True)
        for i, (real_imgs, labels) in progress_bar:
            batch_size = real_imgs.size(0)
            real_imgs, labels = real_imgs.to(device), labels.to(device)
            
            # --- Train Discriminator ---
            opt_disc.zero_grad()
            
            # Reali
            real_targets = torch.ones(batch_size, 1).to(device)
            output_real = disc(real_imgs, labels)
            loss_real = criterion(output_real, real_targets)
            
            # Falsi
            noise = torch.randn(batch_size, latent_dim).to(device)
            gen_labels = torch.randint(0, n_classes, (batch_size,)).to(device)
            fake_imgs = gen(noise, gen_labels)
            
            output_fake = disc(fake_imgs.detach(), gen_labels)
            fake_targets = torch.zeros(batch_size, 1).to(device)
            loss_fake = criterion(output_fake, fake_targets)
            
            loss_d = (loss_real + loss_fake) / 2
            loss_d.backward()
            opt_disc.step()
            
            # --- Train Generator ---
            opt_gen.zero_grad()
            output = disc(fake_imgs, gen_labels)
            loss_g = criterion(output, real_targets)
            loss_g.backward()
            opt_gen.step()
            
            progress_bar.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            progress_bar.set_postfix(D_loss=loss_d.item(), G_loss=loss_g.item())

# 

# ==============================================================================
# 6. VISUALIZZAZIONE (Adattata per output 4D)
# ==============================================================================
def visualizza_risultati():
    gen.eval()
    classes = ('aereo', 'auto', 'uccello', 'gatto', 'cervo', 'cane', 'rana', 'cavallo', 'nave', 'camion')
    noise = torch.randn(10, latent_dim).to(device)
    labels = torch.arange(10).to(device)
    
    with torch.no_grad():
        fake_imgs = gen(noise, labels).cpu()
    
    plt.figure(figsize=(15, 6))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        img = fake_imgs[i].permute(1, 2, 0) # Da (C,H,W) a (H,W,C)
        img = (img * 0.5) + 0.5 # Denormalizzazione
        plt.imshow(img.clamp(0, 1))
        plt.title(classes[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_model(num_epochs=20)
    visualizza_risultati()
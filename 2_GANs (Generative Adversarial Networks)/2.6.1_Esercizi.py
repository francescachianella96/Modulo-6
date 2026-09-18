# Esercizio
# Modifica l'architettura del Generatore aggiungendo un quinto layer convoluzionale per gestire un output a risoluzione 64x64
# come cambiano i parametri di padding e stride per mantenere la coerenza spaziale?
# Porva a caricare il dataset SVHN e calcola il tempo medio di un'epoca rispetto al MNIST, cosa noti nell'occupazione della VRAM?

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

# --- ARCHITETTURA (Definite fuori dal main per essere serializzabili) ---

def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x): return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x): return self.main(x).view(-1, 1)

# --- FUNZIONE PRINCIPALE ---

def main():
    # 1. SETUP E IPERPARAMETRI
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 100
    lr = 0.0002
    beta1 = 0.5 
    batch_size = 64
    epochs = 10

    print(f"Esecuzione su: {device}")

    # 2. DATASET
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = datasets.SVHN(root='.', split='train', download=True, transform=transform)
    # Su Windows num_workers > 0 richiede il blocco if __name__ == '__main__'
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

    # 3. MODELLI
    netG = Generator(latent_dim).to(device)
    netD = Discriminator().to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    optD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

    # 4. TRAINING LOOP
    print("Inizio Addestramento...")
    for epoch in range(epochs):
        epoch_start = time.time()
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), leave=True)
        
        for i, (real_imgs, _) in progress_bar:
            b_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            
            # Update D
            netD.zero_grad()
            label_real = torch.full((b_size, 1), 0.9, device=device)
            output_real = netD(real_imgs)
            lossD_real = criterion(output_real, label_real)
            
            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
            fake_imgs = netG(noise)
            label_fake = torch.zeros((b_size, 1), device=device)
            output_fake = netD(fake_imgs.detach())
            lossD_fake = criterion(output_fake, label_fake)
            
            lossD = lossD_real + lossD_fake
            lossD.backward()
            optD.step()

            # Update G
            netG.zero_grad()
            output_g = netD(fake_imgs)
            lossG = criterion(output_g, label_real)
            lossG.backward()
            optG.step()

            progress_bar.set_description(f"Epoch [{epoch+1}/{epochs}]")
            progress_bar.set_postfix(D_loss=lossD.item(), G_loss=lossG.item())

        # Fine Epoca
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
            utils.save_image(fake, f'svhn_64x64_epoch_{epoch+1}.png', normalize=True)
        
        print(f"Epoca {epoch+1} completata in {time.time()-epoch_start:.2f}s")

# PROTEZIONE FONDAMENTALE PER WINDOWS
if __name__ == '__main__':
    main()
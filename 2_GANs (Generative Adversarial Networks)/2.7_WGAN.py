import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------------------
# 1. CONFIGURAZIONE E IPERPARAMETRI
# -----------------------------------------------------------------------------------------
# Dispositivo: usa la GPU se disponibile per velocizzare il training delle reti convoluzionali.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Iperparametri del modello e del training
z_dim = 100           # Dimensione del vettore di rumore in ingresso
lr = 0.00005          # Learning rate basso per stabilità WGAN
batch_size = 64       # Dimensione del batch
n_critic = 5          # Numero di aggiornamenti del Critico per ogni aggiornamento del Generatore
clip_value = 0.01     # Valore di clipping per i pesi del Critico (vincolo di Lipschitz)
image_size = 32       # Dimensione immagini SVHN (32x32)
channels = 3          # Canali colore SVHN (RGB = 3)

# -----------------------------------------------------------------------------------------
# 2. PREPARAZIONE DEI DATI (SVHN)
# -----------------------------------------------------------------------------------------
# Definiamo le trasformazioni per il dataset SVHN.
# L'input deve essere ridimensionato a 32x32 (nativo SVHN) e normalizzato.
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalizzazione per 3 canali RGB
])

# Carichiamo il dataset SVHN (Street View House Numbers).
# Nota: SVHN usa 'split' invece di 'train' come argomento.
dataloader = DataLoader(
    datasets.SVHN(root='.', split='train', download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)

# -----------------------------------------------------------------------------------------
# 3. ARCHITETTURE CONVOLUZIONALI (DCGAN-like per WGAN)
# -----------------------------------------------------------------------------------------

class Generator(nn.Module):
    """
    Generatore Convoluzionale.
    Trasforma il vettore di rumore (z) in un'immagine RGB 32x32 tramite deconvoluzioni (ConvTranspose2d).
    """
    def __init__(self):
        super(Generator, self).__init__()
        # Input: N x z_dim x 1 x 1 (il vettore z viene rimodellato prima di entrare)
        self.model = nn.Sequential(
            # Layer 1: Da z_dim (100) a 256 canali, feature map 4x4
            # Kernel 4x4, Stride 1, Padding 0 -> Output: (N, 256, 4, 4)
            nn.ConvTranspose2d(z_dim, 256, kernel_size=4, stride=1, padding=0, bias=False),
            nn.ReLU(True),

            # Layer 2: Da 256 a 128 canali, espande a 8x8
            # Kernel 4x4, Stride 2, Padding 1 -> Output: (N, 128, 8, 8)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),

            # Layer 3: Da 128 a 64 canali, espande a 16x16
            # Kernel 4x4, Stride 2, Padding 1 -> Output: (N, 64, 16, 16)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),

            # Layer 4 (Output): Da 64 a 3 canali (RGB), espande a 32x32
            # Kernel 4x4, Stride 2, Padding 1 -> Output: (N, 3, 32, 32)
            nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh() # Output finale tra -1 e 1
        )

    def forward(self, z):
        # Per usare ConvTranspose2d, z deve avere 4 dimensioni: (Batch, z_dim, 1, 1)
        z = z.view(z.size(0), z.size(1), 1, 1) # Reshape
        return self.model(z)

class Critic(nn.Module):
    """
    Critico Convoluzionale.
    Prende un'immagine 32x32 RGB e restituisce un punteggio reale (scalar score).
    Non usa Sigmoide finale (WGAN).
    """
    def __init__(self):
        super(Critic, self).__init__()
        # Input: (N, 3, 32, 32)
        self.model = nn.Sequential(
            # Layer 1: Da 3 canali a 64 canali. Downsampling a 16x16.
            # Kernel 4, Stride 2, Padding 1.
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True), # LeakyReLU è standard nei discriminatori

            # Layer 2: Da 64 a 128 canali. Downsampling a 8x8.
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: Da 128 a 256 canali. Downsampling a 4x4.
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4 (Output): Collassa tutto in un singolo valore.
            # Input 4x4 -> Kernel 4x4 -> 1x1.
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0, bias=False)
            # Uscita: (N, 1, 1, 1). Nessuna attivazione (Linear).
        )

    def forward(self, x):
        out = self.model(x)
        # Ritorna un tensore piatto (N, 1) o scalare, rimuovendo le dimensioni spaziali 1x1
        return out.view(-1)

# -----------------------------------------------------------------------------------------
# 4. INIZIALIZZAZIONE
# -----------------------------------------------------------------------------------------
gen = Generator().to(device)
critic = Critic().to(device)

# Inizializzazione dei pesi (Best practice per DCGAN/WGAN)
def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

gen.apply(weights_init)
critic.apply(weights_init)

# Ottimizzatori RMSprop (Standard per WGAN con weight clipping)
opt_G = optim.RMSprop(gen.parameters(), lr=lr)
opt_C = optim.RMSprop(critic.parameters(), lr=lr)

# -----------------------------------------------------------------------------------------
# 5. TRAINING LOOP ASIMMETRICO
# -----------------------------------------------------------------------------------------
print("Inizio addestramento WGAN su SVHN...")

epochs = 20 

for epoch in range(epochs):
    for i, (real_imgs, _) in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        curr_batch_size = real_imgs.size(0)

        # === STEP 1: TRAINING DEL CRITICO ===
        # Addestriamo il critico più volte
        loss_C_val = 0
        for _ in range(n_critic):
            opt_C.zero_grad()
            
            # Generazione fake
            z = torch.randn(curr_batch_size, z_dim, device=device)
            fake_imgs = gen(z).detach() # Detach per non propagare gradienti al Gen
            
            # Loss WGAN: -(E[D(real)] - E[D(fake)])
            # Il critico deve dare punteggi alti ai veri e bassi ai falsi
            loss_C = -(torch.mean(critic(real_imgs)) - torch.mean(critic(fake_imgs)))
            
            loss_C.backward()
            opt_C.step()

            # Weight Clipping
            for p in critic.parameters():
                p.data.clamp_(-clip_value, clip_value)
            
            loss_C_val = loss_C.item()

        # === STEP 2: TRAINING DEL GENERATORE ===
        opt_G.zero_grad()
        
        # Generazione fake (questa volta tracciamo i gradienti)
        z = torch.randn(curr_batch_size, z_dim, device=device)
        gen_fake = gen(z)
        
        # Loss Generatore: -E[D(fake)]
        # Il generatore vuole che il critico dia punteggi alti (realistici) ai fake
        loss_G = -torch.mean(critic(gen_fake))
        
        loss_G.backward()
        opt_G.step()

        # Log ogni 100 batch
        if i % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}] Batch {i}/{len(dataloader)} \tW-Dist: {-loss_C_val:.4f}")

    print(f"Epoca {epoch} completata. Distanza Wasserstein finale: {-loss_C_val:.4f}")

# -----------------------------------------------------------------------------------------
# 6. VISUALIZZAZIONE RISULTATI
# -----------------------------------------------------------------------------------------
print("\nGenerazione campioni finali...")
gen.eval()

with torch.no_grad():
    # Input rumore
    noise = torch.randn(16, z_dim, device=device)
    # Generazione
    fake_samples = gen(noise).cpu()
    
    # Visualizzazione Griglia 4x4
    plt.figure(figsize=(8, 8))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        # Denormalizzazione: [-1, 1] -> [0, 1]
        img = fake_samples[i].permute(1, 2, 0) # Da (C, H, W) a (H, W, C) per matplotlib
        img = img * 0.5 + 0.5
        plt.imshow(img.numpy()) # .numpy() non strettamente necessario con tensori recenti ma sicuro
        plt.axis('off')
    
    plt.suptitle("SVHN: Campioni Generati con WGAN Convoluzionale", fontsize=16)
    plt.show()
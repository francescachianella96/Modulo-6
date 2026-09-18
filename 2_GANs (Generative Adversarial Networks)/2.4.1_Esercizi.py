# Esercizio 
# Prova ad addestrare un discrimintaore per 10 iterazioni pr ogni singola iterazione del Generatore e osserva se questo sbilanciamento provoca la scomparsa del gradiente 
# o se aiuta a prevenire il mode collapse. Visualizza la loss di entrambi e annota le differenze nella nitidezza delle immagini prodotte. 

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# --- 1. CONFIGURAZIONE E IPERPARAMETRI ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.0002
batch_size = 64
latent_dim = 100
image_dim = 784  # 28x28
n_critic = 10    # La richiesta della traccia: 10 iterazioni di D per 1 di G
epochs = 20

# Trasformazioni: Normalizzazione [-1, 1] per matchare la Tanh del Generatore
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root="dataset/", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- 2. ARCHITETTURE DENSE (MLP) ---
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, image_dim),
            nn.Tanh() 
        )
    def forward(self, x):
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(image_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid() 
        )
    def forward(self, x):
        return self.disc(x)

# Inizializzazione modelli e ottimizzatori
gen = Generator().to(device)
disc = Discriminator().to(device)
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# --- 3. CUSTOM TRAINING LOOP (10:1 Ratio) ---
loss_history_G = []
loss_history_D = []

print(f"Inizio addestramento su {device}...")

for epoch in range(epochs):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.view(-1, 784).to(device)
        curr_b_size = real.shape[0]

        # --- STEP 1: ADDESTRAMENTO INTENSIVO DEL DISCRIMINATORE (n_critic volte) ---
        # In questo loop D diventa molto esperto prima che G possa muovere un passo
        for _ in range(n_critic):
            noise = torch.randn(curr_b_size, latent_dim).to(device)
            fake = gen(noise)
            
            # Perdita sui reali
            disc_real = disc(real).view(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            
            # Perdita sui fake (.detach() è fondamentale per non toccare G)
            disc_fake = disc(fake.detach()).view(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            
            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

        # --- STEP 2: ADDESTRAMENTO DEL GENERATORE (1 volta) ---
        # G prova a rispondere al Discriminatore ormai molto "severo"
        noise = torch.randn(curr_b_size, latent_dim).to(device)
        fake = gen(noise)
        output = disc(fake).view(-1)
        
        # Trucco non-saturante: G vuole che D dica "Vero" (1)
        loss_gen = criterion(output, torch.ones_like(output))
        
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

    # Logging
    loss_history_D.append(loss_disc.item())
    loss_history_G.append(loss_gen.item())
    print(f"Epoca [{epoch+1}/{epochs}] | Loss D: {loss_disc:.4f} | Loss G: {loss_gen:.4f}")

# --- 4. VISUALIZZAZIONE RISULTATI ---
# Grafico delle Loss
plt.figure(figsize=(10, 5))
plt.plot(loss_history_D, label="Loss Discriminatore (10 steps)")
plt.plot(loss_history_G, label="Loss Generatore (1 step)")
plt.title("Competizione Sbilanciata 10:1")
plt.xlabel("Epoca")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Immagini generate
with torch.no_grad():
    sample_noise = torch.randn(16, latent_dim).to(device)
    samples = gen(sample_noise).cpu().view(-1, 28, 28)
    plt.figure(figsize=(8, 4))
    for i in range(16):
        plt.subplot(2, 8, i+1)
        plt.imshow(samples[i], cmap="gray")
        plt.axis("off")
    plt.suptitle("Campionamento Finale (D addestrato 10x)")
    plt.show()

# Commento finale su modello addestrato:
# Addestrando D ogni 10 iterazioni è possibile vedere molto bene come il modello soffra di Mode Collapse, in quanto le immagini generate non hanno diversità tra loro
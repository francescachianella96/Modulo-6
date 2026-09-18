import torch
import torch.nn as nn

# --- GENERATORE ---
# Input: Vettore latente (100,) -> Output: Immagine piatta (784,)
generator = nn.Sequential(
    nn.Linear(100, 256),
    nn.ReLU(),
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Linear(512, 784),
    nn.Tanh()  # Fondamentale: porta i pixel nel range [-1, 1]
)

# --- DISCRIMINATORE ---
# Input: Immagine piatta (784,) -> Output: Probabilità (1,)
discriminator = nn.Sequential(
    nn.Linear(784, 512),
    nn.LeakyReLU(0.2), # Standard nelle GAN per evitare gradienti nulli
    nn.Linear(512, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 1),
    nn.Sigmoid() # Output tra 0 e 1 (reale o falso)
)


# 1. Creiamo un vettore di rumore casuale (Latent Vector)
# Shape: (batch_size=1, latent_dim=100)
noise = torch.randn(1, 100)

# 2. Passaggio nel Generatore
fake_image = generator(noise)
print(f"Shape output Generatore: {fake_image.shape}") 
# Atteso: torch.Size([1, 784])

# 3. Passaggio nel Discriminatore
decision = discriminator(fake_image)
print(f"Shape output Discriminatore: {decision.shape}")
print(f"Valore di probabilità: {decision.item():.4f}")
# Atteso: torch.Size([1, 1])
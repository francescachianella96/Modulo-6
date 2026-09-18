# Import delle librerie necessarie
import torch
from torch import nn, optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np

# --- 1. CONFIGURAZIONE E IPERPARAMETRI ---
# Definiamo il dispositivo di calcolo (GPU se disponibile, altrimenti CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dimensione del vettore latente (input del generatore). E' il "seme" casuale da cui nasce l'immagine.
latent_dim = 100

# Iperparametri di addestramento
lr = 0.0002       # Learning rate (basso per stabilizzare le GAN)
beta1 = 0.5       # Parametro beta1 per l'ottimizzatore Adam (consigliato 0.5 dal paper DCGAN)
epochs = 10       # Numero di passaggi completi sul dataset

# Definizione delle trasformazioni per le immagini di input
transform = transforms.Compose([
    transforms.Resize(32),  # Ridimensioniamo tutto a 32x32 pixel
    transforms.ToTensor(),  # Convertiamo l'immagine in un tensore PyTorch (valori 0-1)
    # Normalizzazione: porta i valori da [0, 1] a [-1, 1].
    # Questo e' fondamentale perche' il Generatore usa Tanh come attivazione finale, producendo valori in [-1, 1].
    # La formula e': output = (input - mean) / std -> (input - 0.5) / 0.5
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Caricamento del dataset SVHN (Street View House Numbers)
train_data = datasets.SVHN(root='.', split='train', download=True, transform=transform)

# DataLoader gestisce il caricamento in batch e il mescolamento dei dati
dataloader = DataLoader(train_data, batch_size=128, shuffle=True)

# --- 2. DEFINIZIONE DELL'ARCHITETTURA ---

# Funzione per inizializzare i pesi della rete.
# Nelle GAN profonde (DCGAN), inizializzare i pesi con una distribuzione normale
# media 0 e deviazione standard 0.02 aiuta la convergenza (best practice dal paper).
def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# CLASSE GENERATORE
# Scopo: Trasformare un vettore di rumore casuale (latent_dim) in un'immagine RGB 32x32.
# Funzionamento: Usa ConvTranspose2d (convoluzioni trasposte o "deconvoluzioni") per fare upsampling.
# Parte da un tensore piccolo ma profondo (tanti canali) ed espande le dimensioni spaziali riducendo i canali.
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # Input: vettore latente Z (es. 100 x 1 x 1)
            # ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
            
            # Layer 1: Espansione iniziale
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), 
            nn.ReLU(True), # ReLU aiuta ad apprendere non-linearità complesse
            
            # Layer 2: Upsampling intermedio
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), 
            nn.ReLU(True),
            
            # Layer 3: Upsampling intermedio
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), 
            nn.ReLU(True),
            
            # Layer 4: Output finale verso l'immagine
            # Output: 3 canali (RGB), Kernel 4, Stride 2 -> Immagine 32x32
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            
            # Tanh comprime l'output in [-1, 1], corrispondente al range delle immagini normalizzate
            nn.Tanh()
        )
    def forward(self, x): return self.main(x)

# CLASSE DISCRIMINATORE
# Scopo: Classificatore binario che prende un'immagine (3x32x32) e restituisce una probabilità.
# Output: Probabilità che l'immagine sia REALE (vicino a 1) o FALSA/GENERATA (vicino a 0).
# Funzionamento: Usa Conv2d standard per ridurre la dimensione spaziale ed estrarre feature (downsampling).
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # Layer 1: Input immagine RGB 3x32x32
            nn.Conv2d(3, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=False), # LeakyReLU è preferito nelle GAN per evitare gradienti nulli
            
            # Layer 2: Estrazione feature
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), 
            nn.LeakyReLU(0.2, inplace=False),
            
            # Layer 3: Estrazione feature
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512), 
            nn.LeakyReLU(0.2, inplace=False),
            
            # Layer 4: Output singolo scalare (probabilità)
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            
            # Sigmoid schiaccia l'output in [0, 1] per interpretarlo come probabilità
            nn.Sigmoid()
        )
    # .view(-1, 1) assicura che l'output sia un vettore colonna (Batch_Size, 1) compatibile con la loss
    def forward(self, x): return self.main(x).view(-1, 1)

# Inizializzazione delle reti e applicazione dei pesi custom
netG = Generator().to(device).apply(weights_init)
netD = Discriminator().to(device).apply(weights_init)

# Funzione di costo: Binary Cross Entropy Loss
# E' la loss standard per problemi di classificazione binaria (Vero/Falso).
criterion = nn.BCELoss()

# Ottimizzatori Adam per entrambe le reti
optD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Vettore di rumore "fisso" per monitorare i progressi.
# Generando immagini sempre dallo stesso seme casuale, possiamo vedere oggettivamente
# come cambia la capacita' del generatore durante le epoche.
fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

# Creazione cartella per salvare i campioni generati
os.makedirs("samples", exist_ok=True)

# --- 3. CICLO DI ADDESTRAMENTO ---
print(f"Addestramento iniziato su {device}...")

for epoch in range(epochs):
    # Utilizziamo tqdm per avere una barra di avanzamento visiva durante l'epoca
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), leave=True)
    
    for i, (real_imgs, _) in progress_bar:
        # Dimensione del batch corrente (potrebbe essere minore di 128 nell'ultimo batch)
        b_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)
        
        # ==================================================================
        # FASE 1: ADDESTRAMENTO DEL DISCRIMINATORE (netD)
        # Obiettivo: Massimizzare log(D(x)) + log(1 - D(G(z)))
        # Deve riconoscere le immagini vere come VERE e quelle generate come FALSE.
        # ==================================================================
        
        netD.zero_grad() # Pulizia gradienti precedenti
        
        # 1.1 Addestramento con immagini REALI
        # Creiamo un tensore di etichette "Reali". Usiamo 0.9 invece di 1.0 (Label Smoothing)
        # per evitare che il discriminatore diventi troppo sicuro di se' troppo in fretta.
        label_real = torch.full((b_size, 1), 0.9, device=device)
        output_real = netD(real_imgs)
        lossD_real = criterion(output_real, label_real) # Calcolo errore sulle reali
        
        # 1.2 Addestramento con immagini FALSE (Generate)
        # Generiamo rumore casuale
        noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
        # Il generatore crea immagini false partendo dal rumore
        fake_imgs = netG(noise)
        # Etichetta "Falso" e' 0
        label_fake = torch.zeros((b_size, 1), device=device)
        # .detach() e' CRUCIALE qui: scollega il grafo computazionale del Generatore.
        # Quando calcoliamo la loss del Discriminatore, non vogliamo modificare i pesi del Generatore,
        # quindi "stacchiamo" le immagini false dalla storia che le ha generate.
        output_fake = netD(fake_imgs.detach()) 
        lossD_fake = criterion(output_fake, label_fake) # Calcolo errore sulle false
        
        # Aggiornamento pesi Discriminatore
        lossD = lossD_real + lossD_fake # Errore totale
        lossD.backward()
        optD.step()

        # ==================================================================
        # FASE 2: ADDESTRAMENTO DEL GENERATORE (netG)
        # Obiettivo: Massimizzare log(D(G(z)))
        # Il Generatore vuole ingannare il Discriminatore, facendogli credere che
        # le immagini generate siano vere (Label = 1).
        # ==================================================================
        
        netG.zero_grad()
        
        # Passiamo di nuovo le immagini false al discriminatore (senza detach()...)
        # perche' ora vogliamo che il gradiente fluisca all'indietro fino al generatore.
        output_g = netD(fake_imgs)
        
        # Il TRUCCO: calcoliamo la loss confrontando le immagini generate con l'etichetta REALE (1).
        # Se il discriminatore dice "Falso" (0), la loss sara' alta -> il Generatore impara a correggersi.
        lossG = criterion(output_g, label_real)
        
        # Aggiornamento pesi Generatore
        lossG.backward()
        optG.step()

        # Aggiornamento barra di avanzamento con le loss correnti
        progress_bar.set_description(f"Epoch [{epoch+1}/{epochs}]")
        progress_bar.set_postfix(D_loss=lossD.item(), G_loss=lossG.item())

    # --- FINE EPOCA ---
    # Generiamo immagini di esempio usando il rumore fisso definito all'inizio
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
        # Salvataggio su disco
        utils.save_image(fake, f'samples/epoch_{epoch+1}.png', normalize=True)

print("Addestramento completato!")

# --- 4. VISUALIZZAZIONE RISULTATI ---
# Confronto visivo finale: Immagini Reali vs Immagini Generate

# Preleviamo un singolo batch di dati reali dal dataloader
real_batch = next(iter(dataloader))

plt.figure(figsize=(15, 8))

# Pannello Sinistro: Immagini Reali
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Immagini Reali (Training Data)")
# make_grid: crea una griglia di immagini.
# .permute(1, 2, 0) o .transpose: Matplotlib richiede formato (Altezza, Larghezza, Canali),
# mentre PyTorch usa (Canali, Altezza, Larghezza). Invertiamo gli assi.
plt.imshow(np.transpose(utils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))

# Pannello Destro: Immagini Generate (dal modello finale)
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Immagini Generate (Generatore Finale)")
with torch.no_grad():
    fake = netG(fixed_noise).detach().cpu()
plt.imshow(np.transpose(utils.make_grid(fake, padding=2, normalize=True), (1, 2, 0)))

print("Mostrando i risultati a video...")
plt.show()
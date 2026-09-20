import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# --- CONFIGURAZIONE DEL SISTEMA ---
# Selezione del dispositivo di calcolo: GPU se disponibile, altrimenti CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dimensione del vettore latente (lo spazio compresso dove il modello impara le caratteristiche).
latent_dim = 32

# Parametro Beta per il disentanglement: controlla quanto il modello deve separare le singole 
# caratteristiche (es. forma, dimensione) nel vettore latente. Un valore > 1 forza una maggiore separazione.
beta = 4.0

# Numero di epoche per l'addestramento.
epochs = 20

# Caricamento del dataset FashionMNIST (immagini di abiti 28x28 in scala di grigi).
# Applichiamo una trasformazione ToTensor per convertire le immagini in tensori PyTorch (0.0 - 1.0).
transform = transforms.Compose([transforms.ToTensor()])
train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('.', train=True, download=True, transform=transform),
    batch_size=128, 
    shuffle=True
)

# --- ARCHITETTURA MODELLO: FashionBVAE ---
# Il Beta-VAE è composto da due parti principali: un Encoder che comprime i dati
# e un Decoder che cerca di ricostruirli partendo dalla rappresentazione compressa.
class FashionBVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        
        # L'Encoder trasforma l'immagine di input (28x28) in due parametri statistici:
        # la media (mu) e il logaritmo della varianza (logvar).
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1), # 14x14
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1), # 7x7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent_dim * 2)
        )
        
        # Il Decoder riceve un punto dello spazio latente e rigenera l'immagine 28x28.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1), # 28x28
            nn.Sigmoid()
        )

    def encode(self, x):
        """
        Passa l'immagine attraverso l'encoder per ottenere 
        la media e la varianza della distribuzione latente.
        """
        h = self.encoder(x)
        # Suddividiamo l'output in due parti uguali: mu e logvar
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Il cuore del VAE: campiona un punto 'z' dalla distribuzione definita da mu e logvar.
        Si usa il trucco della riparametrizzazione per permettere il passaggio del gradiente:
        z = mu + epsilon * deviazione_standard
        """
        std = torch.exp(0.5 * logvar) # Calcolo della deviazione standard dal logaritmo della varianza
        eps = torch.randn_like(std)   # Campionamento di rumore casuale normale
        return mu + eps * std

    def forward(self, x):
        """
        Definisce il flusso del dato: Input -> Encoder -> Reparametrizzazione -> Decoder -> Output.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        # Restituiamo la ricostruzione, la media e la varianza (necessarie per la loss function)
        return self.decoder(z), mu, logvar

# --- PROCEDURA DI ADDESTRAMENTO ---
# Inizializzazione del modello e dell'ottimizzatore Adam.
model = FashionBVAE(latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Inizio fase di addestramento Beta-VAE...")
model.train()
for epoch in range(epochs):
    total_loss = 0
    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Esecuzione del modello sull'input corrente
        recon, mu, logvar = model(data)
        
        # La Loss del BVAE è composta da due parti:
        # 1. Recon Loss: misura quanto l'immagine ricostruita è fedele all'originale.
        recon_loss = F.binary_cross_entropy(recon, data, reduction='sum')
        
        # 2. KLD Loss: misura quanto la distribuzione latente si allontana da una Normale standard.
        # Viene moltiplicata per 'beta' per controllare il grado di disentanglement.
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Somma delle due componenti per il calcolo del gradiente
        loss = recon_loss + beta * kld_loss
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoca {epoch+1}/{epochs} completata. Perdita media: {total_loss/len(train_loader.dataset):.4f}")

# --- ANALISI 1: LATENT WALK (Morphing tra immagini) ---
@torch.no_grad()
def latent_walk_visualization(model, z1, z2, steps=10):
    """
    Visualizza la transizione fluida tra due punti distinti nello spazio latente.
    Mostra come il decoder interpreta i punti intermedi tra z1 e z2.
    """
    model.eval()
    plt.figure(figsize=(15, 2))
    # Creiamo un'interpolazione lineare tra i due vettori latenti
    for i, t in enumerate(torch.linspace(0, 1, steps)):
        z = (1 - t) * z1 + t * z2
        img = model.decoder(z).squeeze().cpu() # Passaggio dal decoder all'immagine
        plt.subplot(1, steps, i + 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.suptitle("Analisi 1: Navigazione nello spazio latente (Interpolazione)")
    plt.show()

# --- ANALISI 2: ALGEBRA DELLE CARATTERISTICHE (Feature Arithmetic) ---
@torch.no_grad()
def feature_algebra_visualization(model, loader):
    """
    Dimostra la capacità del modello di isolare e manipolare attributi specifici.
    Calcoliamo un 'vettore di differenza' tra due categorie per applicarlo a nuovi dati.
    """
    model.eval()
    # Raccogliamo vettori latenti per "T-shirt" (Classe 0) e "Abiti lunghi" (Classe 3)
    z_tshirt, z_dress = [], []
    for data, labels in loader:
        data = data.to(device)
        mu, _ = model.encode(data)
        z_tshirt.append(mu[labels == 0])
        z_dress.append(mu[labels == 3])
        if len(z_tshirt) > 10 and len(z_dress) > 10: break
    
    # Il vettore differenza cattura l'essenza della "lunghezza/stile" di un abito rispetto a una maglietta
    v_feature = torch.cat(z_dress).mean(0) - torch.cat(z_tshirt).mean(0)
    
    # Prendiamo una maglietta originale
    z_orig = torch.cat(z_tshirt)[0:1]
    # Aggiungiamo il 'vettore abito' alla maglietta per trasformarla artificialmente
    z_mod = z_orig + 2.0 * v_feature 
    
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(model.decoder(z_orig).squeeze().cpu(), cmap='gray')
    plt.title("Originale (T-shirt)")
    plt.subplot(1, 2, 2)
    plt.imshow(model.decoder(z_mod).squeeze().cpu(), cmap='gray')
    plt.title("Trasformazione (+ Caratteristica Abito)")
    plt.show()

# --- ESPORTAZIONE: DEPLOYMENT (Quantizzazione e TorchScript) ---
def export_for_deployment(model):
    """
    Applica la quantizzazione dinamica a 8 bit ed esporta il decoder in formato TorchScript.
    Questo riduce la dimensione dei pesi (Linear) migliorando la velocità su CPU.
    """
    model.eval()
    
    # Spostiamo il modello su CPU per la quantizzazione (requisito standard)
    model_cpu = model.cpu()
    
    # Applichiamo la quantizzazione dinamica ai layer Linear per convertirli a 8-bit (int8)
    quantized_decoder = torch.quantization.quantize_dynamic(
        model_cpu.decoder, 
        {nn.Linear}, 
        dtype=torch.qint8
    )
    
    # Creiamo un input fittizio su CPU per tracciare il decoder quantizzato
    example_z = torch.randn(1, latent_dim)
    traced_decoder = torch.jit.trace(quantized_decoder, example_z)
    traced_script = torch.jit.script(traced_decoder)
    
    # Salvataggio del modello serializzato e quantizzato
    traced_script.save("decoder_fashion_quant.pt")
    print("\n--- Esportazione completata: decoder_fashion_quant.pt (8-bit) pronto per il deployment ---")

# --- AVVIO DELLE ANALISI ---
# Generiamo due punti casuali nello spazio latente per il morphing
z_start = torch.randn(1, latent_dim).to(device)
z_end = torch.randn(1, latent_dim).to(device)

# Esecuzione delle visualizzazioni e dei test di esportazione
latent_walk_visualization(model, z_start, z_end)
feature_algebra_visualization(model, train_loader)
export_for_deployment(model)
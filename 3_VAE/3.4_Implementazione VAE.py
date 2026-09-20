import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- CONFIGURAZIONE E DEFINIZIONE DEGLI IPERPARAMETRI ---
# Selezione del dispositivo di calcolo: utilizza la GPU (cuda) se disponibile per velocizzare l'esecuzione, altrimenti ripiega sulla CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dimensione dello spazio latente: rappresenta il numero di variabili utilizzate per la codifica compressa dell'immagine.
latent_dim = 32 

# Numero di epoche: quante volte l'intero dataset viene passato attraverso il modello durante l'addestramento.
epochs = 15

# Dimensione del batch: numero di immagini processate contemporaneamente prima di ogni aggiornamento dei pesi.
batch_size = 128

# Tasso di apprendimento (Learning Rate): controlla l'entità degli aggiornamenti apportati ai parametri del modello.
lr = 1e-3

# Caricamento e trasformazione dei dati: utilizziamo il dataset Fashion MNIST.
# L'operazione "ToTensor" converte le immagini in tensori di PyTorch scalandone i valori tra 0 e 1.
transform = transforms.Compose([transforms.ToTensor()])
train_loader = DataLoader(
    datasets.FashionMNIST('.', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)

# --- ARCHITETTURA DEL MODELLO VARIATIONAL AUTOENCODER (VAE) ---
# Il modello implementa un autoencoder variazionale, una rete neurale in grado di apprendere rappresentazioni 
# strutturate dei dati e generare nuovi campioni simili a quelli di addestramento.
class FashionVAE(nn.Module):
    def __init__(self):
        super().__init__()
        
        # L'Encoder analizza l'immagine di input (appiattita in un vettore da 784 elementi) 
        # e ne estrae le caratteristiche principali attraverso strati densi e funzioni di attivazione ReLU.
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU()
        )
        
        # Gli strati lineari "fc_mu" e "fc_logvar" trasformano l'output dell'encoder nei parametri di una distribuzione:
        # - mu: la media della distribuzione nello spazio latente.
        # - logvar: il logaritmo della varianza, che definisce l'incertezza della codifica.
        self.fc_mu = nn.Linear(200, latent_dim)
        self.fc_logvar = nn.Linear(200, latent_dim)
        
        # Il Decoder riceve un campione dallo spazio latente e cerca di ricostruire l'immagine originale.
        # L'ultimo strato applica una Sigmoid per garantire che i valori dei pixel siano compatibili con l'input iniziale.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        """
        Applica il metodo di riparametrizzazione.
        Per permettere il passaggio dei gradienti durante il campionamento, separiamo la componente stocastica 
        (epsilon) dai parametri della distribuzione (media e varianza).
        """
        # Calcolo della deviazione standard partendo dal logaritmo della varianza.
        std = torch.exp(0.5 * logvar)
        # Generazione di rumore casuale da una distribuzione normale standard.
        eps = torch.randn_like(std)
        # Trasformazione: calcola il vettore latente z mantenendo la differenziabilità.
        return mu + eps * std

    def forward(self, x):
        """
        Definisce come l'input 'x' attraversa il modello:
        1. L'Encoder codifica l'immagine in parametri statistici.
        2. Viene campionato un vettore 'z' dalla distribuzione risultante.
        3. Il Decoder ricostruisce l'immagine a partire da 'z'.
        """
        # Appiattimento dell'immagine (vettore riga) per l'elaborazione degli strati lineari.
        h = self.encoder(x.view(-1, 784))
        # Estrazione dei parametri mu e logvar.
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        # Campionamento del codice latente tramite riparametrizzazione.
        z = self.reparameterize(mu, logvar)
        # Ritorno del batch ricostruito e dei parametri necessari per il calcolo della loss.
        return self.decoder(z), mu, logvar

# --- FUNZIONE DI PERDITA (LOSS FUNCTION) ---
def loss_function(recon_x, x, mu, logvar, beta):
    """
    Calcola la funzione di costo totale del VAE composta da due termini fondamentali:
    1. Reconstruction Loss (BCE): penalizza il modello se l'output differisce dall'input.
    2. KL Divergence (KLD): obbliga lo spazio latente a seguire una distribuzione normale, 
       favorendo una struttura regolare e utile alla generazione di nuovi dati.
    """
    #Binary cross entropy tra l'immagine originale e la sua ricostruzione.
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    # Kullback-Leibler Divergence: quantifica quanto la distribuzione latente calcolata si scosta dalla normale standard.
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Somma pesata delle due loss: il valore beta controlla l'effetto della regolarizzazione.
    return BCE + beta * KLD, BCE, KLD

# --- PROCESSO DI ADDESTRAMENTO ---
# Inizializzazione della rete neurale e spostamento sulla memoria del dispositivo scelto (CPU/GPU).
model = FashionVAE().to(device)
# Ottimizzatore Adam per l'aggiornamento automatico dei pesi basato sulla discesa del gradiente stocastica.
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train() # Imposta il modello in modalità addestramento.
    train_loss = 0
    
    # Calcolo di Beta (KL Annealing): aumentiamo gradualmente l'importanza della regolarizzazione.
    # Nelle fasi iniziali il modello si concentra sulla qualità dell'immagine, successivamente sulla coerenza latente.
    beta = min(1.0, epoch / (epochs * 0.5))
    
    # Barra di avanzamento per monitorare l'andamento del loop corrente.
    pbar = tqdm(train_loader, desc=f"Epoca {epoch+1}")
    for i, (data, _) in enumerate(pbar):
        data = data.to(device)
        # Reset dei gradienti per evitare che si accumulino tra un'iterazione e l'altra.
        optimizer.zero_grad()
        
        # Passaggio dei dati nel modello per ottenere ricostruzioni e parametri latenti.
        recon_batch, mu, logvar = model(data)
        
        # Calcolo della perdita totale attraverso i componenti BCE e KLD.
        loss, bce, kld = loss_function(recon_batch, data, mu, logvar, beta)
        
        # Calcolo dei gradienti tramite procedura di backpropagation.
        loss.backward()
        # Aggiornamento dei parametri della rete neurale.
        optimizer.step()
        
        train_loss += loss.item()
        # Aggiornamento dei dati a schermo per il monitoraggio in tempo reale.
        pbar.set_postfix(BCE=bce.item()/len(data), KLD=kld.item()/len(data), Beta=beta)

# --- VALUTAZIONE E GENERAZIONE DI NUOVI DATI ---
# Disattivazione del calcolo dei gradienti e passaggio alla modalità valutazione.
model.eval()
with torch.no_grad():
    # Per generare nuovi dati, campioniamo casualmente 64 vettori dallo spazio latente.
    # Questi non derivano da immagini reali, ma sono semplici coordinate del 'mondo appreso'.
    sample = torch.randn(64, latent_dim).to(device)
    # Utilizziamo solo il Decoder per trasformare queste coordinate in immagini di capi di abbigliamento.
    sample = model.decoder(sample).cpu()
    
    # Salvataggio della griglia di immagini in formato PNG.
    utils.save_image(sample.view(64, 1, 28, 28), 'fashion_results.png')
    
    # Visualizzazione grafica dei capi generati casualmente.
    plt.imshow(utils.make_grid(sample.view(64, 1, 28, 28)).permute(1, 2, 0))
    plt.title("Immagini di abbigliamento generate casualmente tramite spazio latente")
    plt.axis('off')
    plt.show()
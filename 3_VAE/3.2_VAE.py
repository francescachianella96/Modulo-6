import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    Classe che implementa un Variational Autoencoder (VAE).
    A differenza di un Autoencoder classico, il VAE modella lo spazio latente 
    come una distribuzione di probabilità invece di un singolo punto vettoriale.
    """
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        
        # DEFINIZIONE DELL'ENCODER
        # Trasforma l'immagine di input in una rappresentazione intermedia (hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # PARAMETRI DELLA DISTRIBUZIONE LATENTE
        # L'output dell'encoder viene utilizzato per calcolare i due parametri 
        # fondamentali della distribuzione normale nello spazio latente:
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)       # Vettore delle medie (mu)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)   # Logaritmo della varianza (log_var)

        # DEFINIZIONE DEL DECODER
        # Prende un campione dallo spazio latente e tenta di ricostruire l'immagine originale
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid() # Funzione di attivazione per riportare i valori dei pixel nel range [0, 1]
        )

    def reparameterize(self, mu, logvar):
        """
        IL TRUCCO DELLA RIPARAMETRIZZAZIONE (Reparameterization Trick)
        Necessario per permettere la backpropagation attraverso un nodo stocastico.
        Invece di campionare direttamente da N(mu, std), campioniamo epsilon da N(0, 1)
        e calcoliamo z = mu + epsilon * std.
        """
        # Calcolo della deviazione standard partendo dal logaritmo della varianza
        std = torch.exp(0.5 * logvar)
        
        # Campionamento di un rumore casuale epsilon con la stessa dimensione di std
        eps = torch.randn_like(std)
        
        # Trasformazione: z ora è un campione derivabile rispetto a mu e logvar
        return mu + eps * std

    def forward(self, x):
        """
        Flusso dei dati attraverso il modello:
        1. L'immagine viene compressa dall'encoder.
        2. Vengono estratti mu e logvar.
        3. Si genera un punto z nello spazio latente tramite reparameterize.
        4. Il decoder ricostruisce l'immagine a partire da z.
        """
        # Fase 1: Compressione (appiattiamo l'immagine se necessario a 784 pixel)
        h = self.encoder(x.view(-1, 784))
        
        # Fase 2: Estrazione dei parametri distribuzionali
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Fase 3: Campionamento nello spazio latente
        z = self.reparameterize(mu, logvar)
        
        # Fase 4: Ricostruzione finale
        # Restituiamo l'immagine ricostruita insieme a mu e logvar per il calcolo della loss
        return self.decoder(z), mu, logvar

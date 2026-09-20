import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURAZIONE DEI PARAMETRI E DEL DISPOSITIVO
# =============================================================================
# Impostiamo il dispositivo di calcolo: GPU se disponibile, altrimenti CPU.
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )

# Iperparametri per il Vector Quantized Variational Autoencoder (VQ-VAE).
# num_embeddings: Dimensione del "Dizionario" (Codebook). Rappresenta quanti vettori discreti distinti
# il modello può scegliere per rappresentare l'informazione compressa.
num_embeddings = 128    

# embedding_dim: La dimensione di ciascun vettore nel codebook. Ogni "parola" del dizionario
# è un vettore di 32 numeri.
embedding_dim = 32     

# commitment_cost: Un fattore di ponderazione (beta) per la loss function.
# Serve a bilanciare quanto l'encoder deve "impegnarsi" a produrre vettori vicini a quelli del codebook.
commitment_cost = 0.25 

batch_size = 128       # Numero di immagini processate in parallelo
epochs = 10            # Numero di cicli di addestramento su tutto il dataset

# =============================================================================
# PREPARAZIONE DEI DATI (MNIST)
# =============================================================================
# Trasformazione base: converte le immagini in tensor PyTorch.
transform = transforms.Compose( [transforms.ToTensor()] )

# Caricamento del dataset MNIST. Viene scaricato automaticamente se non presente.
# Il DataLoader gestisce il caricamento in batch e il mescolamento (shuffle) dei dati.
train_loader = DataLoader(
    datasets.FashionMNIST( '.', train=True, download=True, transform=transform ),
    batch_size=batch_size,
    shuffle=True
)

# =============================================================================
# CLASSE: VECTOR QUANTIZER (IL CUORE DEL VQ-VAE)
# =============================================================================
# Questa classe gestisce la parte discreta del modello. Invece di passare vettori continui
# dall'encoder al decoder, li sostituisce con il vettore più simile presente in un "codebook" fisso.
class VectorQuantizer( nn.Module ):
    def __init__( self, num_embeddings, embedding_dim, commitment_cost ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # Il Codebook (o dizionario) è implementato come uno strato di Embedding.
        # Contiene 'num_embeddings' vettori, ognuno di dimensione 'embedding_dim'.
        # Questi sono i prototipi discreti che il modello imparerà ad usare.
        self.codebook = nn.Embedding( self.num_embeddings, self.embedding_dim )
        
        # Inizializzazione uniforme dei pesi del codebook per aiutare la convergenza iniziale.
        self.codebook.weight.data.uniform_( -1 / num_embeddings, 1 / num_embeddings )

    def forward( self, inputs ):
        # inputs ha forma: (Batch, Canali/Dim, Altezza, Larghezza).
        # Per confrontare i vettori con il codebook, dobbiamo spostare la dimensione dei canali (Embedding Dim)
        # all'ultima posizione.
        # Trasformazione: (B, C, H, W) -> (B, H, W, C)
        inputs = inputs.permute( 0, 2, 3, 1 ).contiguous()
        
        # Appiattiamo il tensore mantenendo solo la dimensione dell'embedding all'ultima posizione.
        # Ora abbiamo una lunga lista di vettori di dimensione 'embedding_dim'.
        # Forma: (Batch * H * W, Embedding Dim)
        flat_input = inputs.view( -1, self.embedding_dim )
        
        # CALCOLO DELLE DISTANZE
        # Calcoliamo la distanza euclidea quadrata tra ogni vettore di input appiattito
        # e ogni vettore nel codebook.
        # Questo ci dice quanto ogni parte dell'immagine codificata somiglia a parole del dizionario.
        # torch.cdist calcola la distanza p-norma tra ogni coppia di vettori.
        distances = torch.cdist( flat_input, self.codebook.weight )
        
        # TROVARE L'INDICE MIGLIORE
        # Per ogni vettore di input, troviamo l'indice del vettore del codebook che ha la distanza minore.
        # Questo è il processo di discretizzazione: assegniamo ogni input al suo "centroide" più vicino.
        encoding_indices = torch.argmin( distances, dim=1 ).unsqueeze( 1 )
        
        # QUANTIZZAZIONE
        # Usiamo gli indici trovati per estrarre i vettori corrispondenti dal codebook.
        # 'quantized' contiene ora i vettori del dizionario invece dei vettori originali dell'encoder.
        # Riportiamo la forma a quella dell'input: (Batch, H, W, C)
        quantized = self.codebook( encoding_indices ).view( inputs.shape )
        
        # --- CALCOLO DELLE LOSS SPECIFICHE DEL VQ-VAE ---
        # Il VQ-VAE utilizza un approccio particolare per l'addestramento perché coinvolge 
        # componenti discrete (il codebook) e continue (encoder/decoder).
        # Dobbiamo gestire attentamente chi aggiorna chi.
        
        # 1. Codebook Loss (VQ Loss)
        # OBIETTIVO: Addestrare il Codebook.
        # Vogliamo che i vettori del codebook ('quantized') si spostino verso i vettori prodotti dall'encoder ('inputs').
        # In pratica, stiamo dicendo al dizionario: "I dati reali sono qui, spostati più vicino a loro".
        # TECNICA: Usiamo 'inputs.detach()' perché in questa fase consideriamo l'output dell'encoder come 
        # un bersaglio fisso (Ground Truth). Non vogliamo che questa loss modifichi i pesi dell'encoder, 
        # ma SOLO i pesi del codebook.
        # Gradient Flow: Codebook <--- (SI) --- Loss --- (NO/Bloccato) ---> Encoder
        vq_loss = F.mse_loss( quantized, inputs.detach() )
        
        # 2. Commitment Loss
        # OBIETTIVO: Addestrare l'Encoder a "impegnarsi" (commit).
        # Vogliamo che l'encoder produca vettori che siano già vicini ai vettori esistenti nel codebook.
        # Senza questa loss, l'encoder potrebbe produrre vettori che fluttuano selvaggiamente, rendendo 
        # il codebook instabile e la quantizzazione imprecisa (errore grande tra input e quantized).
        # TECNICA: Usiamo 'quantized.detach()' perché qui consideriamo il codebook come fisso.
        # Vogliamo spostare gli 'inputs' (e quindi aggiornare l'encoder) verso il codebook, non viceversa.
        # Gradient Flow: Codebook <--- (NO/Bloccato) --- Loss --- (SI) ---> Encoder
        commitment_loss = F.mse_loss( quantized.detach(), inputs )
        
        # La loss totale restituita da questo modulo è la somma pesata.
        # vq_loss aggiorna SOLO il codebook.
        # commitment_loss aggiorna SOLO l'encoder.
        loss = vq_loss + self.commitment_cost * commitment_loss
        
        # --- STRAIGHT-THROUGH ESTIMATOR (STE) ---
        # Questo è il passaggio cruciale per la backpropagation.
        # L'operazione di quantizzazione (argmin) non ha gradiente (gradiente zero quasi ovunque).
        # Se usassimo direttamente 'quantized', il gradiente si fermerebbe qui.
        #
        # Trucco: nel forward pass usiamo 'quantized', ma nel backward pass approssimiamo il gradiente
        # come se avessimo passato 'inputs'.
        #
        # Formula: quantized = inputs + (quantized - inputs).detach()
        #
        # Spiegazione matematica:
        # - Forward: inputs + (quantized - inputs) = quantized. (Corretto, usiamo i valori quantizzati).
        # - Backward: il termine (quantized - inputs).detach() non ha gradiente.
        #             Quindi d(quantized)/d(inputs) = d(inputs)/d(inputs) = 1.
        # Il gradiente fluisce direttamente dal decoder all'encoder copiando i gradienti da 'quantized' a 'inputs'.
        quantized = inputs + ( quantized - inputs ).detach()
        
        # Riportiamo le dimensioni all'ordine originale per i layer convoluzionali successivi (Decoder).
        # (B, H, W, C) -> (B, C, H, W)
        return quantized.permute( 0, 3, 1, 2 ).contiguous(), loss

# =============================================================================
# MODELLO COMPLETO: VQ-VAE
# =============================================================================
class VQVAE( nn.Module ):
    def __init__( self ):
        super().__init__()
        
        # ENCODER
        # Riduce l'immagine di input (28x28) in una rappresentazione latente più piccola.
        # Utilizza strati convoluzionali con stride 2 per dimezzare le dimensioni spaziali due volte.
        # Input: (Batch, 1, 28, 28) -> Output intermedio: (Batch, embedding_dim, 7, 7)
        self.encoder = nn.Sequential(
            nn.Conv2d( 1, 32, 4, stride=2, padding=1 ),  # Riduce a 14x14
            nn.ReLU(),
            nn.Conv2d( 32, 64, 4, stride=2, padding=1 ), # Riduce a 7x7
            nn.ReLU(),
            nn.Conv2d( 64, embedding_dim, 1 )           # Proietta nello spazio degli embedding (16 canali)
        )
        
        # ISTANZA DEL VECTOR QUANTIZER
        # Gestisce la discretizzazione dello spazio latente.
        self.vq = VectorQuantizer( num_embeddings, embedding_dim, commitment_cost )
        
        # DECODER
        # Ricostruisce l'immagine originale partendo dalla rappresentazione quantizzata.
        # Usa ConvTranspose2d per aumentare le dimensioni spaziali (upsampling).
        # Input: (Batch, embedding_dim, 7, 7) -> Output: (Batch, 1, 28, 28)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d( embedding_dim, 64, 4, stride=2, padding=1 ), # Aumenta a 14x14
            nn.ReLU(),
            nn.ConvTranspose2d( 64, 32, 4, stride=2, padding=1 ),            # Aumenta a 28x28
            nn.ReLU(),
            nn.Conv2d( 32, 1, 1 ),                                           # Riduce i canali a 1 (scala di grigi)
            nn.Sigmoid()                                                     # Output tra 0 e 1
        )

    def forward( self, x ):
        # 1. Encoding: otteniamo lo spazio latente continuo z_e
        z_e = self.encoder( x )
        
        # 2. Vector Quantization: otteniamo lo spazio latente discreto z_q e la loss di quantizzazione
        z_q, vq_loss = self.vq( z_e )
        
        # 3. Decoding: ricostruiamo l'immagine partendo dai vettori discreti
        x_recon = self.decoder( z_q )
        
        return x_recon, vq_loss

# =============================================================================
# CICLO DI ADDESTRAMENTO
# =============================================================================
# Istanziamo il modello e lo spostiamo sul device (GPU/CPU)
model = VQVAE().to( device )

# Ottimizzatore Adam per aggiornare i pesi di Encoder, Decoder e Codebook
optimizer = optim.Adam( model.parameters(), lr=1e-3 )

print( "Inizio addestramento VQ-VAE..." )

for epoch in range( epochs ):
    model.train() # Imposta il modello in modalità training
    total_loss = 0
    
    for data, _ in train_loader:
        # Spostiamo i dati sul device corretto
        data = data.to( device )
        
        # Azzeriamo i gradienti accumulati nel passo precedente
        optimizer.zero_grad()
        
        # Forward pass: otteniamo la ricostruzione e la loss del quantizzatore
        recon, vq_loss = model( data )
        
        # Calcoliamo la Reconstruction Loss (errore di ricostruzione)
        # Misura quanto l'immagine ricostruita è simile all'originale
        recon_loss = F.mse_loss( recon, data )
        
        # La Loss totale è la somma dell'errore di ricostruzione e dell'errore di quantizzazione
        loss = recon_loss + vq_loss
        
        # Backward pass: calcolo dei gradienti
        loss.backward()
        
        # Aggiornamento dei pesi
        optimizer.step()
        
        total_loss += loss.item()
        
    # Stampa della loss media per l'epoca corrente
    print( f"Epoca {epoch+1}/{epochs}, Loss Media: {total_loss / len(train_loader):.4f}" )

# =============================================================================
# VISUALIZZAZIONE DEI RISULTATI
# =============================================================================
model.eval() # Imposta il modello in modalità valutazione (disabilita dropout, batchnorm training, ecc.)
with torch.no_grad():
    # Prendiamo un batch di dati di test
    data, _ = next( iter( train_loader ) )
    data = data.to( device )
    
    # Passiamo i dati nel modello per ottenere le ricostruzioni
    recon, _ = model( data )
    
    # Visualizziamo Originale vs Ricostruito
    fig, ax = plt.subplots( 2, 1, figsize=(10, 4) )
    
    # Immagini originali
    ax[0].imshow( utils.make_grid( data[:8] ).permute( 1, 2, 0 ).cpu() )
    ax[0].set_title( "Immagini Originali" )
    ax[0].axis( "off" )
    
    # Immagini ricostruite dal codebook discreto
    ax[1].imshow( utils.make_grid( recon[:8] ).permute( 1, 2, 0 ).cpu() )
    ax[1].set_title( "Immagini Ricostruite (da Codebook Discreto)" )
    ax[1].axis( "off" )
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# VISUALIZZAZIONE DEL CODEBOOK (GLI "ATOMI" APPRESI)
# =============================================================================


def visualize_index_map(model, loader):
    model.eval()
    with torch.no_grad():
        # Prendiamo un batch di immagini reali
        images, _ = next(iter(loader))
        images = images.to(device)
        
        # Passiamo dall'encoder
        z_e = model.encoder(images)
        
        # Calcoliamo le distanze dal codebook per trovare gli indici
        # (Riprendiamo la logica interna del VectorQuantizer)
        z_e_flattened = z_e.permute(0, 2, 3, 1).contiguous().view(-1, embedding_dim)
        distances = torch.cdist(z_e_flattened, model.vq.codebook.weight)
        indices = torch.argmin(distances, dim=1).view(images.shape[0], 7, 7)
        
        # Visualizzazione
        num_examples = 5
        plt.figure(figsize=(15, 5))
        for i in range(num_examples):
            # Immagine Originale
            plt.subplot(3, num_examples, i + 1)
            plt.imshow(images[i].cpu().squeeze(), cmap='gray')
            plt.axis('off')
            if i == 0: plt.title("Originale")

            # Mappa degli Indici (Colorata)
            # Ogni quadratino colorato corrisponde a un "atomo" del codebook
            plt.subplot(3, num_examples, i + num_examples + 1)
            plt.imshow(indices[i].cpu().numpy(), cmap='tab20') 
            plt.axis('off')
            if i == 0: plt.title("Mappa Indici (7x7)")

            # Ricostruzione Finale
            plt.subplot(3, num_examples, i + 2*num_examples + 1)
            recon, _ = model(images[i].unsqueeze(0))
            plt.imshow(recon.cpu().squeeze(), cmap='gray')
            plt.axis('off')
            if i == 0: plt.title("Ricostruzione")
            
        plt.tight_layout()
        plt.show()

# Eseguiamo la nuova visualizzazione
visualize_index_map(model, train_loader)
import torch
import torch.nn as nn
import torch.nn.functional as F

# SISTEMA DI PROGRESSIVE GROWING GAN (PGGAN) SEMPLIFICATO
# --------------------------------------------------------------------
# 1. Crescita Progressiva: Si inizia generando immagini a bassa risoluzione (es. 4x4)
#    e si aggiungono man mano livelli per aumentare i dettagli (8x8, 16x16, ecc.).
# 2. Alpha Fading: Quando si aggiunge un nuovo livello, non lo si usa subito al 100%.
#    Si introduce gradualmente tramite un parametro 'alpha' che varia da 0 a 1.
#    Questo permette alla rete di rimanere stabile durante la transizione.

class SimpleBlock(nn.Module):
    # Questa classe rappresenta un singolo blocco di costruzione della rete.
    # Ogni volta che la risoluzione aumenta, viene aggiunto un nuovo SimpleBlock.
    #
    # Ruolo:
    # Elabora le caratteristiche (features) dell'immagine alla risoluzione corrente.
    # Riceve in ingresso un tensore e restituisce un tensore processato.
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Primo strato convoluzionale: mantiene la dimensione spaziale (padding=1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Secondo strato convoluzionale: raffina ulteriormente le features
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Funzione di attivazione LeakyReLU: permette un piccolo passaggio di gradienti negativi
        # per evitare neuroni morti.
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        # Passaggio 1: Convoluzione -> Attivazione
        x = self.conv1(x)
        x = self.activation(x)
        
        # Passaggio 2: Convoluzione -> Attivazione
        x = self.conv2(x)
        x = self.activation(x)
        
        # Restituisce le feature map elaborate
        return x

class SimpleGenerator(nn.Module):
    # Questa classe è il Generatore che crea l'immagine partendo da rumore casuale.
    # Gestisce la crescita della risoluzione e il fading tra i livelli.
    #
    # Interazione con SimpleBlock:
    # Il generatore mantiene una lista di SimpleBlock. Ogni volta che chiamiamo il metodo
    # 'grow()', un nuovo SimpleBlock viene aggiunto alla lista per gestire la nuova risoluzione.
    
    def __init__(self, z_dim=512):
        super().__init__()
        self.z_dim = z_dim
        
        # Layer Iniziale:
        # Trasforma il vettore latente (rumore) in un volume 4x4 iniziale.
        # Immaginalo come la "bozza" grezza dell'immagine.
        self.input_layer = nn.Linear(z_dim, 512 * 4 * 4)
        
        # Lista dei blocchi convoluzionali (nn.ModuleList)
        # Iniziamo con un solo blocco che gestisce la risoluzione 4x4.
        # Man mano che la rete cresce, aggiungeremo blocchi qui (8x8, 16x16...).
        self.blocks = nn.ModuleList([
            SimpleBlock(512, 512)
        ])
        
        # Lista dei convertitori 'ToRGB'
        # Ogni livello di risoluzione ha bisogno del proprio strato per convertire
        # le feature interne nel formato immagine (3 canali RGB).
        # Questo ci permette di vedere l'immagine a qualsiasi stadio della crescita.
        self.to_rgb_layers = nn.ModuleList([
            nn.Conv2d(512, 3, kernel_size=1)
        ])
        
        # Stato attuale del generatore
        self.current_res_step = 0  # 0 indica che siamo alla risoluzione base (4x4)
        self.alpha = 1.0           # Parametro per il fading: 1.0 = transizione completa
                                   # Se alpha e 0.5, siamo a meta transizione tra vecchia e nuova risoluzione.

    def grow(self):
        # Aumenta la risoluzione del generatore aggiungendo nuovi strati.
        
        # Incrementa il contatore dello step di risoluzione
        self.current_res_step += 1
        
        # Calcola la nuova dimensione in pixel solo per stampa informativa
        new_res = 4 * (2 ** self.current_res_step)
        print(f"\n[SISTEMA] Crescita attivata. Nuova risoluzione target: {new_res}x{new_res}")
        
        # Logica per ridurre i canali man mano che la risoluzione aumenta
        # (per risparmiare memoria). Esempio: 512 -> 256 -> 128
        prev_channels = 512
        new_channels = 512 # In questo esempio semplificato non riduciamo i canali per chiarezza
        
        # 1. Creiamo il nuovo blocco convoluzionale per la nuova risoluzione
        new_block = SimpleBlock(prev_channels, new_channels)
        
        # 2. Lo aggiungiamo alla lista dei blocchi attivi
        self.blocks.append(new_block)
        
        # 3. Creiamo e aggiungiamo il nuovo convertitore ToRGB per questa risoluzione
        new_to_rgb = nn.Conv2d(new_channels, 3, kernel_size=1)
        self.to_rgb_layers.append(new_to_rgb)
        
        # IMPORTANTE: Resettiamo alpha a 0.0
        # Questo significa che inizialmente il nuovo livello non contribuirà all'immagine.
        # L'immagine sarà formata solo dall'upscaling del livello precedente.
        # Aumenteremo alpha gradualmente nel loop di training (qui simulato).
        self.alpha = 0.0

    def forward(self, z):
        # Questo metodo definisce il flusso dei dati attraverso la rete.
        # è qui che avviene la magia del Progressive Growing e dell'Alpha Fading.
        
        # Fase 1: Proiezione da Vettore Latente a Tensore 4x4
        # Trasformiamo il vettore di input z (dimensione [Batch, 512]) in un tensore piatto
        out = self.input_layer(z)
        # Rimodelliamo il tensore piatto in un cubo [Batch, 512, 4, 4]
        out = out.view(-1, 512, 4, 4)
        
        # Fase 2: Passaggio attraverso i blocchi 'stabili'
        # Passiamo attraverso tutti i blocchi tranne l'ultimo (che e quello in fase di fading).
        # Se siamo allo step 0, questo loop non viene eseguito.
        for i in range(self.current_res_step):
            # Elaborazione del blocco i-esimo
            out = self.blocks[i](out)
            # Upsampling: raddoppiamo la dimensione spaziale (es. 4x4 -> 8x8)
            # Usiamo 'nearest' neighbor che è semplice ed efficace per i primi test
            out = F.interpolate(out, scale_factor=2, mode='nearest')
            
        # A questo punto 'out' ha la dimensione della nuova risoluzione, ma contiene
        # ancora le informazioni elaborate dai livelli precedenti (upsamplati).
        
        # Fase 3: Gestione del Fading (Alpha Blending)
        
        # CASO A: Siamo alla risoluzione base (step 0)
        if self.current_res_step == 0:
            # Passiamo nel primo blocco ed esce subito l'RGB
            out = self.blocks[0](out)
            return self.to_rgb_layers[0](out)
        
        # CASO B: Siamo in una fase di crescita (step > 0)
        
        # Calcoliamo il percorso "VECCHIO" (Old Branch):
        # Convertiamo subito in RGB l'output proveniente dall'upsampling dei livelli precedenti.
        # Questo rappresenta l'immagine a bassa risoluzione, semplicemente ingrandita.
        old_rgb = self.to_rgb_layers[self.current_res_step - 1](out)
        
        # Calcoliamo il percorso "NUOVO" (New Branch):
        # Facciamo passare i dati attraverso il NUOVO blocco appena aggiunto.
        # Questo blocco imparera i dettagli fini della nuova risoluzione.
        new_features = self.blocks[self.current_res_step](out)
        # Convertiamo queste nuove feature in RGB
        new_rgb = self.to_rgb_layers[self.current_res_step](new_features)
        
        # Mix finale (Alpha Blending):
        # Combiniamo le due immagini in base al valore di alpha.
        # Se alpha=0: L'immagine e 100% old_rgb (solo upsample, il nuovo blocco e ignorato).
        # Se alpha=1: L'immagine e 100% new_rgb (il nuovo blocco e pienamente operativo).
        # Se alpha=0.5: L'immagine e una media tra le due.
        final_rgb = (1 - self.alpha) * old_rgb + self.alpha * new_rgb
        
        return final_rgb

# --- BLOCCO DI SIMULAZIONE ---
# Questa sezione simula l'uso della rete senza un vero addestramento,
# solo per mostrare come cambiano le dimensioni e come funziona il fading.

def test_pggan_flow():
    # Fissiamo il seme per riproducibilità
    torch.manual_seed(42)
    z_dim = 512
    batch_size = 1
    
    # Istanziamo il generatore
    print("Inizializzazione Generatore...")
    generator = SimpleGenerator(z_dim)
    
    # Creiamo un vettore di rumore casuale (input del generatore)
    z = torch.randn(batch_size, z_dim)
    
    # --- STEP 1: Risoluzione Base 4x4 ---
    print("\n--- FASE 1: Risoluzione Base (4x4) ---")
    img = generator(z)
    print(f"Output shape: {img.shape}")
    print(f"Valore Alpha: {generator.alpha} (1.0 significa stabile)")
    
    # --- STEP 2: Crescita a 8x8 ---
    print("\n--- FASE 2: Crescita a 8x8 (Inizio Fading) ---")
    # Chiamiamo grow() per aggiungere i layer 8x8
    generator.grow() 
    
    # Simuliamo il passaggio progressivo di alpha da 0 a 1
    steps_di_fading = 5
    for i in range(steps_di_fading + 1):
        # Aggiorniamo manualmente alpha (in un training vero lo farebbe il loop di training)
        generator.alpha = i / steps_di_fading
        
        # Generiamo l'immagine
        img = generator(z)
        
        print(f"Fading Step {i}: Alpha {generator.alpha:.1f} -> Dimensione Immagine {img.shape}")
        
        if i == 0:
            print("   (NOTA: Con Alpha=0, l'immagine e solo un upsample del 4x4 precedente)")
        elif i == steps_di_fading:
            print("   (NOTA: Con Alpha=1, l'immagine usa pienamente i nuovi strati 8x8)")
            
    # --- STEP 3: Crescita a 16x16 ---
    print("\n--- FASE 3: Ulteriore Crescita a 16x16 ---")
    generator.grow()
    
    # Impostiamo alpha a meta per vedere cosa succede
    generator.alpha = 0.5 
    img = generator(z)
    print(f"Test a meta fading (Alpha 0.5) -> Dimensione Immagine {img.shape}")
    print("   (L'immagine risultante e una media pesata tra l'upsample 8x8 e il nuovo 16x16)")

if __name__ == "__main__":
    test_pggan_flow()
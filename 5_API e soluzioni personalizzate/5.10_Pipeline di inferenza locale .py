import asyncio
import time
import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import FastAPI
from contextlib import asynccontextmanager
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- SEZIONE 1: CONFIGURAZIONE E STRUTTURE DATI ---
# Questo intervallo definisce quanto tempo il server aspetta per accumulare richieste
# prima di processarle. Un valore di 1.0s e' molto alto per scopi dimostrativi,
# per permetterci di inviare piu' richieste manualmente e vederle gestite insieme.
BATCH_INTERVAL = 1.0 

# model_storage funge da memoria condivisa globale per mantenere il modello e gli scaler
# accessibili in tutta l'applicazione (pattern simile a un Singleton).
model_storage = {}

# request_queue e' il canale di comunicazione tra l'endpoint API (produttore)
# e il loop di processamento batch (consumatore). E' thread-safe e async-safe.
request_queue = asyncio.Queue()

# --- SEZIONE 2: DEFINIZIONE DEL MODELLO NEURALE ---
# Definiamo una semplice rete neurale con PyTorch per classificare i fiori Iris.
# Questa classe estende nn.Module, la classe base per tutti i moduli di reti neurali in PyTorch.
class SimpleIrisNet(nn.Module):
    def __init__(self):
        super(SimpleIrisNet, self).__init__()
        # Primo livello lineare: prende 4 feature in input e produce 16 output nascosti
        self.fc1 = nn.Linear(4, 16)
        # Secondo livello lineare: prende i 16 output nascosti e produce 3 output finali (le 3 classi)
        self.fc2 = nn.Linear(16, 3) 
        # Funzione di attivazione ReLU per introdurre non linearita'
        self.relu = nn.ReLU()
        # Softmax per convertire gli output in probabilita' (somma = 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Il metodo forward definisce come i dati attraversano la rete
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

# --- SEZIONE 3: IL MOTORE DI BATCHING (CONSUMER) ---
# Questa funzione asincrona gira in background per tutta la vita dell'applicazione.
# Il suo compito e' prelevare richieste dalla coda, raggrupparle ed eseguire l'inferenza.
async def batch_processing_loop():
    print(f"Batch processor avviato. Accumulo richieste per {BATCH_INTERVAL}s...")
    
    while True:
        # FASE A: Attesa (Windowing)
        # Il processore dorme per l'intervallo definito. Questo crea la "finestra temporale"
        # in cui le richieste in arrivo si accumulano nella coda.
        await asyncio.sleep(BATCH_INTERVAL)
        
        # FASE B: Raccolta (Fetching)
        batch_items = []
        # Svuotiamo la coda di tutte le richieste arrivate durante l'attesa.
        # Utilizziamo get_nowait() perche' non vogliamo bloccarci se la coda e' vuota.
        while not request_queue.empty():
            try:
                item = request_queue.get_nowait()
                batch_items.append(item)
            except asyncio.QueueEmpty:
                break
            
        # Se non ci sono richieste, torniamo all'inizio del loop (attesa)
        if not batch_items:
            continue 
            
        print(f"Processando batch di {len(batch_items)} richieste...")
        
        # FASE C: Preparazione del Batch
        # Separiamo i dati di input (features) dai futuri (le "promesse" di risposta).
        inputs = [item['features'] for item in batch_items]
        futures = [item['future'] for item in batch_items]
        
        # FASE D: Inferenza Batch (Il cuore dell'ottimizzazione)
        # Invece di passare un singolo esempio alla volta, creiamo un tensore che contiene
        # TUTTI gli esempi. Questo permette a PyTorch di parallelizzare i calcoli matriciali.
        # E' qui che otteniamo il guadagno di performance rispetto all'elaborazione sequenziale.
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        
        with torch.no_grad(): # Disabilitiamo il calcolo dei gradienti per risparmiare memoria
            outputs = model_storage['model'](inputs_tensor)
            # Otteniamo l'indice della classe con la probabilita' piu' alta per ogni esempio
            _, predicted_indices = torch.max(outputs, 1)
            
        # FASE E: Risoluzione delle Richieste
        results = predicted_indices.tolist()
        class_names = model_storage['class_names']
        
        # Ora dobbiamo restituire il risultato corretto a ogni richiesta in attesa.
        # Iteriamo su coppie (future, risultato) e impostiamo il risultato nel future.
        # Questo sblocchera' l'endpoint API che sta aspettando quella specifica risposta.
        for future, idx in zip(futures, results):
            if not future.done():
                future.set_result(class_names[idx])


# --- SEZIONE 4: GESTIONE DEL CICLO DI VITA (LIFESPAN) ---
# Questa funzione gestisce le operazioni di avvio e spegnimento dell'applicazione.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- AVVIO (STARTUP) ---
    print("Caricamento e training modello Iris...")
    
    # 1. Caricamento Dati
    iris = load_iris()
    X, y = iris.data, iris.target
    model_storage['class_names'] = iris.target_names
    
    # 2. Preprocessing (Standardizzazione)
    # Le reti neurali lavorano meglio con dati scalati (media 0, varianza 1).
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model_storage['scaler'] = scaler 
    
    # 3. Addestramento "al volo"
    # Creiamo istanza del modello, funzione di perdita e ottimizzatore
    model = SimpleIrisNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    
    # Convertiamo i dati numpy in tensori PyTorch
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    
    # Loop di addestramento (150 epoche)
    for _ in range(150): 
        optimizer.zero_grad() # Reset dei gradienti
        output = model(X_t)   # Forward pass
        loss = criterion(output, y_t) # Calcolo errore
        loss.backward()       # Backpropagation
        optimizer.step()      # Aggiornamento pesi
        
    model_storage['model'] = model
    print("Modello addestrato e pronto.")
    
    # 4. Avvio Task in Background
    # Avviamo il loop infinito del batch processor come task asyncio indipendente.
    batch_task = asyncio.create_task(batch_processing_loop())
    
    yield # Punto di sospensione: qui il server inizia a servire le richieste HTTP
    
    # --- SPEGNIMENTO (SHUTDOWN) ---
    # Quando il server si ferma, cancelliamo il task di background per chiudere pulito.
    batch_task.cancel()
    print("Server spento.")

# Inizializzazione dell'app FastAPI con il gestore lifespan
app = FastAPI(lifespan=lifespan)

# --- SEZIONE 5: ENDPOINT API (PRODUCER) ---
@app.post("/predict")
async def predict(features: list[float]):
    """
    Endpoint per la predizione della classe del fiore Iris.
    Non esegue la predizione immediatamente, ma delega al batch processor.
    """
    start_time = time.time()
    
    # 1. Validazione Input
    if len(features) != 4:
        return {"error": "Servono esattamente 4 features (sepal length, sepal width, petal length, petal width)"}
        
    # 2. Preprocessing Speculare al Training
    # Dobbiamo scalare i dati di input esattamente come abbiamo scalato i dati di training.
    scaler = model_storage['scaler']
    features_scaled = scaler.transform([features])[0]
    
    # 3. Creazione del Future
    # Otteniamo il loop di eventi corrente e creiamo un oggetto Future.
    # Un Future e' un oggetto che rappresentera' un risultato che non esiste ancora.
    loop = asyncio.get_running_loop()
    future = loop.create_future() 
    
    # 4. Inserimento in Coda
    # Inseriamo nella coda un dizionario contenente i dati da elaborare e il future associato.
    # Il batch processor trovera' questo pacchetto e lo elaborera'.
    await request_queue.put({
        'features': features_scaled,
        'future': future
    })
    
    # 5. Attesa Asincrona
    # Qui avviene la magia: await future sospende l'esecuzione di QUESTA specifica richiesta.
    # FastAPI e il loop asyncio sono liberi di gestire altre richieste mentre questa aspetta.
    # L'esecuzione riprendera' solo quando il batch processor chiamera' future.set_result().
    prediction = await future
    
    # 6. Risposta al Client
    elapsed = time.time() - start_time
    return {
        "class": prediction,
        "latency": f"{elapsed:.2f}s",
        "mode": "Batched (1s window)"
    }

# Nota per l'esecuzione:
# Eseguire questo file con: uvicorn 'esempio_60:app' --reload
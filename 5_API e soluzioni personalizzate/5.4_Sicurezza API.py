import asyncio
import time
import torch
import torch.nn as nn
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel, Field, ConfigDict

# --- SEZIONE 1: DEFINIZIONE E ADDESTRAMENTO DEL MODELLO DI INTELLIGENZA ARTIFICIALE ---

# Inizializziamo i dati per l'addestramento (X_train rappresenta la dimensione, y_train il prezzo)
# La relazione che il modello dovrà imparare è: Prezzo = 2 * Dimensione + 1
X_train = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=torch.float32)
y_train = torch.tensor([[3.0], [5.0], [7.0], [9.0], [11.0]], dtype=torch.float32)

# Definiamo un modello lineare semplice con un ingresso (dimensione) e un'uscita (prezzo)
model = nn.Linear(1, 1)

# Utilizziamo l'algoritmo SGD (Stochastic Gradient Descent) per aggiornare i parametri del modello
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Definiamo la funzione di errore (MSE - Mean Squared Error) per misurare la precisione delle predizioni
criterion = nn.MSELoss()

# Avviamo il ciclo di addestramento per 1000 iterazioni (epoche)
for _ in range(1000):
    # Il modello effettua una previsione basata sui dati di input correnti
    prediction = model(X_train)
    # Calcoliamo la differenza tra la previsione e il risultato reale
    loss = criterion(prediction, y_train)
    # Calcoliamo i gradienti (direzione dell'errore) tramite backpropagation
    loss.backward()
    # Aggiorniamo i pesi del modello per ridurre l'errore nel prossimo passaggio
    optimizer.step()
    # Resettiamo i gradienti per evitare che si accumulino nell'iterazione successiva
    optimizer.zero_grad()

# Impostiamo il modello in modalità valutazione dopo l'addestramento
model.eval()

# --- SEZIONE 2: ARCHITETTURA DI SICUREZZA E CONTROLLO ACCESSI ---

# Elenco delle chiavi API autorizzate ad accedere al servizio
API_KEYS = {"admin-key", "research-token-01"}

# Database in memoria per gestire il controllo della frequenza delle richieste (Rate Limiting)
# La struttura è: {chiave_api: [lista di timestamp delle ultime chiamate]}
rate_limit_db = {} 

def security_layer(x_api_key: str = Header(None)):
    """
    Funzione di middleware per gestire sicurezza e rate limiting.
    Viene richiamata automaticamente da FastAPI prima di eseguire l'endpoint.
    """
    # Verifica la presenza e la validità della chiave API nell'header della richiesta
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Accesso Negato: API Key non valida")
    
    # Gestione del Rate Limiting
    now = time.time()
    # Recuperiamo lo storico delle chiamate per l'utente corrente
    user_history = rate_limit_db.get(x_api_key, [])
    
    # Manteniamo nello storico solo le chiamate effettuate negli ultimi 20 secondi
    user_history = [t for t in user_history if now - t < 20]
    
    # Se l'utente ha effettuato 5 o più richieste in 20 secondi, blocchiamo la chiamata
    if len(user_history) >= 5:
        raise HTTPException(status_code=429, detail="Troppe richieste: rallenta!")
    
    # Registriamo il timestamp della richiesta corrente nello storico dell'utente
    user_history.append(now)
    rate_limit_db[x_api_key] = user_history
    
    # Restituiamo la chiave validata (verrà passata all'endpoint come parametro)
    return x_api_key

def sanitize(comment: str):
    """
    Funzione di controllo per prevenire l'inserimento di contenuti non sicuri o comandi vietati.
    """
    blacklist = ["DROP TABLE", "DELETE", "ADMIN_ACCESS"]
    # Convertiamo il commento in maiuscolo per un controllo case-insensitive
    if any(word in comment.upper() for word in blacklist):
        raise HTTPException(status_code=403, detail="Contenuto non autorizzato rilevato")
    return comment

# --- SEZIONE 3: DEFINIZIONE DEL SERVIZIO API (FASTAPI) ---

# Inizializziamo l'applicazione principale
app = FastAPI(title="Secure AI Deploy")

class PriceRequest(BaseModel):
    """
    Schema dei dati in ingresso (Pydantic). 
    Definisce la struttura della richiesta JSON e applica validazioni automatiche.
    """
    # La dimensione in metri quadri deve essere un numero positivo (gt=0)
    square_meters: float = Field(..., gt=0)
    # Nota opzionale dell'utente con valore predefinito
    user_note: str = Field(default="Nessuna nota")
    
    # Configurazione per documentare l'API con un esempio pratico
    model_config = ConfigDict(json_schema_extra={"example": {"square_meters": 75.0, "user_note": "Stima urgente"}})

@app.post("/estimate")
async def get_estimate(data: PriceRequest, key: str = Depends(security_layer)):
    """
    Punto di accesso principale per ottenere la stima dei prezzi.
    
    Interazioni:
    1. Il parametro 'key' utilizza 'Depends(security_layer)' per attivare l'autenticazione.
    2. I dati 'data' vengono validati automaticamente tramite la classe PriceRequest.
    3. Il corpo della funzione esegue sanitizzazione, simulazione di rete e inferenza AI.
    """
    
    # Pulizia manuale del campo 'user_note' per prevenire codice malevolo
    safe_note = sanitize(data.user_note)
    
    # Simulazione di un leggero ritardo dovuto alla comunicazione di rete (0.4 secondi)
    await asyncio.sleep(0.4)
    
    # Convertiamo il dato in ingresso in un formato compatibile con PyTorch (Tensor)
    input_tensor = torch.tensor([[data.square_meters]], dtype=torch.float32)
    
    # Eseguiamo l'inferenza (predizione) senza calcolare i gradienti per ottimizzare le prestazioni
    with torch.no_grad():
        # Passiamo il dato al modello e otteniamo il valore numerico finale
        prediction = model(input_tensor).item()
    
    # Restituiamo il risultato finale formattato come JSON
    return {
        "estimated_price_k": round(prediction, 2),
        "sanitized_note": safe_note,
        "auth_status": "Verified"
    }

# Istruzioni per l'esecuzione del server:
# uvicorn nome_file:app --reload
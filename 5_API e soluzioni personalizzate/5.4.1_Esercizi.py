import asyncio
import time
import torch
import torch.nn as nn
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel, Field, ConfigDict

# --- SEZIONE 1: MODELLO AI (PYTORCH) ---
# Modello lineare: Prezzo = 2 * Dimensione + 1
X_train = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=torch.float32)
y_train = torch.tensor([[3.0], [5.0], [7.0], [9.0], [11.0]], dtype=torch.float32)

model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for _ in range(1000):
    prediction = model(X_train)
    loss = criterion(prediction, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

model.eval()

# --- SEZIONE 2: MIDDLEWARE DI SICUREZZA ---

API_KEYS = {"studente-ai-2026", "chiave-test"}
rate_limit_db = {} 

def security_layer(x_api_key: str = Header(None, alias="X-API-Key")):
    """
    Punto 1 & 2: Verifica Header 'X-API-Key' e Rate Limiting (3 req / 10 sec).
    """
    # 1. Verifica Header
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="X-API-Key non valida o mancante")
    
    # 2. Logica Rate Limiting
    now = time.time()
    user_history = rate_limit_db.get(x_api_key, [])
    
    # Filtriamo solo i timestamp degli ultimi 10 secondi
    user_history = [t for t in user_history if now - t < 10]
    
    # Blocco dopo 3 richieste
    if len(user_history) >= 3:
        raise HTTPException(
            status_code=429, 
            detail="Rate limit raggiunto: massimo 3 richieste ogni 10 secondi."
        )
    
    user_history.append(now)
    rate_limit_db[x_api_key] = user_history
    return x_api_key

def sanitize_prompt(text: str):
    """
    Punto 3: Difesa da Prompt Injection.
    """
    trigger = "ignora le istruzioni"
    if trigger in text.lower():
        # Punto 4: Errore specifico per injection rilevata
        raise HTTPException(
            status_code=400, 
            detail="Attacco di Prompt Injection rilevato. Richiesta bloccata."
        )
    return text

# --- SEZIONE 3: SERVIZIO API ---

app = FastAPI(title="Middleware Security Exercise 2026")

class PredictRequest(BaseModel):
    square_meters: float = Field(..., gt=0)
    prompt_context: str = Field(default="Analisi standard")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "square_meters": 50.0, 
            "prompt_context": "Valutazione appartamento centro città"
        }
    })

@app.post("/predict")
async def predict(data: PredictRequest, key: str = Depends(security_layer)):
    """
    Endpoint finale che integra tutti i livelli di sicurezza.
    """
    # Validazione semantica (Sanitizzazione)
    safe_context = sanitize_prompt(data.prompt_context)
    
    # Simulazione asincrona
    await asyncio.sleep(0.5)
    
    # Inferenza PyTorch
    input_tensor = torch.tensor([[data.square_meters]], dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_tensor).item()
    
    return {
        "result": round(prediction, 2),
        "context_status": "verified_safe",
        "remaining_calls": 3 - len(rate_limit_db.get(key, []))
    }
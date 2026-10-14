import asyncio
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# --- 1. PREPARAZIONE DATI E TRAINING (PYTORCH) ---

# Carichiamo il dataset reale
data = fetch_california_housing()
# Selezioniamo 4 colonne: MedInc, HouseAge, AveRooms, AveOccup
X_raw = data.data[:, :4]
y_raw = data.target.reshape(-1, 1)

# Split e Normalizzazione (Best Practice: calcolata solo sul train)
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

mu = torch.tensor(X_train.mean(axis=0), dtype=torch.float32)
sigma = torch.tensor(X_train.std(axis=0), dtype=torch.float32)

def normalize(x):
    return (x - mu) / sigma

# Conversione in Tensori
X_train_t = normalize(torch.tensor(X_train, dtype=torch.float32))
y_train_t = torch.tensor(y_train, dtype=torch.float32)

# Definizione Modello
class HousingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    def forward(self, x):
        return self.regressor(x)

model = HousingNet()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training Loop veloce
print("Inizio addestramento modello...")
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
print(f"Addestramento completato. Loss finale: {loss.item():.4f}")

model.eval() # Modalità inferenza

# --- 2. DEFINIZIONE API (FASTAPI + PYDANTIC) ---

class HousingInput(BaseModel):
    MedInc: float = Field(..., gt=0, description="Reddito medio")
    HouseAge: float = Field(..., gt=0, description="Età della casa")
    AveRooms: float = Field(..., gt=0, description="Stanze medie")
    AveOccup: float = Field(..., gt=0, description="Occupazione media")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "MedInc": 3.5,
                "HouseAge": 15.0,
                "AveRooms": 5.0,
                "AveOccup": 3.0
            }
        }
    )

app = FastAPI(title="California Housing: Training & Deploy 2026")

@app.post("/predict")
async def predict(input_data: HousingInput):
    """
    Esegue l'inferenza asincrona sul modello appena addestrato.
    """
    # Simulazione latenza I/O
    await asyncio.sleep(0.5)
    
    try:
        # Preprocessing: Conversione e Normalizzazione
        raw_values = [input_data.MedInc, input_data.HouseAge, input_data.AveRooms, input_data.AveOccup]
        features = torch.tensor([raw_values], dtype=torch.float32)
        normalized_features = normalize(features)

        # Inferenza (PyTorch puro)
        with torch.no_grad():
            prediction = model(normalized_features)
        
        return {
            "prediction_100k_usd": float(prediction.item()),
            "status": "success",
            "model_info": "PyTorch Neural Network (Simple Regressor)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Comandi per l'esecuzione:
# uvicorn nome_file:app --reload
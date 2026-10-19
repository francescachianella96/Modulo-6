from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd
import io

from backend.services import (
    generate_analysis_code, 
    execute_generated_code, 
    train_vae_model, 
    generate_synthetic_samples
)

app = FastAPI(title="VAE & LLM Analyst API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variabili globali per memorizzare lo stato in memoria
real_df: pd.DataFrame = pd.DataFrame()
synthetic_df: pd.DataFrame = pd.DataFrame()
trained_model = None

class QueryRequest(BaseModel):
    prompt: str

class TrainRequest(BaseModel):
    epochs: int = 50
    latent_dim: int = 4
    batch_size: int = 64

class GenerateRequest(BaseModel):
    num_samples: int = 100

@app.get("/")
def read_root():
    return {"message": "API Backend attiva e pronta!"}

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    global real_df
    try:
        contents = await file.read()
        real_df = pd.read_csv(io.BytesIO(contents))
        return {
            "status": "success",
            "message": f"File '{file.filename}' caricato con successo.",
            "shape": real_df.shape,
            "columns": real_df.columns.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Errore nella lettura del file: {str(e)}")

@app.post("/train")
async def train_vae(params: TrainRequest):
    global real_df, trained_model
    if real_df.empty:
        raise HTTPException(status_code=400, detail="Carica prima un file CSV tramite /upload.")
    
    try:
        trained_model, metrics = train_vae_model(
            df=real_df, 
            epochs=params.epochs, 
            latent_dim=params.latent_dim, 
            batch_size=params.batch_size
        )
        return {
            "status": "success",
            "message": "Addestramento completato con successo.",
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante il training: {str(e)}")

@app.post("/generate")
async def generate_data(params: GenerateRequest):
    global trained_model, real_df, synthetic_df
    if trained_model is None or real_df.empty:
        raise HTTPException(status_code=400, detail="Devi prima addestrare il modello con /train.")
    
    try:
        synthetic_df = generate_synthetic_samples(trained_model, real_df, params.num_samples)
        
        stream = io.StringIO()
        synthetic_df.to_csv(stream, index=False)
        response = StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv"
        )
        response.headers["Content-Disposition"] = "attachment; filename=synthetic_dataset.csv"
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante la generazione: {str(e)}")

@app.post("/api/analyze")
async def analyze(request: QueryRequest):
    global synthetic_df, real_df
    target_df = synthetic_df if not synthetic_df.empty else real_df
    if target_df.empty:
        raise HTTPException(status_code=400, detail="Nessun dataset presente per l'analisi.")
        
    try:
        code = generate_analysis_code(request.prompt)
        img_base64 = execute_generated_code(code, target_df)
        return {
            "status": "success",
            "generated_code": code,
            "image_base64": img_base64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
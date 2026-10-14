import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np

# --- CONFIGURAZIONE DEL MODELLO DI MACHINE LEARNING ---
# In questa fase prepariamo il modello che verrà utilizzato dall'API.
# Il dataset Iris contiene misurazioni di fiori (lunghezza/larghezza sepali e petali).
iris = load_iris()
X, y = iris.data, iris.target

# Inizializziamo e addestriamo un modello di regressione logistica per la classificazione.
# Questo oggetto (model) sarà condiviso tra le diverse richieste gestite da FastAPI.
model = LogisticRegression(max_iter=200).fit(X, y)

# --- DEFINIZIONE DELLO SCHEMA DEI DATI (PYDANTIC) ---
# La classe IrisSpecs eredita da BaseModel di Pydantic.
# Il suo compito è validare i dati JSON in ingresso prima che raggiungano la logica di predizione.
class IrisSpecs(BaseModel):
    # Ogni campo è definito con un tipo (float) e vincoli di validazione (gt=0: deve essere maggiore di zero).
    # Field permette di aggiungere metadati e regole di validazione automatica.
    sepal_length: float = Field(..., gt=0, description="Lunghezza del sepalo in centimetri")
    sepal_width: float = Field(..., gt=0, description="Larghezza del sepalo in centimetri")
    petal_length: float = Field(..., gt=0, description="Lunghezza del petalo in centimetri")
    petal_width: float = Field(..., gt=0, description="Larghezza del petalo in centimetri")

    # Utilizziamo model_config (ConfigDict) per definire i comportamenti del modello.
    # json_schema_extra fornisce un esempio pratico per la documentazione interattiva.
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }
    )

# --- INIZIALIZZAZIONE DELL'APPLICAZIONE WEB (FASTAPI) ---
# L'oggetto 'app' è il punto centrale che coordina il routing e la gestione delle richieste.
app = FastAPI(title="Iris Predictor API")

@app.post("/predict_species")
async def predict_species(specs: IrisSpecs):
    """
    Endpoint per la predizione della specie.
    Riceve un oggetto IrisSpecs, esegue una simulazione di asincronia e restituisce il risultato.
    L'interazione avviene tramite il passaggio dell'oggetto 'specs' validato da Pydantic.
    """
    
    # Simuliamo un'operazione asincrona (ad esempio un log su database o una chiamata esterna).
    # L'uso di 'await' garantisce che il server non rimanga bloccato durante l'attesa,
    # permettendo di gestire altre richieste simultaneamente.
    await asyncio.sleep(0.8) 
    
    try:
        # Trasformiamo l'oggetto Pydantic in un array NumPy compatibile con Scikit-Learn.
        # Estraiamo i valori singoli dall'istanza della classe IrisSpecs.
        input_data = np.array([[
            specs.sepal_length, 
            specs.sepal_width, 
            specs.petal_length, 
            specs.petal_width
        ]])
        
        # Eseguiamo l'inferenza utilizzando il modello addestrato all'avvio dell'app.
        # Il metodo predict restituisce la classe predetta (0, 1 o 2).
        prediction = model.predict(input_data)[0]
        
        # Recuperiamo il nome leggibile della specie dal dataset Iris.
        # iris.target_names contiene ['setosa', 'versicolor', 'virginica'].
        species_name = iris.target_names[prediction]
        
        # Restituiamo un dizionario che FastAPI convertirà automaticamente in formato JSON.
        return {
            "status": "successo",
            "prediction": {
                "class_index": int(prediction),
                "species_name": species_name
            },
            "note": "Classificazione effettuata tramite modello di regressione logistica"
        }
        
    except Exception as e:
        # Se si verifica un errore durante il calcolo, solleviamo un'eccezione HTTP 500.
        # Questo assicura che il client riceva una risposta strutturata invece di un crash generico.
        raise HTTPException(status_code=500, detail=f"Errore tecnico durante l'inferenza: {str(e)}")

# Istruzioni per l'esecuzione in locale:
# Digitare nel terminale: uvicorn nome_file:app --reload
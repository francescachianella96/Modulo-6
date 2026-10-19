import os
import io
import base64
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from openai import OpenAI

from src.vae_model import TabularVAE
from src.evaluation import loss_function

# Backend non-interattivo di Matplotlib per evitare errori su server headless
matplotlib.use('Agg')

# Inizializza il client OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# System Prompt per il Natural Language Analyst
SYSTEM_PROMPT = """
Sei un Data Analyst esperto. Il tuo compito è analizzare un DataFrame Pandas chiamato `df`.
I dati contengono informazioni su attributi finanziari e personali (es. reddito, risparmio, età, ecc.).

REGOLE TASSATIVE:
1. Genera SOLO ed ESCLUSIVAMENTE codice Python valido per analizzare i dati o creare un grafico con Matplotlib/Seaborn.
2. Non racchiudere il codice in blocchi markdown (no ```python ... ```) e non aggiungere altro testo.
3. Non chiamare mai `plt.show()`.
4. Aggiungi sempre titolo ed etichette agli assi del grafico.
"""

def generate_analysis_code(prompt: str) -> str:
    """Richiede a OpenAI di generare codice Python in base alla richiesta dell'utente."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Richiesta utente: {prompt}"}
        ],
        temperature=0.0
    )
    
    content = response.choices[0].message.content
    if not content:
        raise ValueError("L'LLM non ha restituito alcun contenuto.")
        
    code = content.strip()
    
    # Rimuove eventuali tag markdown se presenti
    if code.startswith("```"):
        code = "\n".join(code.split("\n")[1:-1])
    return code

def execute_generated_code(code: str, df: pd.DataFrame) -> str:
    """Esegue il codice Python generato e restituisce il grafico in formato Base64."""
    plt.close('all')  # Pulisce le figure precedenti
    
    local_vars = {"df": df, "plt": plt, "pd": pd}
    
    try:
        exec(code, {}, local_vars)
        
        fig = plt.gcf()
        if fig.get_axes():  # Se è stato generato un grafico
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close('all')
            return img_base64
        else:
            plt.close('all')
            return None
    except Exception as e:
        plt.close('all')
        raise RuntimeError(f"Errore durante l'esecuzione del codice: {str(e)}")

def train_vae_model(df: pd.DataFrame, epochs: int = 50, latent_dim: int = 4, batch_size: int = 64):
    """
    Addestra il modello TabularVAE utilizzando i dati del DataFrame.
    Restituisce il modello addestrato e le metriche finali.
    """
    data_tensor = torch.tensor(df.values, dtype=torch.float32)
    input_dim = data_tensor.shape[1]
    
    dataset = TensorDataset(data_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = TabularVAE(input_dim=input_dim, latent_dim=latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    final_loss = 0.0
    final_mse = 0.0
    final_kld = 0.0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_kld = 0.0
        
        for batch in train_loader:
            x = batch[0]
            optimizer.zero_grad()
            
            recon_x, mu, logvar = model(x)
            loss, mse, kld = loss_function(recon_x, x, mu, logvar, beta=1.0)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_mse += mse.item()
            epoch_kld += kld.item()
            
        num_batches = len(train_loader)
        final_loss = epoch_loss / num_batches
        final_mse = epoch_mse / num_batches
        final_kld = epoch_kld / num_batches
        
    metrics = {
        "epochs": epochs,
        "final_loss": round(final_loss, 4),
        "final_mse": round(final_mse, 4),
        "final_kld": round(final_kld, 4)
    }
    
    return model, metrics

def generate_synthetic_samples(model: TabularVAE, real_df: pd.DataFrame, num_samples: int = 100) -> pd.DataFrame:
    """
    Genera nuovi campioni sintetici campionando dallo spazio latente del VAE addestrato.
    Restituisce un DataFrame Pandas con lo stesso formato e colonne dell'originale.
    """
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim)
        synthetic_tensor = model.decoder(z)
        synthetic_data = synthetic_tensor.numpy()
        
    synthetic_df = pd.DataFrame(synthetic_data, columns=real_df.columns)
    return synthetic_df
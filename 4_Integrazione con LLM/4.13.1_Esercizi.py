import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# 1. Caricamento variabili d'ambiente
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, ".env")
load_dotenv(dotenv_path=env_path)

api_token = os.getenv("HF_TOKEN")

# 2. Configurazione Client
# Se vuoi forzare un errore di "Modello non trovato", cambia model_id in "modello-inesistente"
model_id = "gpt2" 
client = InferenceClient(model=model_id, token=api_token)

def esegui_completamento():
    prompt = "L'integrazione di sistemi IA richiede"
    
    print(f"Tentativo di chiamata API con modello: {model_id}...")
    
    try:
        # 3. Chiamata al modello per completamento testuale
        # Nota: usiamo text_generation per gpt2 (modello di completamento, non chat)
        response = client.text_generation(
            prompt, 
            max_new_tokens=20,
            stop_sequences=["."]
        )
        
        print("-" * 30)
        print(f"Risultato: {prompt} {response}")
        print("-" * 30)

    except Exception as e:
        # 4. Gestione Errori
        print("\n[!] ERRORE RILEVATO")
        if "401" in str(e):
            print("Causa probabile: Il token 'HF_TOKEN' nel file .env non è valido.")
        elif "404" in str(e):
            print(f"Causa probabile: Il modello '{model_id}' non è stato trovato.")
        else:
            print(f"Dettaglio tecnico: {e}")

if __name__ == "__main__":
    if not api_token:
        print("Errore: Variabile HF_TOKEN mancante nel file .env")
    else:
        esegui_completamento()
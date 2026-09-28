import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Determiniamo il percorso assoluto della cartella dello script per caricare il file .env correttamente,
# indipendentemente da dove viene lanciato il comando nel terminale.
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, ".env")
load_dotenv(dotenv_path=env_path)

api_token = os.getenv("HF_TOKEN")

# Controllo preventivo dell'autenticazione.
# Verifichiamo immediatamente se il token e' stato caricato correttamente. Se manca,
# solleviamo un'eccezione per bloccare l'esecuzione prima di tentare chiamate API destinate al fallimento.
if not api_token:
    raise ValueError("Errore critico: la variabile HF_TOKEN non e' stata configurata nel file .env.")

# Configurazione del client Hugging Face.
# Proviamo con Qwen 2.5 7B Instruct, un modello molto recente e performante,
# solitamente ben supportato dall'API gratuita.
model_id = "Qwen/Qwen2.5-7B-Instruct"
client = InferenceClient(model=model_id, token=api_token)

def genera_social_post(testo_lungo):
    """
    Trasforma un testo complesso in un riassunto per i social media usando l'API Conversational.
    
    Vantaggi:
    - Separazione tra istruzioni di sistema (ruolo 'system') e input utente (ruolo 'user').
    - Maggiore coerenza nelle risposte del modello.
    """
    
    # Definiamo la struttura della conversazione.
    messages = [
        {"role": "system", "content": "Sei un esperto di social media. Riassumi i testi in post per X (max 20 parole)."},
        {"role": "user", "content": testo_lungo}
    ]
    
    try:
        # Chiamata alla Chat Completion API.
        # Restituisce un oggetto strutturato da cui estraiamo il contenuto del messaggio.
        response = client.chat_completion(
            messages=messages,
            max_tokens=60,
            temperature=0.7
        )
        
        # Estraiamo il testo della risposta.
        return response.choices[0].message.content
    except Exception as e:
        return f"Impossibile completare la richiesta. Dettaglio errore: {e}"

# Flusso principale di esecuzione del programma.
if __name__ == "__main__":
    # 1. Definizione dell'input: un breve testo descrittivo.
    articolo = "L'intelligenza artificiale nel 2026 ha raggiunto vette incredibili grazie alla scalabilità delle GPU."
    
    # 2. Feedback visivo per l'utente sulla console.
    print("Contattando il modello per la generazione del post...")
    
    # 3. Esecuzione della logica: passiamo i dati alla funzione dedicata.
    # Qui avviene l'interazione tra la logica di business dello script e il servizio API esterno.
    risultato = genera_social_post(articolo)
    
    # 4. Output finale: stampa del risultato ottenuto o dell'eventuale messaggio di errore.
    print("-" * 30)
    print(f"Contenuto generato:\n{risultato}")
    print("-" * 30)
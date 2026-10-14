import torch
from huggingface_hub import InferenceClient
import io
import os
from dotenv import load_dotenv

# --- SEZIONE 1: CONFIGURAZIONE E AMBIENTE ---
# In questa fase inizializziamo le dipendenze esterne necessarie al funzionamento del sistema.

# Il modulo 'dotenv' permette di caricare le variabili d'ambiente da un file esterno (.env).
# Questo approccio separa la configurazione sensibile (come le chiavi API) dal codice logico.
load_dotenv()

# Estraiamo il token di autenticazione per Hugging Face.
# Il token permette al client di identificarsi sui server remoti per eseguire i modelli.
HF_TOKEN = os.getenv("HF_TOKEN")

# Istanziamo la classe 'InferenceClient'. 
# Questo oggetto funge da tramite (proxy) tra la nostra applicazione locale e l'Hub di Hugging Face.
# Gestisce le richieste HTTP, l'invio dei dati ai modelli remoti e la ricezione delle risposte.
client = InferenceClient(token=HF_TOKEN)

def smart_analyzer(text_input, image_url=None):
    """
    Questa funzione analizza un input misto (testo ed eventualmente immagine).
    Coordina l'uso di modelli NLP (testo) e modelli Computer Vision (immagini).
    
    L'interazione avviene in tre fasi:
    1. Comunicazione con l'API remota tramite 'client'.
    2. Elaborazione dei dati ricevuti tramite la libreria 'torch'.
    3. Interpretazione e stampa dei risultati finali.
    """
    print("--- Inizio Elaborazione ---")

    # --- FASE 1: ANALISI DEL TESTO (SENTIMENT ANALYSIS) ---
    # In questa sezione utilizziamo un modello di Natural Language Processing (NLP).
    print(f"Analisi testo in corso: {text_input}")
    
    # Il metodo 'text_classification' invia il testo al modello specificato.
    # Il server restituisce una lista di punteggi di confidenza per ogni possibile emozione.
    results = client.text_classification(text_input, model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    
    # Utilizziamo PyTorch per trasformare i dati grezzi dell'API in strutture numeriche gestibili.
    # Creiamo un tensore (un vettore matematico ottimizzato) contenente i punteggi.
    # Questo passaggio permette di operare su grandi volumi di dati con estrema velocità.
    scores = torch.tensor([res['score'] for res in results])
    
    # Salviamo le etichette (es. 'positivo', 'negativo') in una lista parallela ai punteggi.
    labels = [res['label'] for res in results]
    
    # La funzione 'torch.max' analizza il tensore dei punteggi per trovare il valore più alto.
    # Restituisce il valore massimo (max_score) e la sua posizione nell'indice (idx).
    max_score, idx = torch.max(scores, dim=0)
    
    # Utilizziamo l'indice trovato sopra per estrarre il nome dell'emozione corrispondente dalla lista etichette.
    sentiment = labels[idx.item()]
    
    print(f"Risultato NLP rilevato: {sentiment} (Indice di confidenza: {max_score:.2f})")

    # --- FASE 2: ANALISI DELL'IMMAGINE (IMAGE CLASSIFICATION) ---
    # Questa sezione viene eseguita solo se l'utente fornisce un URL di un'immagine.
    if image_url:
        print("\nAnalisi immagine allegata in corso...")
        
        # 'image_classification' carica l'immagine dall'URL e la processa tramite un modello Vision Transformer.
        # Il modello tenta di identificare l'oggetto principale all'interno dell'immagine.
        vision_res = client.image_classification(image_url, model="google/vit-base-patch16-224")
        
        # Anche qui sfruttiamo i tensori di PyTorch per una selezione rapida del risultato migliore.
        v_scores = torch.tensor([r['score'] for r in vision_res])
        v_labels = [r['label'] for r in vision_res]
        
        # 'torch.topk' con k=1 seleziona direttamente il primo elemento con la probabilità maggiore.
        top_val, top_idx = torch.topk(v_scores, k=1)
        
        # Stampiamo l'oggetto identificato e la probabilità associata.
        print(f"L'immagine sembra contenere: {v_labels[top_idx.item()]} (Probabilità calcolata: {top_val.item():.2f})")

    print("--- Operazione completata ---\n")

# --- SEZIONE DI TEST ---
# Eseguiamo due scenari d'uso per dimostrare la versatilità del sistema.

# Scenario 1: Input multimodale completo.
# Il sistema analizzerà sia il tono entusiasta del testo sia il contenuto dell'immagine.
smart_analyzer(
    text_input="Adoro questo nuovo framework, rende tutto più semplice!",
    image_url="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
)

# Scenario 2: Input solo testuale.
# Il sistema ignorerà la fase di analisi immagine, concentrandosi sulla rilevazione della criticità nel testo.
smart_analyzer(
    text_input="Il sistema è troppo lento e crasha spesso in produzione."
)
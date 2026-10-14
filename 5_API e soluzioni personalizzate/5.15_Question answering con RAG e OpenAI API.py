import os
import faiss
import numpy as np
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Configurazione del client per collegarsi al server locale di LM Studio
# Il client utilizza lo standard OpenAI per comunicare con l'istanza locale.
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# 1. GESTIONE HARDWARE E CARICAMENTO MODELLO DI EMBEDDING
# Verifichiamo la disponibilita' di hardware dedicato (GPU) per velocizzare i calcoli.
# Se non viene trovata una GPU NVIDIA, il sistema utilizzerà automaticamente la CPU.
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Caricamento modello di embedding su: {device}")

# Il modello 'all-MiniLM-L6-v2' trasforma ogni stringa di testo in un vettore di 384 dimensioni.
# Viene specificato il device per evitare conflitti con la gestione multi-GPU su sistemi Windows.
embed_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# 2. DATASET DI ESEMPIO (MEMORIA ESTERNA)
# Questi documenti rappresentano la base di conoscenza che il modello non conosce nativamente.
documents = [
    {"id": 1, "text": "Il Deep Learning e' una sottocategoria del Machine Learning basata su reti neurali artificiali."},
    {"id": 2, "text": "RAG (Retrieval-Augmented Generation) combina il recupero di documenti con la generazione di testo."},
    {"id": 3, "text": "FAISS e' una libreria sviluppata da Facebook per la ricerca efficiente di similarità in dataset densi."},
    {"id": 4, "text": "OpenAI fornisce API per accedere a modelli linguistici avanzati come GPT-4."},
    {"id": 5, "text": "LM Studio permette di eseguire modelli open-source localmente sul proprio computer."}
]

# 3. PREPARAZIONE E POPOLAMENTO DEL VECTOR DATABASE (FAISS)
# Impostiamo la dimensione dei vettori in base all'output del modello scelto (384).
dimensions = 384 

# IndexFlatL2 crea un indice che misura la distanza euclidea tra i vettori.
# E' ideale per dataset di piccole e medie dimensioni grazie alla sua precisione assoluta.
index = faiss.IndexFlatL2(dimensions)

print("Trasformazione testi in vettori e creazione indice...")
# Estraiamo solo il testo dai documenti per generare gli embeddings.
doc_texts = [doc["text"] for doc in documents]

# Trasformiamo la lista di testi in una matrice di vettori numerici.
embeddings = embed_model.encode(doc_texts)

# FAISS richiede che i dati siano in formato float32 per ottimizzare le operazioni matematiche.
# Una volta aggiunti all'indice, i documenti sono pronti per essere cercati semanticamente.
index.add(np.array(embeddings).astype('float32'))

def retrieve(query, k=2):
    """
    Cerca i k documenti più simili a una frase di input (query).
    
    Interazione:
    1. La query viene convertita in un vettore tramite il modello di embedding.
    2. FAISS confronta il vettore della query con tutti i vettori nell'indice.
    3. Vengono restituiti gli indici dei documenti con la distanza minore.
    """
    # Trasformiamo la domanda dell'utente nello stesso spazio vettoriale dei documenti.
    query_vector = embed_model.encode([query])
    
    # Eseguiamo la ricerca: 'distances' indica quanto sono simili, 'indices' indica quali sono.
    distances, indices = index.search(np.array(query_vector).astype('float32'), k)
    
    # Recuperiamo i documenti originali corrispondenti agli indici trovati.
    results = []
    for i in range(k):
        idx = indices[0][i]
        results.append(documents[idx])
    return results

def generate_answer(query):
    """
    Esegue l'intero ciclo RAG: Recupero delle info -> Arricchimento del Prompt -> Generazione Risposta.
    """
    
    # FASE 1: RETRIEVAL (RECUPERO)
    # Cerchiamo nei nostri documenti quelli che contengono la risposta potenziale.
    context_docs = retrieve(query)
    
    # FASE 2: AUGMENTATION (ARRICCHIMENTO)
    # Formattiamo i documenti recuperati in un'unica stringa leggibile per l'LLM.
    # Includiamo gli ID per permettere al modello di citare correttamente le fonti.
    context_text = "\n".join([f"Documento {d['id']}: {d['text']}" for d in context_docs])
    
    # Costruiamo il prompt specificando istruzioni rigide per evitare 'allucinazioni'.
    # Il modello deve rispondere SOLO basandosi sul contesto fornito.
    prompt = f"""Utilizza solo il CONTESTO fornito per rispondere alla domanda. 
Se il contesto non contiene le informazioni necessarie, rispondi che non lo sai.
Cita sempre il numero del documento tra parentesi quadre alla fine della frase (es. [1]).

CONTESTO:
{context_text}

DOMANDA:
{query}

RISPOSTA:"""

    # FASE 3: GENERATION (GENERAZIONE)
    # Inviamo il prompt costruito a LM Studio.
    try:
        response = client.chat.completions.create(
            model="gpt-oss-20b", 
            messages=[
                {"role": "system", "content": "Sei un assistente specializzato che utilizza esclusivamente i dati forniti per rispondere."},
                {"role": "user", "content": prompt}
            ],
            # La temperatura a 0.2 rende il modello deterministico e meno creativo, ideale per task di precisione.
            temperature=0.2 
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Errore durante la comunicazione con il server locale: {e}"

# PUNTO DI ACCESSO PRINCIPALE
if __name__ == "__main__":
    # Test della pipeline con una domanda specifica
    domanda_utente = "Cosa serve FAISS e chi lo ha sviluppato?"
    print(f"\nDomanda inoltrata: {domanda_utente}")
    
    # Avvio della pipeline RAG
    risposta = generate_answer(domanda_utente)
    
    print("\nRISULTATO DELLA GENERAZIONE RAG:\n")
    print(risposta)
    print("\n-----------------------------------\n")
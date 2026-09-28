import torch
from sentence_transformers import SentenceTransformer, util

# Configurazione del dispositivo di calcolo (GPU se disponibile, altrimenti CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. DEFINIZIONE DELLA KNOWLEDGE BASE
# Lista di documenti che rappresentano il contesto informativo del sistema.
documenti = [
    "Il Colosseo è un anfiteatro romano situato nel centro della città di Roma.",
    "La fotosintesi clorofilliana permette alle piante di trasformare la luce in energia chimica.",
    "Il protocollo TCP/IP è alla base del funzionamento della rete internet moderna.",
    "La pizza Margherita è stata creata in onore della Regina d'Italia nel 1889.",
    "I buchi neri sono regioni dello spaziotempo con un campo gravitazionale così forte che nulla può sfuggire."
]

# 2. GENERAZIONE DEGLI EMBEDDING
# Caricamento del modello pre-addestrato per la trasformazione del testo in vettori.
model_name = 'all-MiniLM-L6-v2'
embedder = SentenceTransformer(model_name, device=device)

# Codifica dei documenti in tensori numerici (embeddings)
corpus_embeddings = embedder.encode(documenti, convert_to_tensor=True)

# 3. FUNZIONE DI RICERCA SEMANTICA (RETRIEVAL)
def retrieve_top_k(query, k=3):
    """
    Individua i k documenti più simili alla query fornita in input.
    La similarità viene calcolata tramite il prodotto scalare dei vettori normalizzati:
    $$ \text{similarity} = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|} $$
    """
    # Trasformazione della query in embedding
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    
    # Calcolo della similarità cosina tra query e documenti
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    
    # Selezione dei k risultati con il punteggio più alto
    top_results = torch.topk(cos_scores, k=k)
    
    return top_results

# 4. ESECUZIONE DEL PROCESSO DI AUGMENTATION
query_utente = "Come fanno i vegetali a produrre nutrimento?"
top_k_results = retrieve_top_k(query_utente, k=3)

# Estrazione e formattazione dei documenti recuperati
contesto_recuperato = ""
for score, idx in zip(top_k_results[0], top_k_results[1]):
    contesto_recuperato += f"- {documenti[idx]} (Score: {score:.4f})\n"

# Costruzione del prompt strutturato per il modello linguistico (LLM)
prompt_finale = f"""
Sei un assistente utile. Rispondi alla domanda dell'utente utilizzando esclusivamente il contesto fornito. 
Se le informazioni non sono presenti nel contesto, ammetti di non saperlo.

CONTESTO RECUPERATO:
{contesto_recuperato}

DOMANDA: 
{query_utente}

RISPOSTA:
"""

# Output dei risultati
print("--- DOCUMENTI RECUPERATI ---")
print(contesto_recuperato)
print("--- PROMPT FINALE PER L'LLM ---")
print(prompt_finale)
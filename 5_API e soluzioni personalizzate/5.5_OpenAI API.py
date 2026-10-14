import torch
import json
#1
# Definiamo lo schema per il modello (JSON Schema)
# Questo istruisce l'LLM a rispondere solo con i campi richiesti
ticket_tool = {
    "type": "function",
    "function": {
        "name": "classify_ticket",
        "description": "Estrae categoria e urgenza da un ticket tecnico",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "enum": ["hardware", "software", "network"]},
                "priority": {"type": "integer", "description": "Scala 1-5"},
                "is_urgent": {"type": "boolean"}
            },
            "required": ["topic", "priority", "is_urgent"]
        }
    }
}

# Esempio di messaggio di sistema per definire il "carattere" 
system_prompt = "Sei un analista tecnico senior. Rispondi sempre in modo conciso e strutturato."

# 2
def check_similarity(query_vector, historical_vectors):
    """
    Confronta un nuovo ticket con lo storico usando PyTorch.
    query_vector: tensor [1, 1536]
    historical_vectors: tensor [N, 1536]
    """
    # 1. Convertiamo in tensori PyTorch (se non lo sono già)
    q = torch.tensor(query_vector)
    h = torch.tensor(historical_vectors)

    # 2. Calcolo della similitudine cosina (vettorializzato)
    # Finiamo per fare un prodotto scalare su vettori già normalizzati
    similarities = torch.nn.functional.cosine_similarity(q, h)
    
    # 3. Troviamo il match più vicino
    best_match_idx = torch.argmax(similarities)
    return best_match_idx.item(), similarities[best_match_idx].item()

# Esempio d'uso fittizio
current_ticket = [0.12, -0.05, 0.88] # Immaginiamo sia un embedding ridotto
history = [[0.11, -0.04, 0.85], [0.9, 0.1, -0.2]]
idx, score = check_similarity(current_ticket, history)

print(f"Match trovato nel database storico all'indice {idx} con score: {score:.4f}")



# 3

{
  "messages": [
    {"role": "system", "content": "Sei un assistente tecnico che usa un linguaggio formale e privo di gergo inutile."},
    {"role": "user", "content": "Il server non risponde."},
    {"role": "assistant", "content": "Ho preso in carico la segnalazione. Procedo con la verifica dello stato del nodo e dei log di sistema."}
  ]
}
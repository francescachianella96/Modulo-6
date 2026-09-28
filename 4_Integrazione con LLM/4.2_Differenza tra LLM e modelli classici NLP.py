import gensim.downloader as api

# Carichiamo un modello leggero (glove o word2vec)
# Scegliamo 'glove-wiki-gigaword-50' per la rapidità
model_static = api.load("glove-wiki-gigaword-50")

# Calcolo somiglianza (usando traduzioni o termini italiani se il modello lo permette)
# GloVe è rigido: 'mela' e 'pesca' avranno sempre lo stesso score.
sim = model_static.similarity('apple', 'peach') 
print(f"Somiglianza statica (GloVe) mela/pesca: {sim:.4f}")


import torch
from transformers import AutoTokenizer, AutoModel

# Usiamo un modello BERT italiano per coerenza
model_name = "dbmdz/bert-base-italian-xxl-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_context = AutoModel.from_pretrained(model_name)

frasi = [
    "Mi piace la pesca al lago", # Senso: Sport
    "Ho mangiato una pesca dolce" # Senso: Frutta
]

def get_word_vector(sentence, word):
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model_context(**inputs)
    
    # Troviamo l'indice del token "pesca"
    token_logits = tokenizer.tokenize(sentence)
    word_idx = token_logits.index(word.lower()) + 1 # +1 per il token [CLS]
    
    # Estraiamo l'embedding dall'ultimo hidden state
    return outputs.last_hidden_state[0, word_idx, :]

vec_sport = get_word_vector(frasi[0], "pesca")
vec_frutta = get_word_vector(frasi[1], "pesca")



# Calcolo con PyTorch puro
cos = torch.nn.CosineSimilarity(dim=0)
distanza_contestuale = cos(vec_sport, vec_frutta)

print(f"Somiglianza tra i due contesti di 'pesca': {distanza_contestuale.item():.4f}")

# ANALISI
# GloVe: Distanza = 0 (perché il vettore è IDENTICO, la 'pesca' è una sola)
# BERT: La distanza riflette lo slittamento semantico.
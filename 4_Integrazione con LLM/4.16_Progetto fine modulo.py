import os
import json
import time
from typing import List, Dict, Any
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,  # Importato per la Soluzione 1
    BitsAndBytesConfig,
    TextIteratorStreamer
)
from threading import Thread
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ==============================================================================
# FASE 1: SETUP E CARICAMENTO MODELLO QUANTIZZATO (4-bit BitsAndBytes)
# ==============================================================================
print("=== FASE 1: CARICAMENTO MODELLO E TOKENIZER ===")

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# 1. Configurazione della quantizzazione NF4 a 4-bit per GPU T4
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# 2. Caricamento del Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. Caricamento del Modello Quantizzato su GPU con Type Hinting esplicito (Soluzione 1)
model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quantization_config,
    device_map="auto"
)

def format_llama3_prompt(system_prompt: str, user_prompt: str) -> str:
    """
    Utility per formattare i prompt secondo lo standard nativo di Llama 3.
    """
    return (
        f"<|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n{user_prompt}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )

def genera_risposta(prompt: str, max_new_tokens: int = 512, temperature: float = 0.1, top_p: float = 0.9) -> str:
    """
    Funzione helper per l'inferenza standard (non-streaming).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id
        )
    # Decodifichiamo solo i nuovi token generati dal modello
    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


# ==============================================================================
# FASE 2: PROMPT ENGINEERING E REASONING (Chain of Thought - CoT)
# ==============================================================================
print("\n=== FASE 2: CHAIN OF THOUGHT (CoT) REASONING ===")

system_prompt_cot = (
    "Sei un assistente AI specializzato nel ragionamento logico-matematico. "
    "Devi analizzare i problemi passo dopo passo (Chain of Thought), spiegando il "
    "tuo ragionamento in modo chiaro e strutturato prima di fornire la risposta finale."
)

user_problem_cot = (
    "In un'officina ci sono auto e moto. In totale ci sono 30 veicoli e 84 ruote. "
    "Quante auto e quante moto ci sono nell'officina? Spiega il ragionamento passo dopo passo."
)

prompt_cot = format_llama3_prompt(system_prompt_cot, user_problem_cot)
risposta_cot = genera_risposta(prompt_cot, max_new_tokens=400, temperature=0.1)

print("--- [Input Utente] ---")
print(user_problem_cot)
print("\n--- [Risposta CoT del Modello] ---")
print(risposta_cot)


# ==============================================================================
# FASE 3: RETRIEVAL-AUGMENTED GENERATION (RAG)
# ==============================================================================
print("\n=== FASE 3: RETRIEVAL-AUGMENTED GENERATION (RAG) ===")

# 1. Dataset di conoscenza
knowledge_base = [
    "Il progetto Apollo 11 è stata la missione spaziale che ha portato i primi uomini sulla Luna il 20 luglio 1969.",
    "Gli astronauti Neil Armstrong e Buzz Aldrin sono camminati sulla Luna, mentre Michael Collins è rimasto in orbita nel modulo di comando.",
    "Il calcolo quantistico utilizza i qubit al posto dei bit classici. Un qubit può esistere in uno stato di sovrapposizione.",
    "Il Telescopio Spaziale James Webb (JWST) è stato lanciato il 25 dicembre 2021 ed opera principalmente negli infrarossi.",
    "La Regola dell'Ottetto in chimica stabilisce che gli atomi tendono a combinarsi in modo da avere 8 elettroni nel loro guscio di valenza."
]

# 2. Inizializzazione del modello di embedding e creazione del Vector Index (FAISS)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(knowledge_base, convert_to_numpy=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype(np.float32))

def retrieve_context(query: str, k: int = 2) -> str:
    """Cerca i documenti più pertinenti nel Vector Store."""
    query_vector = embedding_model.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(query_vector, k)
    retrieved_chunks = [knowledge_base[i] for i in indices[0]]
    return "\n".join(f"- {chunk}" for chunk in retrieved_chunks)

# 3. Esecuzione RAG Query
user_query_rag = "Chi è rimasto in orbita durante lo sbarco sulla Luna del 1969?"
contexto = retrieve_context(user_query_rag, k=2)

system_prompt_rag = (
    "Sei un assistente preciso. Rispondi alla domanda dell'utente utilizzando "
    "ESCLUSIVAMENTE le informazioni fornite nel CONTESTO. Se l'informazione non è "
    "presente nel contesto, rispondi 'Non riesco a trovare questa informazione nel contesto fornito.'"
)

user_prompt_rag = f"CONTESTO:\n{contexto}\n\nDOMANDA: {user_query_rag}"
prompt_rag = format_llama3_prompt(system_prompt_rag, user_prompt_rag)

risposta_rag = genera_risposta(prompt_rag, max_new_tokens=200, temperature=0.0)

print(f"--- [Query RAG]: {user_query_rag} ---")
print(f"--- [Contesto Recuperato] ---\n{contexto}")
print("\n--- [Risposta RAG] ---")
print(risposta_rag)


# ==============================================================================
# FASE 4: ZERO-SHOT SENTIMENT ANALYSIS E INFERENCE STREAMING
# ==============================================================================
print("\n=== FASE 4: ZERO-SHOT ANALYSIS CON STREAMING ===")

recensioni_test = [
    "Il prodotto è arrivato puntuale, qualità eccellente e imballaggio perfetto!",
    "Servizio clienti pessimo, il pacco è arrivato danneggiato e in ritardo.",
    "Funziona come descritto, niente di eccezionale ma fa il suo dovere."
]

system_prompt_sentiment = (
    "Sei un sistema di analisi del sentiment. Classifica il testo fornito "
    "in una delle seguenti categorie: 'Positivo', 'Negativo' o 'Neutro'. "
    "Rispondi SOLO ed ESCLUSIVAMENTE con una di queste tre parole."
)

def esegui_streaming(prompt: str):
    """
    Esegue l'inferenza trasmettendo l'output token per token (Streaming)
    sfruttando TextIteratorStreamer su un thread separato.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=50,
        temperature=0.1,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Avviamo la generazione su un thread separato per non bloccare il ciclo di lettura
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Stampiamo i token man mano che vengono generati dal modello
    for new_text in streamer:
        print(new_text, end="", flush=True)
    print()

print("--- Testing Streaming per Zero-Shot Sentiment Analysis ---")
for i, rec in enumerate(recensioni_test, 1):
    prompt_sentiment = format_llama3_prompt(system_prompt_sentiment, f"Testo: '{rec}'")
    print(f"\nRecensione {i}: \"{rec}\"")
    print("Sentiment: ", end="")
    esegui_streaming(prompt_sentiment)


# ==============================================================================
# FASE 5: OUTPUT STRUTTURATO (JSON)
# ==============================================================================
print("\n=== FASE 5: OUTPUT STRUTTURATO IN FORMATO JSON ===")

system_prompt_json = (
    "Sei un estrattore di dati strutturati. Analizza il testo dell'utente ed estrai le informazioni. "
    "Devi rispondere ESCLUSIVAMENTE con un oggetto JSON valido, senza testo introduttivo, "
    "spiegazioni o blocchi markdown di codice. Il formato deve essere:\n"
    "{\n"
    '  "prodotto": "nome del prodotto",\n'
    '  "valutazione": "Positiva/Negativa/Nuetral",\n'
    '  "aspetti_chiave": ["aspetto1", "aspetto2"]\n'
    "}"
)

testo_recensione_json = (
    "Ho acquistato le cuffie Wireless SoundPro. Il suono è limpido e i bassi sono potenti, "
    "ma la durata della batteria è deludente e l'archetto fa male dopo un'ora di utilizzo."
)

prompt_json = format_llama3_prompt(system_prompt_json, f"Testo da analizzare: '{testo_recensione_json}'")
risposta_json_raw = genera_risposta(prompt_json, max_new_tokens=250, temperature=0.1)

print("--- [Output Grezzo dal Modello] ---")
print(risposta_json_raw)

# Validazione del JSON
print("\n--- [Validazione del JSON] ---")
try:
    dati_json = json.loads(risposta_json_raw)
    print("JSON Valido ed Estratto con Successo:")
    print(json.dumps(dati_json, indent=2, ensure_ascii=False))
except json.JSONDecodeError as e:
    print(f"Errore nella decodifica JSON: {e}")
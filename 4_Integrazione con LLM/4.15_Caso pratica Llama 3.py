import os
import time
from dotenv import load_dotenv
from llama_cpp import Llama # pip install llama-cpp-python


# SEZIONE 1: DEFINIZIONE DEI DATI E LOGICA DI FORMATTAZIONE
# In questa sezione prepariamo i dati di input e definiamo come devono essere presentati al modello.

# Lista di stringhe che rappresentano i dati grezzi da analizzare.
# Ogni elemento sara' processato individualmente dal modello di linguaggio.
recensioni = [
    "Un capolavoro assoluto, la regia è incredibile!",
    "Pessimo film, una noia mortale dall'inizio alla fine.",
    "Non saprei, alcune parti sono belle ma il finale è debole.",
    "Esperienza visiva discreta, ma la trama manca di spessore.",
    "Il miglior film dell'anno, attori straordinari.",
    "Delusione totale, soldi buttati."
]

def format_prompt(text):
    """
    Questa funzione implementa il 'Prompt Template' specifico per l'architettura Llama 3.
    Struttura il testo inserendo dei marcatori (tag) che il modello riconosce per distinguere
    le istruzioni di sistema dall'input dell'utente.
    """
    return (
        "<|start_header_id|>system<|end_header_id|>\n"
        "Sei un analista di sentiment. Rispondi esclusivamente con: 'Positive', 'Negative' o 'Neutral'.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"Testo: {text}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

# SEZIONE 2: CUORE COMPUTAZIONALE E LOGICA DI BENCHMARK
# Questa funzione incapsula l'istanza del modello e la logica di misurazione delle prestazioni.

def esegui_analisi(descrizione, num_threads, batch_size):
    """
    Parametri:
    - descrizione: Etichetta del test per i log di output.
    - num_threads: Numero di processori (core) assegnati alla computazione.
    - batch_size: Quantita' di dati processati contemporaneamente.
    """
    print(f"\nAvvio test: {descrizione}")
    print(f"Configurazione: {num_threads} Thread(s), Batch Size {batch_size}")
    
    # Istanziazione della classe Llama.
    # Questo oggetto carica il modello pesantemente quantizzato (.gguf) in memoria RAM.
    # La configurazione n_threads e n_batch determina l'efficienza computazionale su CPU.
    llm = Llama.from_pretrained(
        repo_id="bartowski/Meta-Llama-3-8B-Instruct-GGUF",
        filename="Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        n_ctx=512,           # Dimensione massima della finestra di contesto (token)
        n_threads=num_threads, # Parallelismo a livello di istruzioni CPU
        n_batch=batch_size,    # Dimensione del blocco di token per il prompt processing
        n_gpu_layers=0,      # Specifica l'uso esclusivo della CPU (0 layer su GPU)
        verbose=False        # Disattiva i log interni della libreria per pulizia output
    )
    
    risultati_test = []
    # Registrazione del tempo iniziale per il calcolo delle metriche temporali
    start_time = time.time()
    
    # Iterazione sulla lista delle recensioni
    for rec in recensioni:
        # Trasformazione del testo semplice in un prompt strutturato
        prompt = format_prompt(rec)
        
        # Chiamata al metodo __call__ dell'oggetto llm, che esegue l'inferenza.
        # temperature=0.1 garantisce risposte quasi deterministiche.
        # stop definisce i simboli che interrompono la generazione di testo.
        response = llm(
            prompt,
            max_tokens=10,
            temperature=0.1, 
            stop=["<|eot_id|>", "\n"]
        )
        
        # Estrazione della risposta testuale dal dizionario restituito dal modello
        sentiment = response["choices"][0]["text"].strip()
        risultati_test.append(sentiment)
    
    # Calcolo della durata complessiva dell'operazione
    end_time = time.time()
    durata = end_time - start_time
    # Calcolo del Throughput: numero di operazioni completate per unita' di tempo (secondi)
    throughput = len(recensioni) / durata
    
    return durata, throughput, risultati_test

# SEZIONE 3: ESECUZIONE DEI TEST COMPARATIVI
# Qui confrontiamo due diverse strategie di utilizzo delle risorse hardware.

# Scenario A: Simulazione di un carico di lavoro non ottimizzato (esecuzione seriale).
tempo_lento, tp_lento, _ = esegui_analisi(
    "Scenario A: Configurazione Seriale/Limitata", 
    num_threads=1, 
    batch_size=8
)

# Scenario B: Utilizzo ottimizzato del parallelismo hardware tramite thread multipli e batch piu' ampi.
tempo_veloce, tp_veloce, risultati_finali = esegui_analisi(
    "Scenario B: Configurazione Parallelizzata/Batch", 
    num_threads=4, 
    batch_size=512
)

# SEZIONE 4: PRESENTAZIONE DEI RISULTATI E ANALISI METRICHE
# Confronto finale tra i due scenari per validare l'efficacia delle ottimizzazioni.

print(f"\n--- CONFRONTO PRESTAZIONI ---")
print(f"Tempo Scenario A (Limitato):   {tempo_lento:.2f} secondi")
print(f"Tempo Scenario B (Ottimizzato): {tempo_veloce:.2f} secondi")

# Calcolo matematico della percentuale di miglioramento tra i due tempi di esecuzione.
miglioramento = ((tempo_lento - tempo_veloce) / tempo_lento) * 100
print(f"Incremento di velocità: +{miglioramento:.1f}%")

# Il throughput indica quanti campioni il sistema riesce a gestire ogni secondo.
print(f"Throughput finale: {tp_veloce:.2f} campioni al secondo")

print("\n--- ESEMPIO RISULTATI FINALI ---")
# Visualizzazione pratica della classificazione effettuata dal modello.
for rec, sent in zip(recensioni[:3], risultati_finali[:3]):
    print(f"Testo: {rec[:40]}... -> Sentiment: {sent}")
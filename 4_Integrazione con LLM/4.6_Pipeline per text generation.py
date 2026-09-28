import torch
from transformers import pipeline, AutoTokenizer


# 1. LA PIPELINE: L'astrazione di alto livello
# La funzione pipeline è lo strumento principale della libreria Transformers per eseguire task predefiniti.
# Internamente, essa coordina l'interazione tra tre componenti fondamentali:
# - Il Tokenizer: che prepara i dati in ingresso.
# - Il Modello: che esegue il calcolo delle probabilità statistica.
# - Il Decoder: che trasforma i risultati numerici nuovamente in testo leggibile.
# Utilizziamo 'distilgpt2', una versione alleggerita di GPT-2 progettata per essere veloce ed efficiente.
# Il parametro 'device=0' indica al sistema di utilizzare la prima GPU disponibile (CUDA),
# mentre 'framework="pt"' forza esplicitamente l'uso di PyTorch.
generator = pipeline('text-generation', model='distilgpt2', device=0, framework='pt')

# Definiamo il 'prompt', ovvero la sequenza di parole iniziale che il modello dovrà completare.
# Usiamo l'inglese per ottenere risultati migliori, dato che il modello è stato addestrato principalmente su questa lingua.
prompt_base = "In the heart of the forest, an ancient robot"

print("--- 1. GENERAZIONE STANDARD (PIPELINE) ---")
# Richiamando l'oggetto 'generator', la pipeline esegue l'intero ciclo di vita dell'inferenza:
# 1. Tokenizzazione del prompt.
# 2. Passaggio dei token al modello linguistico.
# 3. Campionamento delle parole successive fino al raggiungimento di 'max_new_tokens'.
# 4. Decodifica dei token generati in testo leggibile.
risultato_base = generator(prompt_base, max_new_tokens=20, truncation=True)
print(risultato_base[0]['generated_text'])


# 2. IL TOKENIZER: La rappresentazione numerica del linguaggio
# I modelli linguistici non comprendono i caratteri alfabetici, ma operano esclusivamente su numeri.
# La classe AutoTokenizer carica la configurazione specifica per il modello selezionato ('distilgpt2').
# Questo garantisce che il testo venga scomposto esattamente come il modello si aspetta.
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')

print("\n--- 2. DIETRO LE QUINTE: TOKENIZZAZIONE ---")
# Il metodo 'encode' trasforma la stringa in una sequenza di identificativi numerici (ID).
# Ogni numero rappresenta un'unità minima di significato presente nel vocabolario del modello.
tokens = tokenizer.encode(prompt_base)

print(f"Testo originale: {prompt_base}")
print(f"Equivalenti numerici (ID): {tokens}")

# Eseguiamo un ciclo per decodificare singolarmente ogni ID e vedere come il tokenizer ha "spezzato" la frase.
# Noterete che la punteggiatura e gli spazi iniziali vengono spesso accorpati alle parole.
print(f"Scomposizione dei singoli token: {[tokenizer.decode([t]) for t in tokens]}")


# 3. I PARAMETRI DI GENERAZIONE: Controllo della logica e della creatività
# Quando il modello genera una parola, assegna una probabilità a ogni termine del suo vocabolario.
# Possiamo influenzare questo processo di selezione attraverso specifici parametri chiamati iperparametri.

print("\n--- 3. IL BILANCIAMENTO DELLA GENERAZIONE ---")

# ESEMPIO A: Bassa Temperatura (Massima coerenza)
# La 'temperature' controlla quanto il modello debba essere "audace".
# Con un valore basso (0.1), il modello sceglie quasi esclusivamente le parole più probabili.
# Risultato: Testo molto logico e strutturato, ma potenzialmente ripetitivo.
output_freddo = generator(
    prompt_base, 
    max_new_tokens=30, 
    temperature=0.1,  # Valore vicino allo zero: approccio quasi deterministico
    do_sample=True,
    truncation=True
)

# ESEMPIO B: Alta Temperatura e Top-P (Creatività controllata)
# Una temperatura elevata (1.8) "appiattisce" le probabilità, rendendo parole rare molto più probabili.
# Per evitare che il testo diventi del tutto incoerente, usiamo il 'top_p' (Nucleus Sampling).
# Il parametro 'top_p=0.85' dice al modello di considerare solo le parole che sommate costituiscono l'85% 
# della massa di probabilità, eliminando la "coda" di termini senza senso.
output_caldo = generator(
    prompt_base, 
    max_new_tokens=30, 
    temperature=1.8,  # Alta creatività, rischio di instabilità
    top_p=0.85,       # Filtro per mantenere una coerenza grammaticale
    do_sample=True,
    truncation=True
)

print(f"APPROCCIO LOGICO (T=0.1): {output_freddo[0]['generated_text']}")
print(f"APPROCCIO CREATIVO (T=1.8): {output_caldo[0]['generated_text']}")
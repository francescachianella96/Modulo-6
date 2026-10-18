import pandas as pd
import numpy as np
import hashlib
import json
import os
import random
import string

# FASE 1: ESTRAZIONE E GENERAZIONE DATI (JSONL)
# Generiamo un file 'data.jsonl' con 100 record casuali.
# Il formato JSONL (JSON Lines) è molto comune per i dataset di testo.

def generate_random_text(min_len=2, max_len=10):
    """Genera una stringa casuale di lunghezza variabile."""
    length = random.randint(min_len, max_len)
    return ''.join(random.choices(string.ascii_letters + " ", k=length))

def generate_initial_data(filename, num_records=100):
    print(f"Generazione di {num_records} record in {filename}...")
    with open(filename, 'w', encoding='utf-8') as f:
        for i in range(num_records):
            record = {
                "id": i,
                "text": generate_random_text(),
                "label": random.randint(0, 1)
            }
            f.write(json.dumps(record) + "\n")

raw_jsonl = "data.jsonl"
generate_initial_data(raw_jsonl)

# FASE 2: TRASFORMAZIONE (CLEANING)
# Carichiamo i dati e applichiamo la logica di pulizia richiesta:
# Rimuovere i record con testo più corto di 5 caratteri.

def clean_and_process(input_file):
    # Leggiamo il file JSONL usando pandas
    df = pd.read_json(input_file, lines=True)
    
    # Visualizziamo la dimensione iniziale
    initial_count = len(df)
    
    # Applichiamo il filtro: teniamo solo testi con lunghezza >= 5
    df_cleaned = df[df['text'].str.len() >= 5].copy()
    
    final_count = len(df_cleaned)
    print(f"Pulizia completata: {initial_count - final_count} record rimossi perché troppo brevi.")
    
    return df_cleaned

df_cleaned = clean_and_process(raw_jsonl)

# FASE 3: CARICAMENTO (SALVATAGGIO IN PARQUET)
# Il formato Parquet garantisce efficienza e preserva i tipi dei dati.

output_parquet = "data.parquet"
df_cleaned.to_parquet(output_parquet, engine='pyarrow', compression='snappy')
print(f"Dataset salvato correttamente in: {output_parquet}")

# FASE 4: SIMULAZIONE TRACCIAMENTO DVC (HASHING)
# Calcoliamo l'hash MD5 per identificare univocamente questa versione dei dati.

def calculate_file_hash(path):
    """Calcola l'impronta MD5 di un file per il versionamento."""
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        # Lettura a blocchi per efficienza
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

data_hash = calculate_file_hash(output_parquet)

print(f"\n--- SIMULAZIONE DVC ---")
print(f"File monitorato: {output_parquet}")
print(f"Hash MD5 (Versione Corrente): {data_hash}")
print(f"Istruzione didattica: In DVC, questo hash verrebbe salvato nel file 'data.parquet.dvc'")

# Opzionale: Creazione del file .dvc per completezza
dvc_data = {
    "outs": [{
        "md5": data_hash,
        "path": output_parquet
    }]
}

with open(f"{output_parquet}.dvc", "w") as f:
    json.dump(dvc_data, f, indent=4)

print(f"File di metadati creato: {output_parquet}.dvc")
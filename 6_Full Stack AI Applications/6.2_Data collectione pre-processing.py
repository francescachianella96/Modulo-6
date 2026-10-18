import pandas as pd
import numpy as np
import os
import hashlib
import json

# FASE 1: GENERAZIONE DI UN DATASET AMPIO
# Per apprezzare la differenza tra i formati di file, generiamo un dataset di dimensioni significative.
# Simuliamo 1 milione di righe con dati numerici e categorici.
print("Generazione del dataset in corso...")
num_rows = 1_000_000

data = {
    'id': np.arange(num_rows),
    'valore_numerico': np.random.randn(num_rows), # Distribuzione normale
    'punteggio': np.random.randint(1, 100, size=num_rows),
    'categoria': np.random.choice(['A', 'B', 'C', 'D'], size=num_rows)
}

df = pd.DataFrame(data)

# FASE 2: CONFRONTO TRA FORMATI (CSV vs PARQUET)
# Salviamo lo stesso dataset in formato CSV (testuale) e Parquet (colonnare binario).

csv_file = "dataset_didatttico.csv"
parquet_file = "dataset_didattico.parquet"

# Salvataggio in CSV
df.to_csv(csv_file, index=False)

# Salvataggio in Parquet (formato standard per Big Data)
df.to_parquet(parquet_file, engine='pyarrow', compression='snappy')

# Calcolo e visualizzazione delle dimensioni dei file
size_csv = os.path.getsize(csv_file) / (1024 * 1024) # Dimensione in MB
size_parquet = os.path.getsize(parquet_file) / (1024 * 1024) # Dimensione in MB

print(f"\n--- CONFRONTO DIMENSIONI ---")
print(f"Dimensione file CSV:     {size_csv:.2f} MB")
print(f"Dimensione file Parquet: {size_parquet:.2f} MB")
print(f"Risparmio di spazio:     {((size_csv - size_parquet) / size_csv) * 100:.2f}%")

# FASE 3: LOGICA DI HASHING E VERSIONAMENTO (FILOSOFIA DVC)
# In ML Ops, non carichiamo i grandi file di dati su Git. Versioniamo i metadati e l'hash del file.

def calculate_md5(file_path):
    """
    Calcola l'impronta digitale (hash) del file. 
    Se anche un solo bit nel file cambia, l'hash sarà completamente diverso.
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        # Leggiamo il file a blocchi per non saturare la memoria con file giganti
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# Calcoliamo l'hash del nostro file Parquet
data_hash = calculate_md5(parquet_file)
print(f"\n--- LOGICA DVC ---")
print(f"Hash calcolato per il dataset: {data_hash}")

# Creazione del file .dvc (Metadati di puntamento)
# Questo è il file che verrebbe effettivamente salvato su Git al posto del dataset reale.
dvc_structure = {
    "outs": [
        {
            "md5": data_hash,
            "path": parquet_file,
            "size": os.path.getsize(parquet_file)
        }
    ],
    "meta": {
        "descrizione": "Dataset generato per test efficienza formati",
        "autore": "Docente",
        "versione": "1.0"
    }
}

dvc_filename = f"{parquet_file}.dvc"
with open(dvc_filename, "w") as f:
    json.dump(dvc_structure, f, indent=4)

print(f"File di versionamento creato: {dvc_filename}")
print("\nNota didattica: In un progetto reale, il file Parquet verrebbe caricato su un Cloud Storage (S3/Azure),")
print("mentre il file .dvc verrebbe tracciato da Git per garantire la riproducibilità degli esperimenti.")

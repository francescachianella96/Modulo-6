import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

# --- 1. GENERAZIONE DEL DATASET SINTETICO ---
# Creiamo un dataset di 1000 campioni che simula il rischio di insolvenza (default) dei prestiti.
n_samples = 1000
np.random.seed(42)

# Feature 1: Stipendio Annuo. Utilizziamo una distribuzione Gamma per renderla più realistica (molti medi, pochi alti).
stipendio = np.random.gamma(shape=5, scale=8000, size=n_samples) 
# Feature 2: Tipo di Contratto. Variabile categorica con frequenze predefinite.
tipo_contratto = np.random.choice(['indeterminato', 'determinato', 'part-time'], 
                                  size=n_samples, p=[0.6, 0.25, 0.15])

# Definizione del target (rischio_default) basato su logica statistica:
# Il rischio diminuisce all'aumentare dello stipendio e aumenta con contratti meno stabili.
rischio_score = -0.6 * (stipendio / 10000)
rischio_score += np.where(tipo_contratto == 'determinato', 1.0, 0)
rischio_score += np.where(tipo_contratto == 'part-time', 2.0, 0)
rischio_score += np.random.normal(0, 1, size=n_samples) # Aggiunta di rumore per non renderlo deterministico

# Creiamo uno sbilanciamento forzato: solo il top 10% dei punteggi di rischio viene marcato come '1' (Default).
soglia = np.percentile(rischio_score, 90)
rischio_default = (rischio_score >= soglia).astype(int)

# Organizzazione dei dati in un DataFrame Pandas per una gestione più semplice
df = pd.DataFrame({
    'stipendio_annuo': stipendio,
    'tipo_contratto': tipo_contratto,
    'rischio_default': rischio_default
})

# --- 2. SUDDIVISIONE DEI DATI (STRATEGIA ANTI-LEAKAGE) ---
# Separazione tra variabili indipendenti (X) e target (y).
X = df.drop('rischio_default', axis=1)
y = df['rischio_default']

# Suddivisione in Train e Test (80/20).
# Usiamo 'stratify=y' per garantire che la proporzione di 10% di default sia mantenuta in entrambi i set.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- 3. PIPELINE DI PRE-PROCESSING ---
# Poiché abbiamo tipi di dati diversi (numerici e categorici), dobbiamo applicare trasformazioni mirate.
numeric_features = ['stipendio_annuo']
categorical_features = ['tipo_contratto']

# ColumnTransformer permette di applicare trasformazioni diverse a colonne diverse in un unico passaggio.
preprocessor = ColumnTransformer(
    transformers=[
        # StandardScaler: Centra la media a 0 e la deviazione standard a 1 (fondamentale per reti neurali).
        ('num', StandardScaler(), numeric_features),
        # OneHotEncoder: Converte stringhe in vettori binari (es. 'part-time' -> [0, 0, 1]).
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
    ]
)

# FIT dello scaler solo sui dati di TRAIN per evitare Data Leakage.
X_train_proc = preprocessor.fit_transform(X_train)

# TRASFORMAZIONE del TEST set usando SOLO le statistiche apprese dal train.
X_test_proc = preprocessor.transform(X_test)

# Conversione dei risultati (che sono array NumPy) in Tensori PyTorch.
X_train_t = torch.tensor(X_train_proc, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

# --- 4. CALCOLO DEI PESI PER LE CLASSI SBILANCIATE ---
# Calcoliamo i pesi correttivi per compensare la rarità della classe 1 (guasti/default).
weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
# In BCELoss, il peso 'pos_weight' si riferisce specificamente alla classe 1.
# weights[1] contiene il peso calcolato per la classe 1.
class_weights = torch.tensor(weights[1], dtype=torch.float32)
print(f"Peso correttivo assegnato alla classe rara (1): {class_weights:.2f}")

# --- 5. DEFINIZIONE DELL'ARCHITETTURA DEL MODELLO ---
class SimpleClassifier(nn.Module):
    """
    Rete neurale semplice con un layer nascosto.
    Interazioni:
    - __init__: Definisce gli strati della rete (Linear -> ReLU -> Linear).
    - forward: Definisce come i dati fluiscono attraverso la rete.
    """
    def __init__(self, input_shape):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_shape, 4), # Primo strato lineare
            nn.ReLU(),                 # Funzione di attivazione non lineare
            nn.Linear(4, 1)            # Layer di output (senza Sigmoid, gestita dalla Loss)
        )
    
    def forward(self, x):
        return self.net(x)

# Inizializzazione del modello basandosi sul numero di feature dopo l'encoding
model = SimpleClassifier(X_train_proc.shape[1])

# --- 6. CONFIGURAZIONE TRAINING ---
# BCEWithLogitsLoss è ideale per classificazioni binarie.
# Passando 'pos_weight', diciamo al modello che sbagliare sulla classe 1 è "più grave" del peso fornito.
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

# Ottimizzatore Adam con learning rate 0.01 per aggiornare i pesi.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Ciclo di addestramento rapido
for epoch in range(100):
    optimizer.zero_grad()      # Reset dei gradienti
    outputs = model(X_train_t) # Forward pass
    loss = criterion(outputs, y_train_t) # Calcolo della loss pesata
    loss.backward()            # Calcolo dei gradienti (Backprop)
    optimizer.step()           # Aggiornamento pesi

print(f"Addestramento completato con successo. Loss finale: {loss.item():.4f}")
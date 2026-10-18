import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

# Simulazione di un dataset di monitoraggio server con dipendenza temporale
def generate_server_data(n_samples=1000):
    """
    Genera dati sintetici che simulano il comportamento di un server nel tempo.
    Le feature (temperatura e carico) presentano un trend crescente e stagionalità.
    Il target 'y' (guasto) è fortemente sbilanciato verso la classe 0 (funzionamento normale).
    """
    np.random.seed(42)
    # Creazione di una sequenza temporale (es. 1000 step consecutivi)
    time = np.linspace(0, 100, n_samples)
    
    # Feature 1: Temperatura. Segue un andamento sinusoidale con rumore e trend crescente.
    temp = np.sin(time) + np.random.normal(0, 0.1, n_samples) + (time * 0.01)
    # Feature 2: Carico CPU. Simile alla temperatura, con parametri leggermente diversi.
    load = np.cos(time) + np.random.normal(0, 0.1, n_samples) + (time * 0.005)
    
    # Calcolo della probabilità di guasto: aumenta all'aumentare di temperatura e carico.
    # Utilizziamo la funzione sigmoid per mappare i valori in un intervallo [0, 1].
    prob = torch.sigmoid(torch.tensor(temp + load - 2.5)).numpy()
    
    # Generazione del target binario: 0 = OK, 1 = Guasto. 
    # Moltiplichiamo per 0.3 per forzare uno sbilanciamento forte (circa 90/10).
    y = (np.random.rand(n_samples) < prob * 0.3).astype(int)
    
    # Organizzazione delle feature in una matrice X di dimensioni (n_samples, 2)
    X = np.stack([temp, load], axis=1)
    return X, y

# Generazione effettiva dei dati
X, y = generate_server_data()
print(f"Distribuzione classi: {np.bincount(y)} (Classe 1 - Guasti: {np.mean(y)*100:.1f}%)")

# Configurazione della convalida incrociata per serie temporali.
# A differenza della K-Fold standard, TimeSeriesSplit assicura che il set di test
# sia sempre cronologicamente successivo al set di addestramento.
tscv = TimeSeriesSplit(n_splits=3)

def train_model(X_train, y_train, X_test, pos_weight=None):
    """
    Gestisce l'intera pipeline di addestramento e valutazione di un singolo fold.
    Parametri:
    - X_train/X_test: matrici delle feature.
    - y_train: target per l'addestramento.
    - pos_weight: peso da assegnare alla classe positiva per bilanciare la loss.
    """
    
    # Inizializzazione dello scaler. Fondamentale per modelli lineari e reti neurali.
    scaler = StandardScaler()
    
    # ADDESTRAMENTO DELLO SCALER: Solo sui dati di train.
    # Questo evita il "Data Leakage", ovvero che informazioni statistiche del futuro (test)
    # influenzino la normalizzazione dei dati del passato (train).
    X_train = scaler.fit_transform(X_train)
    
    # APPLICAZIONE DELLO SCALER: Trasformiamo il test set usando i parametri del train.
    X_test = scaler.transform(X_test)
    
    # Conversione dei dati in Tensori PyTorch per l'elaborazione sulla rete neurale.
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32).view(-1, 1) # Reshape in colonna (N, 1)
    X_ts = torch.tensor(X_test, dtype=torch.float32)
    
    # Definizione del modello: un singolo layer lineare che proietta 2 input su 1 output.
    # In PyTorch, nn.Linear(2, 1) implementa l'operazione y = xW^T + b.
    model = nn.Linear(2, 1)
    
    # Configurazione della funzione di perdita (Loss Function).
    # BCEWithLogitsLoss combina un layer Sigmoid e la Binary Cross Entropy,
    # risultando più stabile numericamente rispetto all'uso separato.
    # Il parametro pos_weight permette di pesare maggiormente gli errori sulla classe 1.
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Ottimizzatore Adam: adatta il learning rate per ogni parametro durante la discesa del gradiente.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    # Ciclo di addestramento (200 epoche)
    for epoch in range(200):
        optimizer.zero_grad()      # Reset dei gradienti accumulati
        outputs = model(X_tr)      # Passaggio in avanti (Forward pass)
        loss = criterion(outputs, y_tr) # Calcolo dell'errore
        loss.backward()            # Propagazione all'indietro (Backpropagation)
        optimizer.step()           # Aggiornamento dei pesi del modello
    
    # Fase di predizione sul set di test
    with torch.no_grad(): # Disabilita il calcolo dei gradienti per risparmiare memoria
        y_pred_logits = model(X_ts) # Otteniamo i valori "crudi" (logits)
        # Convertiamo i logits in probabilità (sigmoid) e poi in classi (soglia 0.5)
        y_pred = (torch.sigmoid(y_pred_logits) > 0.5).int().numpy()
    
    return y_pred

# Liste per memorizzare i risultati dei vari fold di convalida
f1_default_list = []
f1_balanced_list = []

# Ciclo principale di valutazione cross-validata
for train_index, test_index in tscv.split(X):
    # Suddivisione dei dati in base agli indici temporali generati da TimeSeriesSplit
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # SCENARIO A: Modello Standard senza bilanciamento.
    # Impostiamo pos_weight a 1.0 (nessun effetto sulla loss).
    y_pred_def = train_model(X_train, y_train, X_test, pos_weight=torch.tensor([1.0]))
    f1_default_list.append(f1_score(y_test, y_pred_def))
    
    # SCENARIO B: Modello Bilanciato.
    # Calcoliamo il peso della classe positiva come il rapporto tra esempi negativi e positivi.
    # Se abbiamo 90 negativi e 10 positivi, il peso sarà 9.0 (la classe 1 vale 9 volte la 0).
    negativi = len(y_train) - sum(y_train)
    positivi = sum(y_train)
    w_balanced = torch.tensor([negativi / positivi])
    
    y_pred_bal = train_model(X_train, y_train, X_test, pos_weight=w_balanced)
    f1_balanced_list.append(f1_score(y_test, y_pred_bal))

# Report finale dei risultati medi
print(f"\nConfronto prestazioni medie (F1-Score):")
print(f"Modello Default (non pesato):  {np.mean(f1_default_list):.4f}")
print(f"Modello Balanced (pesato):      {np.mean(f1_balanced_list):.4f}")
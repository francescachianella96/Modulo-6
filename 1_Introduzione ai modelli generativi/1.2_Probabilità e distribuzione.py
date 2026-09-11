import os

# Configurazione dell'ambiente di esecuzione.
# Impostiamo PyTorch come motore di calcolo (backend) per Keras prima di importare la libreria stessa.
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import ops, layers
import numpy as np
import matplotlib.pyplot as plt

# Verifica della configurazione: stampa la versione di Keras e il backend effettivamente in uso.
print(f"Versione Keras: {keras.__version__} | Backend: {keras.backend.backend()}")

# --- GENERAZIONE DEI DATI REALI ---
# In questa sezione creiamo un dataset sintetico che funge da "verità di base".
# Ipotizziamo che i nostri dati seguano una distribuzione normale con una media (mu) specifica.
true_mu = 5.0
# Generiamo 1000 campioni estratti da una distribuzione normale centrata in 5.0 con deviazione standard 1.0.
# Il risultato è un array NumPy di forma (1000, 1) di tipo float32, compatibile con i tensori di Deep Learning.
data = np.random.normal(true_mu, 1.0, (1000, 1)).astype("float32")

# --- MODELLO DI MASSIMA VEROSIMIGLIANZA (MLE - Maximum Likelihood Estimation) ---
# L'obiettivo è costruire un modello che "impari" il parametro della media dai dati forniti.

# Definiamo l'ingresso del modello: un singolo valore numerico per ogni record.
inputs = keras.Input(shape=(1,))

# Creiamo un livello Dense con un solo neurone. 
# Disabilitiamo il bias (use_bias=False) perché vogliamo che il peso del neurone rappresenti direttamente la nostra media stimata.
# Inizializziamo il peso a zero.
predicted_mu = layers.Dense(1, use_bias=False, kernel_initializer="zeros")(inputs)

# Colleghiamo ingressi e uscite per istanziare l'oggetto Model di Keras.
mle_model = keras.Model(inputs=inputs, outputs=predicted_mu)

# Definizione della funzione di perdita (Loss Function).
# Per una distribuzione normale con varianza nota e pari a 1, minimizzare la Negative Log-Likelihood (NLL)
# equivale a minimizzare l'errore quadratico medio (MSE).
def nll_loss(y_true, y_pred):
    # Calcola lo scarto tra il dato reale e quello predetto, lo eleva al quadrato e ne fa la media.
    # Usiamo 'ops' di Keras per garantire la compatibilità con il backend PyTorch.
    return ops.mean(ops.square(y_true - y_pred))

# Compiliamo il modello specificando l'ottimizzatore Adam e la nostra funzione di perdita personalizzata.
mle_model.compile(optimizer="adam", loss=nll_loss)

# Fase di addestramento: il modello analizza i dati per regolare il suo peso interno (la media stimata).
# Poiché vogliamo che impari la media dei dati stessi, usiamo 'data' sia come input che come target.
ones = np.ones((1000, 1)).astype("float32")

print("Inizio ottimizzazione dei parametri tramite MLE...")
mle_model.fit(ones, data, epochs=200, verbose=0)

# Recupero del parametro ottimizzato.
# Accediamo ai pesi del livello Dense (il secondo livello del modello, indice 1).
# Convertiamo il valore dal formato tensoriale del backend (PyTorch) a un array NumPy per la stampa.
estimated_mu = ops.convert_to_numpy(mle_model.layers[1].get_weights()[0][0][0])

print(f"Media Reale definita: {true_mu}")
print(f"Media Stimata dal modello dopo l'ottimizzazione: {estimated_mu:.4f}")

# --- ANALISI DELLA MALEDIZIONE DELLA DIMENSIONALITÀ ---
# Questa funzione illustra come la distanza tra i punti cambi drasticamente all'aumentare delle dimensioni dello spazio.
def check_dimensionality_curse(dims):
    avg_distances = []
    
    for d in dims:
        # Generiamo 100 punti casuali distribuiti uniformemente tra 0 e 1 in uno spazio a 'd' dimensioni.
        points = np.random.uniform(0, 1, (100, d)).astype("float32")
        
        # Calcolo delle distanze euclidee a coppie.
        # Espandiamo le dimensioni per sfruttare il broadcasting: trasformiamo i punti in una griglia di differenze.
        # diff[i][j] conterrà la differenza vettoriale tra il punto i e il punto j.
        diff = ops.expand_dims(points, 1) - ops.expand_dims(points, 0)
        
        # Calcoliamo la norma euclidea: radice quadrata della somma dei quadrati delle differenze lungo l'asse delle dimensioni.
        dist_matrix = ops.sqrt(ops.sum(ops.square(diff), axis=-1))
        
        # Calcoliamo la distanza media tra tutti i punti nel dataset generato.
        # Nota: questo include anche la distanza di ogni punto da se stesso (che è 0).
        avg_dist = ops.mean(dist_matrix)
        
        # Aggiungiamo il risultato alla lista convertendolo in formato numerico standard.
        avg_distances.append(ops.convert_to_numpy(avg_dist))
    
    return avg_distances

# Definiamo una serie di dimensioni crescenti da testare.
dimensions = [1, 2, 10, 50, 100, 500, 1000]
distances = check_dimensionality_curse(dimensions)

# --- VISUALIZZAZIONE GRAFICA DEI RISULTATI ---
# Creiamo un grafico per mostrare come la distanza media cresca con la complessità dimensionale.
plt.figure(figsize=(10, 5))
plt.plot(dimensions, distances, marker='o', color='crimson', linewidth=2)

plt.title("Impatto della Dimensionalità sulla Distanza Media tra i Punti", fontsize=14)
plt.xlabel("Numero di Dimensioni (d)", fontsize=12)
plt.ylabel("Distanza Media Calcolata", fontsize=12)

# Aggiungiamo una griglia per facilitare la lettura dei valori.
plt.grid(True, linestyle='--', alpha=0.7)

# Visualizziamo il grafico finale.
plt.show()
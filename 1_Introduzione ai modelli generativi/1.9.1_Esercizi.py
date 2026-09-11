# Esercizio
# Implementa un autoencoder per CIFAR-10 con un bottleneck di 256 neuroni. 
# Confronta visivamente la qualità della ricostruzione utilizzando prima la funzione di perdita mse e poi mae per 10 epoche. 
# Suggerimento: usa plt.subplot per mostrare input, ricostruzione MSE e ricostruzione MAE in un'unica griglia

import os

# Per garantire la massima compatibilità e sfruttare le ottimizzazioni di calcolo più recenti,
# impostiamo il backend di Keras su "torch" (PyTorch) prima di importare la libreria.
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------------------
# FASE 1: ACQUISIZIONE E PREPARAZIONE DEL DATASET CIFAR-10
# -----------------------------------------------------------------------------------------

# 1. Importazione dataset già diviso in train e test
(x_train, _), (x_test, _) = keras.datasets.cifar10.load_data()

# 2. Normalizzazione
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 3. Calcolo dimensione totale dell'input
input_dim = 32 * 32 * 3

# 4. Flattening del vettore (da 3D a 1D)
x_train = x_train.reshape((len(x_train), input_dim))
x_test = x_test.reshape((len(x_test), input_dim))

# -----------------------------------------------------------------------------------------
# FASE 2: PROGETTAZIONE DELL'ARCHITETTURA DELL'AUTOENCODER
# -----------------------------------------------------------------------------------------

def build_autoencoder():
    """
    Costruisce un modello Autoencoder utilizzando la Functional API di Keras.
    L'architettura segue una struttura a 'imbuto' per comprimere e poi ricostruire i dati.
    """
    # Punto di ingresso per il flusso di dati (tensor di input).
    input_img = keras.Input(shape=(input_dim,))
    
    # --- PARTE 1: ENCODER (Compressione) ---
    hidden_enc = layers.Dense(512, activation="relu")(input_img)
    
    # bottleneck a 256 neuroni
    bottleneck = layers.Dense(256, activation="relu")(hidden_enc)
    
    # --- PARTE 2: DECODER (Ricostruzione) ---
    hidden_dec = layers.Dense(512, activation="relu")(bottleneck)
    
    # Output
    output_img = layers.Dense(input_dim, activation="sigmoid")(hidden_dec)
    
    # unione di encoder e decoder
    return keras.Model(input_img, output_img)

# -----------------------------------------------------------------------------------------
# FASE 3: CONFIGURAZIONE E ADDESTRAMENTO CON MSE (MEAN SQUARED ERROR)
# -----------------------------------------------------------------------------------------

print("Operazione: Addestramento ottimizzato tramite MSE")

model_mse = build_autoencoder()
model_mse.compile(optimizer="adam", loss="mse")

# Addestriamo il modello passando x_train sia come input che come target desiderato.
model_mse.fit(
    x_train, x_train,
    epochs=10, 
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test),
    verbose=1
)

# Generiamo le immagini ricostruite dal dataset di test per valutare le prestazioni.
decoded_mse = model_mse.predict(x_test)

# -----------------------------------------------------------------------------------------
# FASE 4: CONFIGURAZIONE E ADDESTRAMENTO CON MAE (MEAN ABSOLUTE ERROR)
# -----------------------------------------------------------------------------------------

print("\nOperazione: Addestramento ottimizzato tramite MAE")

model_mae = build_autoencoder()
model_mae.compile(optimizer="adam", loss="mae")

model_mae.fit(
    x_train, x_train,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test),
    verbose=1
)

# Otteniamo la seconda serie di ricostruzioni per il confronto visivo.
decoded_mae = model_mae.predict(x_test)

# -----------------------------------------------------------------------------------------
# FASE 5: VISUALIZZAZIONE COMPARATIVA E ANALISI DEI RISULTATI
# -----------------------------------------------------------------------------------------

# Selezione di 8 immagini campioni
num_samples = 8
plt.figure(figsize=(20, 10))

for i in range(num_samples):
    # RIGA 1: Immagini originali dal dataset di test.
    # Convertiamo il vettore piatto di nuovo in formato (32, 32, 3) per Matplotlib.
    ax = plt.subplot(3, num_samples, i + 1)
    plt.imshow(x_test[i].reshape(32, 32, 3))
    plt.title("Immagine Originale")
    plt.axis("off")

    # RIGA 2: Immagini ricostruite dal modello addestrato con perdita MSE.
    ax = plt.subplot(3, num_samples, i + 1 + num_samples)
    plt.imshow(decoded_mse[i].reshape(32, 32, 3))
    plt.title("Ricostruzione MSE")
    plt.axis("off")

    # RIGA 3: Immagini ricostruite dal modello addestrato con perdita MAE.
    ax = plt.subplot(3, num_samples, i + 1 + 2 * num_samples)
    plt.imshow(decoded_mae[i].reshape(32, 32, 3))
    plt.title("Ricostruzione MAE")
    plt.axis("off")

plt.tight_layout()
plt.show()

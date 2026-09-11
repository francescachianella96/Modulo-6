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

# Utilizziamo le utility integrate di Keras per scaricare il dataset CIFAR-10.
# Questo dataset contiene 60.000 immagini a colori 32x32, divise in 10 categorie.
# Carichiamo solo le immagini (x) ignorando le etichette (_), poiché l'autoencoder 
# apprende a ricostruire l'input stesso, agendo quindi in modalità non supervisionata.
(x_train, _), (x_test, _) = keras.datasets.cifar10.load_data()

# Le immagini originali hanno pixel con valori tra 0 e 255 (formato unit8).
# Per facilitare la convergenza dell'algoritmo di addestramento, convertiamo i dati 
# in virgola mobile (float32) e normalizziamo i valori nell'intervallo [0, 1].
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Calcoliamo la dimensione totale dell'input: 32 pixel di altezza * 32 di larghezza * 3 canali RGB.
# Il risultato è 3072, che rappresenterà il numero di neuroni dello strato di input.
input_dim = 32 * 32 * 3

# Poiché utilizzeremo strati "Dense" (completamente connessi), dobbiamo trasformare 
# ogni immagine da una matrice 3D (32, 32, 3) in un vettore piatto (1D) di 3072 elementi.
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
    # Definiamo il punto di ingresso per il flusso di dati (tensor di input).
    input_img = keras.Input(shape=(input_dim,))
    
    # --- PARTE 1: ENCODER (Compressione) ---
    # Lo strato nascosto riduce la dimensionalità a 512 neuroni.
    # L'attivazione 'relu' introduce non-linearità per catturare pattern complessi.
    hidden_enc = layers.Dense(512, activation="relu")(input_img)
    
    # Il 'bottleneck' (collo di bottiglia) è lo strato centrale con soli 128 neuroni.
    # Qui il modello è costretto a creare una rappresentazione compressa dell'immagine.
    bottleneck = layers.Dense(128, activation="relu")(hidden_enc)
    
    # --- PARTE 2: DECODER (Ricostruzione) ---
    # Iniziamo la riespansione dei dati portandoli nuovamente a 512 neuroni.
    hidden_dec = layers.Dense(512, activation="relu")(bottleneck)
    
    # Lo strato di output riporta i dati alla dimensione originale (3072).
    # Usiamo la funzione 'sigmoid' per garantire che i valori dei pixel siano tra 0 e 1,
    # coerentemente con la normalizzazione fatta sui dati di input.
    output_img = layers.Dense(input_dim, activation="sigmoid")(hidden_dec)
    
    # Uniamo Encoder e Decoder in un unico oggetto Modello.
    return keras.Model(input_img, output_img)

# -----------------------------------------------------------------------------------------
# FASE 3: CONFIGURAZIONE E ADDESTRAMENTO CON MSE (MEAN SQUARED ERROR)
# -----------------------------------------------------------------------------------------

print("Operazione: Addestramento ottimizzato tramite MSE")

# L'errore quadratico medio (MSE) penalizza gli errori più grandi in modo esponenziale.
# Questo porta il modello a cercare di evitare grandi discrepanze, producendo spesso 
# ricostruzioni che risultano leggermente "sfocate" ma bilanciate.
model_mse = build_autoencoder()
model_mse.compile(optimizer="adam", loss="mse")

# Addestriamo il modello passando x_train sia come input che come target desiderato.
model_mse.fit(
    x_train, x_train,
    epochs=50, 
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

# L'errore assoluto medio (MAE) tratta tutti gli errori allo stesso modo, senza elevarli al quadrato.
# È intrinsecamente più robusto verso i dati anomali (outlier) e tende a prestare più
# attenzione ai dettagli puntuali rispetto al comportamento globale del segnale.
model_mae = build_autoencoder()
model_mae.compile(optimizer="adam", loss="mae")

model_mae.fit(
    x_train, x_train,
    epochs=50,
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

# Selezioniamo il numero di campioni da visualizzare (8 immagini).
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

# Ottimizziamo la spaziatura tra i grafici e mostriamo la finestra.
plt.tight_layout()
plt.show()
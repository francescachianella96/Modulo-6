# Esercizio: crea un autoencoder per insegnare al modello a generare i capi di abbigliamento del feshion mnist. 
# Utilizza il codice visto modificando il numero di neuroni per riuscire a gestire correttamente l'aumento di complessità delle immagini e commenta il risultato ottenuto

import os

# Configurazione del backend di Keras, importando PyTorch come motore di calcolo prima dell'importazione di Keras
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers, ops
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. PREPARAZIONE DEL DATASET
# =============================================================================

# Caricamento del dataset Fashion MNIST. 
# Poiché l'autoencoder è un modello di apprendimento non supervisionato (ricostruisce l'input), 
# ignoriamo le etichette di classificazione (indicate con _).
(x_train, _), (x_test, _) = keras.datasets.fashion_mnist.load_data()

# Normalizzazione dei dati (pixel originari tra 0 e 255) per favorire la convergenza dell'apprendimento
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Operazione di 'Flattening': trasformiamo ogni immagine 28x28 in un vettore lineare di 784 elementi per gli strati Dense che accettano layer piatti
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))

# =============================================================================
# 2. DEFINIZIONE DELL'ARCHITETTURA DELL'AUTOENCODER
# =============================================================================

input_dim = 784    # Dimensione originale (28*28)
latent_dim = 128   # Dimensione dello spazio latente, aumentato rispetto a MNST per cogliere meglio la profondità delle immagini

# --- ENCODER ---
# L'Encoder riceve l'immagine piatta e ne riduce progressivamente la dimensionalità.
inputs = keras.Input(shape=(input_dim,), name="Input_Immagine")
x = layers.Dense(256, activation="relu")(inputs)

# Strato bottleneck
latent = layers.Dense(latent_dim, activation="relu", name="Bottleneck")(x)
# è possibile impostare nel bottleneck activation="None" per evitare di schiacciare o azzerare i valori negativi prima di trasmettere al decoder  

# --- DECODER ---
# Il Decoder riceve la rappresentazione compressa e cerca di espanderla per tornare alla forma originale.
x = layers.Dense(256, activation="relu")(latent)

# Strato di output con attivazione 'sigmoid' per garantire che i valori dei pixel ricostruiti siano compresi 
# tra 0 e 1, coerentemente con i dati di input normalizzati.
outputs = layers.Dense(input_dim, activation="sigmoid", name="Ricostruzione")(x)

# Creazione del modello completo (Autoencoder) che collega input e output.
autoencoder = keras.Model(inputs=inputs, outputs=outputs, name="Autoencoder_Hourglass")

# =============================================================================
# 3. COMPILAZIONE E ADDESTRAMENTO
# =============================================================================

# Utilizziamo l'ottimizzatore Adam e la funzione di perdita MSE (Mean Squared Error).
# L'obiettivo è minimizzare la differenza quadratica tra l'immagine originale e quella ricostruita.
autoencoder.compile(optimizer="adam", loss="mse")

print("Inizio fase di addestramento: il modello imparerà a ricostruire le immagini minimizzando l'errore...")
# Durante il fit, passiamo x_train sia come input che come target, poiché vogliamo che l'output sia uguale all'input.
autoencoder.fit(
    x_train, 
    x_train, 
    epochs=20, 
    batch_size=256, 
    validation_data=(x_test, x_test), 
    verbose=1
)

# =============================================================================
# 4. VALIDAZIONE, VISUALIZZAZIONE DEI RISULTATI E COMMENTO
# =============================================================================

# Utilizziamo il modello addestrato per ricostruire le prime 10 immagini del set di test.
decoded_imgs = autoencoder.predict(x_test[:10], verbose=0)

# Conversione dei tensori di Keras con Pytorch in vettori numpy per la visualizzaione con matplotlib
# attraverso la funzione ops.convert_to_numpy, che gestisce l'eventuale spostamento da memoria GPU a CPU.
decoded_imgs_np = ops.convert_to_numpy(decoded_imgs)
x_test_np = x_test[:10]

# Creazione della griglia di visualizzazione per confrontare input e output.
plt.figure(figsize=(20, 4))
for i in range(10):
    # Visualizzazione dell'immagine originale
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(x_test_np[i].reshape(28, 28), cmap="gray")
    plt.title("Originale")
    plt.axis("off")

    # Visualizzazione dell'immagine ricostruita dal decoder
    ax = plt.subplot(2, 10, i + 11)
    plt.imshow(decoded_imgs_np[i].reshape(28, 28), cmap="gray")
    plt.title("Ricostruito")
    plt.axis("off")

plt.show()

# Commento finale: il datasat utilizzato, rispetto a MNSIT classico, è più complesso e più sfaccettato, per questo ho creato un autoencoder con più neuroni e nell'addestramento del modello ho inserito 20 e non 10 epoche
# Nonostante questo l'immagine ricostruita dal modello si presenta comunque sfocata. Il modello cattura correttamente la forma dei capi, ma non i dettagli.  
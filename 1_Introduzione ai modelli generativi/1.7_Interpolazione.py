import os

# Configurazione del backend di Keras.
# Viene impostato PyTorch come motore di calcolo sottostante prima di importare Keras.
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers, ops
import numpy as np
import matplotlib.pyplot as plt

# --- 1. PREPARAZIONE DEI DATI ---
# Carichiamo il dataset MNIST composto da cifre scritte a mano.
# Ignoriamo le etichette (y_train, y_test) poiché in un autoencoder l'obiettivo è 
# ricostruire l'input stesso, rendendolo un compito di apprendimento non supervisionato.
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

# Normalizzazione dei dati nel range [0, 1] per facilitare la convergenza del modello.
# Le immagini originali hanno valori di pixel tra 0 e 255.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Appiattimento (reshaping) delle immagini da matrici 28x28 a vettori di 784 elementi.
# Questo è necessario perché utilizzeremo strati densi (Fully Connected).
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))

# --- 2. ARCHITETTURA DEL MODELLO (Autoencoder) ---
# Un autoencoder è composto da due parti principali: un Encoder e un Decoder.
# L'obiettivo è comprimere l'informazione in uno spazio ridotto e poi ricostruirla.

# Dimensione dello spazio latente: rappresenta il numero di variabili che il modello
# userà per descrivere le caratteristiche essenziali di una cifra.
latent_dim = 32 

# Definizione dell'ENCODER: mappa l'input (784 pixel) in una rappresentazione compressa (Z).
# L'input riceve il vettore appiattito dell'immagine.
inputs = keras.Input(shape=(784,))
# Strato intermedio per estrarre caratteristiche di alto livello.
x = layers.Dense(128, activation="relu")(inputs)
# Lo strato "Latent_Space" riduce ulteriormente la dimensionalità a 32.
latent_repr = layers.Dense(latent_dim, activation="relu", name="Latent_Space")(x)
# Creazione del sottomodello Encoder.
encoder = keras.Model(inputs, latent_repr, name="Encoder")

# Definizione del DECODER: mappa la rappresentazione compressa (Z) di nuovo nello spazio originale.
# L'input del decoder è il vettore latente prodotto dall'encoder.
latent_inputs = keras.Input(shape=(latent_dim,))
# Strato intermedio che inizia a ricostruire la complessità dell'immagine.
x = layers.Dense(128, activation="relu")(latent_inputs)
# Lo strato finale riporta la dimensione a 784 pixel. 
# Si usa la "sigmoid" perché i pixel normalizzati sono compresi tra 0 e 1.
outputs = layers.Dense(784, activation="sigmoid")(x)
# Creazione del sottomodello Decoder.
decoder = keras.Model(latent_inputs, outputs, name="Decoder")

# AUTOENCODER COMPLETO: Unione di Encoder e Decoder.
# L'output dell'encoder viene passato come input al decoder.
autoencoder = keras.Model(inputs, decoder(encoder(inputs)))

# Compilazione del modello.
# Si usa 'adam' come ottimizzatore e l'errore quadratico medio (MSE) come funzione di perdita,
# che misura la differenza pixel per pixel tra l'immagine originale e quella ricostruita.
autoencoder.compile(optimizer="adam", loss="mse")

# Addestramento del modello.
# Si noti che l'input (x_train) e il target (x_train) sono identici.
# Il modello impara a comprimere i dati cercando di perdere meno informazione possibile.
print("Inizio addestramento: estrazione delle caratteristiche salienti...")
autoencoder.fit(x_train, x_train, epochs=5, batch_size=256, verbose=0)

# --- 3. INTERPOLAZIONE E MORPHING ---
# Questa funzione permette di passare fluidamente da una cifra all'altra esplorando 
# lo spazio latente creato dall'autoencoder.
def interpolate_digits(digit_a_idx, digit_b_idx, steps=10):
    # Selezione di due immagini dal test set.
    img_a = x_test[digit_a_idx : digit_a_idx + 1]
    img_b = x_test[digit_b_idx : digit_b_idx + 1]
    
    # L'encoder trasforma le immagini reali in vettori nel "linguaggio" del modello (z).
    z_a = encoder.predict(img_a, verbose=0)
    z_b = encoder.predict(img_b, verbose=0)
    
    # Interpolazione Lineare (LERP): calcoliamo punti intermedi tra i due vettori latenti.
    # z(t) = (1-t) * z_a + t * z_b, dove t varia da 0 a 1.
    t_values = np.linspace(0, 1, steps)
    interpolated_points = [(1 - t) * z_a + t * z_b for t in t_values]
    interpolated_points = np.vstack(interpolated_points).astype("float32")
    
    # Il Decoder prende questi punti artificiali e li trasforma in nuove immagini visibili.
    reconstructions = decoder.predict(interpolated_points, verbose=0)
    # Conversione da tensore PyTorch a array NumPy per la visualizzazione.
    return ops.convert_to_numpy(reconstructions)

# Creazione della sequenza di trasformazione tra due cifre.
# Ad esempio, osserviamo come un '3' si trasforma gradualmente in un'altra cifra.
morph_sequence = interpolate_digits(18, 0, steps=12)

# --- 4. VISUALIZZAZIONE DEI RISULTATI ---
# Creiamo una striscia di immagini che mostra l'intera evoluzione del morphing.
plt.figure(figsize=(15, 3))
for i, img in enumerate(morph_sequence):
    ax = plt.subplot(1, 12, i + 1)
    # Ogni vettore di 784 pixel viene riportato alla forma 28x28 per la visualizzazione.
    plt.imshow(img.reshape(28, 28), cmap="gray")
    plt.axis("off")
    if i == 0: plt.title("Inizio")
    if i == 11: plt.title("Fine")

plt.suptitle("Morphing Semantico: Trasformazione fluida nello Spazio Latente")
plt.show()
# Esercizio - utilizzando l'autoencoder addestrato, selezione due immagini dal test set (un 4 e un 8) ed estrai i loro vettori latenti tramite encoder
# Genera 10 vettori intermedi latenti tramite encoder e genera 10 vettori intermedi usando l'interpolazione lineare e usa il decoder per visualizzare la trasformazione
# Cosa noti nel punto centrale (t=0.5)?

import os

# Configurazione backend Keras
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers, ops
import numpy as np
import matplotlib.pyplot as plt

# --- 1. PREPARAZIONE DEI DATI ---
# Mantenimento delle etichette y_test per individuare esattamente un 4 e un 8
(x_train, _), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalizzazione dei dati 
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Appiattimento delle immagini da matrici a vettori
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))

# --- 2. ARCHITETTURA DEL MODELLO (Autoencoder) ---

latent_dim = 32 

# Encoder
inputs = keras.Input(shape=(784,))
x = layers.Dense(128, activation="relu")(inputs)
latent_repr = layers.Dense(latent_dim, activation="relu", name="Latent_Space")(x)
encoder = keras.Model(inputs, latent_repr, name="Encoder")

# Decoder
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(128, activation="relu")(latent_inputs) 
# Uso della funzione sigmoide nello stato finale
outputs = layers.Dense(784, activation="sigmoid")(x)
decoder = keras.Model(latent_inputs, outputs, name="Decoder")

# AUTOENCODER COMPLETO: Unione di Encoder e Decoder
autoencoder = keras.Model(inputs, decoder(encoder(inputs)))

# Compilazione del modello
autoencoder.compile(optimizer="adam", loss="mse")

# Addestramento del modello.
print("Inizio addestramento: estrazione delle caratteristiche salienti...")
autoencoder.fit(x_train, x_train, epochs=5, batch_size=256, verbose=0)

# --- 3. INTERPOLAZIONE E MORPHING ---
# Ricerca degli indici corrispondenti alle cifre 4 e 8 nel test set
idx_4 = np.where(y_test == 4)[0][0]
idx_8 = np.where(y_test == 8)[0][0]

# Estrazione dei campioni e dei rispettivi vettori latenti tramite encoder
img_4 = x_test[idx_4 : idx_4 + 1]
img_8 = x_test[idx_8 : idx_8 + 1]

z_4 = encoder.predict(img_4, verbose=0)
z_8 = encoder.predict(img_8, verbose=0)


# --- 4. INTERPOLAZIONE LINEARE (10 VETTORI INTERMEDI) ---
steps = 10
t_values = np.linspace(0, 1, steps)

# Generazione dei 10 vettori intermedi nello spazio latente: z(t) = (1-t)*z_4 + t*z_8
z_interpolated = [(1 - t) * z_4 + t * z_8 for t in t_values]
z_interpolated = np.vstack(z_interpolated).astype("float32")

# Generazione delle immagini ricostruite tramite decoder
reconstructions = decoder.predict(z_interpolated, verbose=0)
reconstructions = ops.convert_to_numpy(reconstructions)


# --- 5. VISUALIZZAZIONE DEI RISULTATI ---
# Creiamo una striscia di immagini che mostra l'intera evoluzione del morphing.
plt.figure(figsize=(15, 3))
for i, img in enumerate(reconstructions):
    ax = plt.subplot(1, steps, i + 1)
    plt.imshow(img.reshape(28, 28), cmap="gray")
    plt.axis("off")
    t_val = t_values[i]
    if i == 0:
        plt.title(f"Start (4)\nt={t_val:.2f}")
    elif i == steps - 1:
        plt.title(f"End (8)\nt={t_val:.2f}")
    elif np.isclose(t_val, 0.5, atol=0.06):
        plt.title(f"Punto Medio\nt={t_val:.2f}", color="red")
    else:
        plt.title(f"t={t_val:.2f}")

plt.suptitle("Interpolazione Lineare nello Spazio Latente (4 -> 8)", y=1.05)
plt.show()
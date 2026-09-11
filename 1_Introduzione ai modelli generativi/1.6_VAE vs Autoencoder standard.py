import os

# 1. SETUP DELL'AMBIENTE DI CALCOLO
# Impostiamo il backend di Keras su PyTorch. Questa configurazione deve avvenire
# prima di importare Keras per garantire che il motore computazionale sia quello desiderato.
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers, ops
import numpy as np
import matplotlib.pyplot as plt

# --- 2. PREPARAZIONE DEL DATASET MNIST ---
# Il dataset MNIST contiene 70.000 immagini in scala di grigi di cifre (0-9) da 28x28 pixel.
print("Caricamento dataset MNIST...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalizzazione: I valori dei pixel (0-255) vengono portati nel range [0, 1].
# Questo facilita la convergenza dell'ottimizzatore durante l'addestramento.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Appiattimento (Flattening): Trasformiamo la matrice 28x28 in un vettore monodimensionale di 784 elementi.
# Questo è necessario perché utilizzeremo layer densi (fully connected) che accettano vettori.
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))

# --- 3. ARCHITETTURA: AUTOENCODER STANDARD (AE) ---
# L'Autoencoder "comprime" l'input in uno spazio a bassa dimensionalità e poi tenta di ricostruirlo.
# Non ha vincoli sulla forma dello spazio latente, che spesso risulta disorganizzato.

# Input: Riceve i 784 pixel dell'immagine appiattita.
inputs_ae = keras.Input(shape=(784,))

# Encoder: Riduce la dimensionalità a 64 e poi a 2 (spazio latente).
encoded_ae = layers.Dense(64, activation="relu")(inputs_ae)  
latent_ae = layers.Dense(2, name="Latent_Space_AE")(encoded_ae) 

# Decoder: Prende il punto nel piano 2D e cerca di rigenerare i 784 pixel originali.
decoded_ae = layers.Dense(64, activation="relu")(latent_ae)
outputs_ae = layers.Dense(784, activation="sigmoid")(decoded_ae) 

# Modello AE: Collega input e output.
model_ae = keras.Model(inputs_ae, outputs_ae, name="Autoencoder_Standard")
model_ae.compile(optimizer="adam", loss="mse")

# --- 4. ARCHITETTURA: VARIATIONAL AUTOENCODER (VAE) ---
# A differenza dell'AE, il VAE mappa l'input in una distribuzione di probabilità (media e varianza).
# Questo costringe lo spazio latente a essere continuo e organizzato (simile a una Gaussiana).

# Layer personalizzato per la KL Divergence.
# In Keras 3, aggiungere una loss interna richiede un layer che utilizzi self.add_loss().
class KLLossLayer(layers.Layer):
    """
    Questo layer calcola la divergenza di Kullback-Leibler, che penalizza
    le distribuzioni latenti che si allontanano troppo da una normale standard (0, 1).
    """
    def call(self, inputs):
        z_mean, z_log_var = inputs
        # Formula matematica: -0.5 * sum(1 + log_var - mean^2 - exp(log_var))
        # Questa perdita agisce come regolarizzatore dello spazio latente.
        loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
        loss = ops.mean(ops.sum(loss, axis=1)) 
        self.add_loss(loss) 
        return inputs 

# Encoder VAE: Produce due vettori, media (mu) e log-varianza (log_var).
inputs_vae = keras.Input(shape=(784,))
h_vae = layers.Dense(64, activation="relu")(inputs_vae)
z_mean = layers.Dense(2, name="z_mean")(h_vae)
z_log_var = layers.Dense(2, name="z_log_var")(h_vae)

# Applichiamo il layer della Loss KL per registrare la regolarizzazione nel grafo del modello.
z_mean, z_log_var = KLLossLayer()([z_mean, z_log_var])

# Reparameterization Trick: Permette la backpropagation attraverso un campionamento stocastico.
# Invece di campionare direttamente da N(mu, sigma), campioniamo epsilon da N(0, 1) 
# e calcoliamo: z = mu + sigma * epsilon.
def sampling(args):
    mean, log_var = args
    epsilon = keras.random.normal(shape=ops.shape(mean))
    return mean + ops.exp(0.5 * log_var) * epsilon

# Il layer Lambda esegue la funzione di campionamento durante il passaggio in avanti.
z = layers.Lambda(sampling, output_shape=(2,))([z_mean, z_log_var])

# Decoder VAE: Ricostruisce l'immagine a partire dal punto campionato 'z'.
dec_h = layers.Dense(64, activation="relu")(z)
outputs_vae = layers.Dense(784, activation="sigmoid")(dec_h)

# Modello VAE: La loss totale sarà la somma di MSE (ricostruzione) + KL (regolarizzazione).
model_vae = keras.Model(inputs_vae, outputs_vae, name="VAE")
model_vae.compile(optimizer="adam", loss="mse")

# --- 5. FASE DI ADDESTRAMENTO ---
# Entrambi i modelli cercano di minimizzare l'errore tra input e output (auto-supervisione).
print("Inizio addestramento (10 epoche)...")
epochs = 10
batch_size = 128

# Addestriamo l'AE
print("Allenamento Autoencoder Standard in corso...")
model_ae.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, verbose=0)

# Addestriamo il VAE
print("Allenamento Variational Autoencoder in corso...")
model_vae.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, verbose=0)

# --- 6. ANALISI E VISUALIZZAZIONE DELLO SPAZIO LATENTE ---
# Estraiamo solo la parte "Encoder" di entrambi i modelli per vedere come raggruppano i dati.
print("Generazione dei grafici dello spazio latente...")

# Sottomodelli per l'estrazione delle coordinate 2D
encoder_ae = keras.Model(inputs_ae, latent_ae)
encoder_vae = keras.Model(inputs_vae, z_mean) # Per il VAE usiamo la media come coordinata rapida

# Trasformiamo i dati di test (mai visti prima) in coordinate 2D
z_ae = ops.convert_to_numpy(encoder_ae.predict(x_test, verbose=0))
z_vae = ops.convert_to_numpy(encoder_vae.predict(x_test, verbose=0))

# Creazione della figura comparativa
plt.figure(figsize=(14, 6))

# Grafico 1: Autoencoder Standard
# Noterete buchi e cluster molto distanti; lo spazio non è pensato per la generazione.
plt.subplot(1, 2, 1)
plt.scatter(z_ae[:, 0], z_ae[:, 1], c=y_test, cmap="tab10", alpha=0.5, s=5)
plt.colorbar(label="Classe del numero (0-9)")
plt.title("VAE\nDistribuzione continua e regolarizzata")
plt.xlabel("Dimensione Latente 1")
plt.ylabel("Dimensione Latente 2")
plt.grid(True, alpha=0.3)

# Grafico 2: Variational Autoencoder
# Noterete una distribuzione più compatta e centrata; ideale per campionare nuovi dati.
plt.subplot(1, 2, 2)
plt.scatter(z_vae[:, 0], z_vae[:, 1], c=y_test, cmap="tab10", alpha=0.5, s=5)
plt.colorbar(label="Classe del numero (0-9)")
plt.title("Spazio Latente: Autoencoder Standard\nDistribuzione frammentata")
plt.xlabel("Dimensione Latente 1")
plt.ylabel("Dimensione Latente 2")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
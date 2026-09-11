import os

# 1. CONFIGURAZIONE DEL BACKEND
# Impostiamo PyTorch come motore di calcolo per Keras prima di importare la libreria.
# Questa impostazione determina come verranno gestiti i tensori e le operazioni matematiche.
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers, ops
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

# 2. CARICAMENTO E PREPARAZIONE DEL DATASET
# Utilizziamo Fashion MNIST per addestrare un modello "Giudice" che sappia distinguere i capi d'abbigliamento.
# Questo modello servirà poi come base per misurare la qualità di immagini generate.
(x_train, y_train), (x_test, _) = keras.datasets.fashion_mnist.load_data()

# Normalizzazione: trasformiamo i valori dei pixel da [0, 255] a [0, 1] per facilitare la convergenza del modello.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Aggiungiamo la dimensione del canale (grigio) per conformarci allo standard (Batch, Altezza, Larghezza, Canali).
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Le etichette vengono convertite in int32, formato richiesto dalla funzione di perdita per la classificazione.
y_train = y_train.astype("int32")

# 3. DEFINIZIONE DELL'ARCHITETTURA DEL GIUDICE
# Creiamo un modello con la Functional API di Keras che restituisce due diversi tipi di output.
def build_judge():
    inputs = keras.Input(shape=(28, 28, 1))
    
    # Strati convoluzionali per estrarre le caratteristiche spaziali dell'immagine.
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    
    # GlobalAveragePooling riduce l'immagine a un vettore di caratteristiche salienti, 
    # eliminando la dipendenza dalla posizione specifica degli oggetti.
    x = layers.GlobalAveragePooling2D()(x)
    
    # Primo Output: Feature profonde (Vettore di 64 elementi).
    # Queste rappresentano "l'essenza" dell'immagine e verranno usate per il calcolo della FID (Distanza di Fréchet).
    features = layers.Dense(64, activation="relu", name="out_features")(x)
    
    # Secondo Output: Probabilità di classe (10 classi).
    # Rappresenta quanto il modello è sicuro dell'appartenenza a una categoria.
    # Verrà usato per calcolare l'Inception Score (IS).
    probs = layers.Dense(10, activation="softmax", name="out_probs")(features)
    
    # Il modello è configurato per restituire entrambi gli output simultaneamente.
    return keras.Model(inputs=inputs, outputs=[features, probs])

# Istanziamo il modello.
judge = build_judge()

# Compiliamo il modello associando la funzione di perdita solo al ramo delle probabilità.
# L'output delle feature non ha bisogno di una loss perché non vogliamo ottimizzarlo direttamente,
# ma ci serve solo estrarlo dopo che il modello ha imparato a classificare correttamente.
judge.compile(
    optimizer="adam",
    loss={
        "out_probs": "sparse_categorical_crossentropy",
        "out_features": None 
    }
)

print("Addestramento del Giudice in corso (Estrazione delle caratteristiche)...")
# Durante il fit, forniamo al modello i dati e i relativi target. 
# Dato che il modello ha due output, passiamo una lista di target:
# 1. Un array di zeri per 'out_features' (che viene ignorato dalla loss 'None').
# 2. Le etichette reali per 'out_probs' (usate per l'addestramento).
judge.fit(
    x=x_train, 
    y=[np.zeros((len(y_train), 64)), y_train], 
    epochs=2, 
    batch_size=256, 
    verbose=1
)

# 4. SIMULAZIONE DI UN MODELLO GENERATIVO
# Per testare le metriche senza dover addestrare una GAN, creiamo dei campioni "finti" 
# aggiungendo rumore casuale a immagini reali.
real_samples = x_test[:500]
noise_factor = 0.1
fake_samples = real_samples + noise_factor * np.random.normal(size=real_samples.shape)
fake_samples = np.clip(fake_samples, 0., 1.).astype("float32")

# 5. CALCOLO DELLA FID (FRÉCHET INCEPTION DISTANCE)
# La FID misura la distanza tra la distribuzione delle immagini reali e quelle generate.
# Un valore basso indica che le immagini generate hanno statistiche simili a quelle reali.
def get_fid(real_imgs, fake_imgs):
    # Passaggio 1: Estraiamo le feature profonde tramite il modello Giudice.
    # predict() restituisce sia le feature che le probabilità, noi prendiamo solo le prime.
    f_real, _ = judge.predict(real_imgs, verbose=0)
    f_fake, _ = judge.predict(fake_imgs, verbose=0)
    
    # Convertiamo i tensori PyTorch in array NumPy per i calcoli statistici.
    f_real = ops.convert_to_numpy(f_real)
    f_fake = ops.convert_to_numpy(f_fake)
    

    # Passaggio 2: Calcoliamo le statistiche (media e covarianza) per entrambe le distribuzioni.
    # mu1 e sigma1 rappresentano la distribuzione delle immagini reali nello spazio delle feature.
    # mu2 e sigma2 rappresentano la distribuzione delle immagini generate (fake).
    mu1, sigma1 = f_real.mean(axis=0), np.cov(f_real, rowvar=False)
    mu2, sigma2 = f_fake.mean(axis=0), np.cov(f_fake, rowvar=False)
    
    # Aggiungiamo un piccolo valore (epsilon) alla diagonale delle matrici di covarianza.
    # Questa tecnica di regolarizzazione serve a rendere le matrici definite positive,
    # evitando errori numerici o matrici singolari durante il calcolo della radice quadrata.
    sigma1 += np.eye(sigma1.shape[0]) * 1e-6
    sigma2 += np.eye(sigma2.shape[0]) * 1e-6

    
    # Passaggio 3: Calcoliamo la distanza tra le medie e il contributo delle covarianze.
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    
    # Gestiamo eventuali componenti immaginarie dovute a imprecisioni nel calcolo della radice quadrata.
    if np.iscomplexobj(covmean): 
        covmean = covmean.real
    
    # Formula finale della Distanza di Fréchet.
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# 6. CALCOLO DELL'INCEPTION SCORE (IS)
# L'IS valuta due aspetti: la chiarezza dell'immagine (deve appartenere chiaramente a una classe) 
# e la varietà (il modello deve generare tutte le classi del dataset).
def get_is(imgs):
    # Passaggio 1: Otteniamo le probabilità di classe (secondo output del Giudice).
    # Per ogni immagine, il modello restituisce un vettore di probabilità (p(y|x)).
    _, probs = judge.predict(imgs, verbose=0)
    
    # Passaggio 2: Calcoliamo la distribuzione marginale (p(y)).
    # Rappresenta la media delle predizioni su tutto il batch di immagini.
    # Se il modello genera immagini variegate, questa distribuzione dovrebbe essere uniforme (alta entropia).
    p_y = np.mean(probs, axis=0)
    
    # Passaggio 3: Calcolo della Divergenza di Kullback-Leibler (KL Divergence).
    # L'Inception Score si basa su due pilastri fondamentali:
    # 1. Nitidezza (Sharpness): ogni immagine deve essere classificata con alta confidenza (bassa entropia di p(y|x)).
    # 2. Diversità (Diversity): l'insieme delle immagini deve coprire tutte le classi (alta entropia di p(y)).
    # La divergenza KL tra p(y|x) e p(y) cattura entrambi: è alta se la predizione per la singola
    # immagine è molto diversa (più "specifica") rispetto alla media di tutte le immagini.
    kl_div = probs * (np.log(probs + 1e-10) - np.log(p_y + 1e-10))
    
    # Passaggio 4: Calcolo dello score finale.
    # Sommiamo la KL su ogni classe, ne facciamo la media su tutto il batch e applichiamo l'esponenziale.
    # Un valore IS elevato indica che le immagini sono sia nitide che varie.
    is_score = np.exp(np.mean(np.sum(kl_div, axis=1)))
    return is_score

# Esecuzione dei calcoli sulle metriche.
fid_score = get_fid(real_samples, fake_samples)
is_val = get_is(fake_samples)

# 7. VISUALIZZAZIONE DEI RISULTATI
# Creiamo un confronto visivo tra un'immagine reale e la sua versione rumorosa (simulata generata).
plt.figure(figsize=(10, 5))

# Immagine Reale
plt.subplot(1, 2, 1)
plt.imshow(real_samples[0].reshape(28,28), cmap='gray')
plt.title("Immagine Reale")
plt.axis("off")

# Immagine Generata (Simulata con rumore)
plt.subplot(1, 2, 2)
plt.imshow(fake_samples[0].reshape(28,28), cmap='gray')
plt.title("Immagine Generata (Rumore)")
plt.axis("off")

# Titolo complessivo con spiegazione rapida delle metriche:
# FID: più è bassa, più il set generato è fedele a quello reale.
# IS: più è alta, più le immagini sono nitide e variegate.
plt.suptitle(f"Analisi Metriche Generative\nFID: {fid_score:.2f} (Valori bassi sono migliori) | IS: {is_val:.2f} (Valori alti sono migliori)")
plt.show()
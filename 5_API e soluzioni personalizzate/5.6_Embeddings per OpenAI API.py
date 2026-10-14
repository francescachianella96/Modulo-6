import torch
import requests
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# CONFIGURAZIONE DEI PARAMETRI PER LM STUDIO
# Questo blocco definisce le coordinate per comunicare con il server locale di LM Studio.
# Il server deve essere avviato e il modello di embedding specificato deve essere caricato.
URL_BASE = "http://localhost:1234/v1/embeddings"
MODEL_NAME = "text-embedding-embeddinggemma-300m"  # Identificativo del modello in uso

def get_local_embedding(text: str):
    """
    Questa funzione si occupa di trasformare una stringa di testo in un vettore numerico (embedding).
    Interagisce direttamente con l'API REST di LM Studio tramite una richiesta POST.
    """
    # Il payload contiene il testo da elaborare e il nome del modello richiesto.
    payload = {"input": text, "model": MODEL_NAME}
    try:
        # Invio della richiesta al server locale.
        response = requests.post(URL_BASE, json=payload)
        # Verifica se la richiesta è andata a buon fine (status code 200).
        response.raise_for_status()
        
        # Estrazione del vettore numerico dalla risposta JSON convertendolo in un tensore PyTorch.
        # Il formato di ritorno dell'API è solitamente una lista di numeri decimali.
        return torch.tensor(response.json()["data"][0]["embedding"], dtype=torch.float32)
    except Exception as e:
        # In caso di errore (es. server spento o modello errato), viene stampato il dettaglio.
        print(f"Errore durante il recupero dell'embedding: {e}")
        return None

# 1. DEFINIZIONE DEL DATASET
# Creiamo una lista di frasi appartenenti a tre categorie semantiche distinte:
# Tecnologia/Cloud, Natura/Animali e Finanza/Economia.
sentences = [
    # Categoria: Tecnologia e Infrastruttura
    "Errore 503 sul load balancer durante il picco di traffico.",
    "Configurazione delle security groups su AWS EC2.",
    "Il container Docker crasha all'avvio per memoria insufficiente.",
    # Categoria: Mondo Naturale
    "Il gatto fa le fusa quando dorme al sole.",
    "Le migrazioni stagionali delle balene nell'oceano.",
    "Un cucciolo di labrador gioca con la pallina.",
    # Categoria: Economia e Finanza
    "La banca centrale ha alzato i tassi di interesse.",
    "L'inflazione annuale ha superato il 3 percento.",
    "Analisi dei dividendi azionari del settore tech."
]

print(f"Inizio generazione degli embedding per {len(sentences)} frasi...")

# Generazione dei vettori per ogni frase.
# Viene eseguita una 'list comprehension' che chiama get_local_embedding per ogni elemento di sentences.
# torch.stack unisce i singoli vettori in un unico grande tensore (matrice), facilitandone la gestione.
embeddings_list = [get_local_embedding(s) for s in sentences]
embeddings_tensor = torch.stack(embeddings_list)

# Conversione dei dati da PyTorch a NumPy.
# Scikit-learn (usata dopo per t-SNE e KMeans) richiede matrici NumPy invece di tensori PyTorch.
X_high_dim = embeddings_tensor.detach().numpy()

# 2. RIDUZIONE DIMENSIONALE CON t-SNE
# Gli embedding originali hanno centinaia o migliaia di dimensioni (es. 1536 o 768).
# Per poterli visualizzare su uno schermo, dobbiamo ridurli a 2 dimensioni.
print("Esecuzione dell'algoritmo t-SNE per la riduzione dello spazio da alta dimensione a 2D...")

# t-SNE (t-distributed Stochastic Neighbor Embedding) preserva le distanze relative:
# frasi simili rimarranno vicine anche nello spazio 2D ridotto.
tsne = TSNE(
    n_components=2,     # Vogliamo ottenere coordinate (x, y)
    perplexity=3,       # Parametro che influenza il bilanciamento tra visione locale e globale (basso per pochi dati)
    random_state=42,    # Fissa il seme per ottenere lo stesso risultato a ogni esecuzione
    init='pca'
)
# fit_transform calcola la riduzione e restituisce le nuove coordinate 2D.
X_2d = tsne.fit_transform(X_high_dim)

# 3. RAGGRUPPAMENTO (CLUSTERING) CON K-MEANS
# Ora che abbiamo i punti in 2D, usiamo K-Means per identificarne i gruppi naturali.
print("Esecuzione del clustering K-Means sulle coordinate bidimensionali...")

# KMeans cercherà di dividere i dati in 3 gruppi (n_clusters=3).
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
# fit_predict assegna a ogni punto un'etichetta numerica (0, 1 o 2) che indica l'appartenenza al cluster.
cluster_labels = kmeans.fit_predict(X_2d)

# 4. CREAZIONE DEL GRAFICO (VISUALIZZAZIONE)
# Utilizziamo la libreria Matplotlib per generare un'immagine comprensibile dei cluster trovati.
plt.figure(figsize=(16, 10))

# Creazione dello scatter plot (grafico a dispersione).
# X_2d[:, 0] sono le ascisse, X_2d[:, 1] le ordinate.
# Il colore 'c' è determinato dalle etichette del cluster identificate da K-Means.
scatter = plt.scatter(
    X_2d[:, 0], X_2d[:, 1],
    c=cluster_labels, 
    cmap='viridis',     # Schema colori per distinguere i cluster
    s=250,              # Dimensione dei punti nel grafico
    edgecolors='black', # Bordo nero per rendere i punti più definiti
    alpha=0.8           # Leggera trasparenza
)

# Inserimento delle annotazioni testuali per ogni punto sul grafico.
print("Inserimento delle etichette descrittive nel grafico...")
for i, txt in enumerate(sentences):
    # Abbreviamo il testo se supera i 40 caratteri per evitare sovrapposizioni eccessive.
    short_txt = (txt[:40] + '..') if len(txt) > 40 else txt
    
    # plt.annotate permette di scrivere del testo accanto a coordinate specifiche.
    plt.annotate(
        short_txt,
        (X_2d[i, 0], X_2d[i, 1]),   # Punto esatto del grafico a cui si riferisce il testo
        xytext=(10, 10),            # Spostamento del testo rispetto al punto (10 pixel a destra e in alto)
        textcoords='offset points', # Specifica che xytext è un offset relativo
        fontsize=9,
        # Definiamo uno stile per il riquadro del testo (sfondo bianco semitrasparente).
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        # Aggiungiamo una piccola linea (freccia leggera) che collega il testo al suo punto.
        arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0.2", color='gray')
    )

# Configurazione finale degli elementi estetici e informativi del grafico.
plt.title(f"Mappa Semantica 2D (t-SNE + K-Means su {MODEL_NAME})", fontsize=16)
plt.xlabel("Dimensione t-SNE 1 (Asse X)")
plt.ylabel("Dimensione t-SNE 2 (Asse Y)")
plt.grid(True, linestyle='--', alpha=0.5)

# Aggiunta della legenda per identificare i cluster in base al colore.
legend = plt.legend(*scatter.legend_elements(), title="ID del Cluster", loc="best")
plt.gca().add_artist(legend)

# Ottimizzazione dello spazio tra i margini e visualizzazione a schermo.
plt.tight_layout()
print("Processo completato. Apertura della finestra del grafico.")
plt.show()
import matplotlib.pyplot as plt
import numpy as np

# 1. DEFINIZIONE DELLE DIMENSIONI DA TESTARE (da 1 a 500)
dimensions = [1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500]

# Liste vuote per salvare i risultati di ogni iterazione
dist_means = []
dist_mins = []
dist_maxs = []

# 2. CALCOLO DELLE DISTANZE IN SPAZI A DIMENSIONALITÀ CRESCENTE
for d in dimensions:
    # Generiamo 1000 coppie di punti casuali in 'd' dimensioni
    A = np.random.uniform(0, 1, size=(1000, d))
    B = np.random.uniform(0, 1, size=(1000, d))

    # Calcolo della distanza euclidea tra le coppie di punti
    distances = np.sqrt(np.sum((A - B) ** 2, axis=1))

    # Calcoliamo e salviamo min, media e max
    dist_means.append(np.mean(distances))
    dist_mins.append(np.min(distances))
    dist_maxs.append(np.max(distances))

# 3. CREAZIONE DEL GRAFICO CON MATPLOTLIB
plt.figure(figsize=(10, 6))

# Tracciamo le tre linee: Minima, Media e Massima
plt.plot(dimensions, dist_maxs, label="Distanza Massima", color="crimson", linestyle="--", marker="o")
plt.plot(dimensions, dist_means, label="Distanza Media", color="forestgreen", linewidth=2, marker="s")
plt.plot(dimensions, dist_mins, label="Distanza Minima", color="royalblue", linestyle="--", marker="^")

# Personalizzazione del grafico
plt.title("Maledizione della Dimensionalità: Convergenza delle Distanze", fontsize=14)
plt.xlabel("Numero di Dimensioni (d)", fontsize=12)
plt.ylabel("Distanza Euclidea", fontsize=12)
plt.grid(True, linestyle=":", alpha=0.7)
plt.legend(fontsize=11)

# Mostra il grafico
plt.show()
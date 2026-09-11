# Implementa una fuzione Python semplificata per calcolare la distanza di Frechet tra due set di vettori latenti monodimensionali
# Utilizza dati sintetici generati come medie e varianze differenti per osservare come il punteggio aumenti al crescere della divergenza tra i gruppi. 
# Suggerimento: usa i vettori medi e le deviazioni standard per calcolare la differenza quadratica e la traccia 

import numpy as np
import matplotlib.pyplot as plt

# 1. FUNZIONE PER IL CALCOLO DELLA DISTANZA DI FRÉCHET 1D
def frechet_distance_1d(x, y):
    """
    Calcola la distanza di Fréchet tra due set di vettori 1D.
    
    Parameters:
        x (array-like): Primo set di campioni 1D.
        y (array-like): Secondo set di campioni 1D.
        
    Returns:
        float: Punteggio della Distanza di Fréchet.
    """
    # Passo A: Calcolo delle medie (mu) e deviazioni standard (sigma)
    mu1, sigma1 = np.mean(x), np.std(x)
    mu2, sigma2 = np.mean(y), np.std(y)
    
    # Passo B: Differenza quadratica tra le medie
    mean_diff_sq = (mu1 - mu2) ** 2
    
    # Passo C: Contributo della variabilità (differenza quadratica tra deviazioni standard)
    std_diff_sq = (sigma1 - sigma2) ** 2
    
    # Passo D: Distanza finale di Fréchet
    fd = mean_diff_sq + std_diff_sq
    return fd

# 2. GENERAZIONE DI DATI SINTETICI DI TEST
np.random.seed(42)

# Gruppo di riferimento (Reale): media = 0, deviazione standard = 1
real_data = np.random.normal(loc=0.0, scale=1.0, size=1000)

# Creazione gruppi sintetici con divergenze crescenti
# Set 1: Molto simile al reale (stessa media, varianza quasi identica)
fake_close = np.random.normal(loc=0.1, scale=1.1, size=1000)

# Set 2: Divergenza media (media spostata e varianza più grande)
fake_medium = np.random.normal(loc=1.5, scale=2.0, size=1000)

# Set 3: Alta divergenza (molto distante dal reale)
fake_far = np.random.normal(loc=4.0, scale=3.5, size=1000)

# 3. CALCOLO E VERIFICA DEI PUNTEGGI
fd_close = frechet_distance_1d(real_data, fake_close)
fd_medium = frechet_distance_1d(real_data, fake_medium)
fd_far = frechet_distance_1d(real_data, fake_far)

print(f"FD - Set Molto Simile:   {fd_close:.4f}")
print(f"FD - Set Divergenza Media: {fd_medium:.4f}")
print(f"FD - Set Alta Divergenza:  {fd_far:.4f}")

# 4. VISUALIZZAZIONE GRAFICA
plt.figure(figsize=(10, 6))
plt.hist(real_data, bins=30, alpha=0.5, label='Reali (μ=0, σ=1)', color='blue')
plt.hist(fake_close, bins=30, alpha=0.5, label=f'Simili (FD: {fd_close:.2f})', color='green')
plt.hist(fake_medium, bins=30, alpha=0.5, label=f'Medi (FD: {fd_medium:.2f})', color='orange')
plt.hist(fake_far, bins=30, alpha=0.5, label=f'Distanti (FD: {fd_far:.2f})', color='red')

plt.title("Confronto Distanza di Fréchet 1D su Distribuzioni Diverse")
plt.xlabel("Valore del Vettore Latente")
plt.ylabel("Frequenza")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
import torch 
import torch.nn as nn 
import torch.optim as optim 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

# --- 1. PREPARAZIONE DEI DATI ---
# Carichiamo il dataset Iris che contiene 150 osservazioni di 3 specie diverse di fiori
iris = load_iris()
# X contiene le caratteristiche: lunghezza e larghezza di sepali e petali
X = iris.data
# y contiene le etichette target: 0 (Setosa), 1 (Versicolor), 2 (Virginica)
y = iris.target

# Normalizzazione dei dati: i modelli di Deep Learning convergono molto più velocemente
# e in modo più stabile se i dati in ingresso hanno scale simili.
scaler = StandardScaler()
# Calcoliamo media e deviazione standard del dataset e trasformiamo i dati originali
X = scaler.fit_transform(X)

# Dividiamo il dataset: 80% per l'addestramento e 20% per testare quanto il modello ha imparato
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertiamo i dati in Tensori di PyTorch, poiché la rete neurale non accetta direttamente array NumPy
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
# Per le etichette di classificazione usiamo LongTensor (numeri interi a 64 bit)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# --- 2. DEFINIZIONE DELL'ARCHITETTURA ---
# Definiamo la classe della nostra rete neurale ereditando da nn.Module
class IrisNet(nn.Module):
    def __init__(self, input_size=4, hidden_size=16, num_classes=3):
        # Chiamata obbligatoria al costruttore della classe base
        super(IrisNet, self).__init__()
        # Creiamo un contenitore sequenziale per i nostri strati
        self.network = nn.Sequential(
            # Strato di ingresso: trasforma le 4 caratteristiche in 16 caratteristiche astratte
            nn.Linear(input_size, hidden_size),
            # Attivazione ReLU: permette alla rete di imparare relazioni non lineari nei dati
            nn.ReLU(),
            # Strato di uscita: riduce le 16 caratteristiche alle 3 classi finali
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # Definisce come l'input 'x' deve passare attraverso la rete per produrre il risultato
        return self.network(x)

# Inizializziamo il modello (necessario prima di definirne l'ottimizzatore)
model = IrisNet()

# --- 3. ADDESTRAMENTO (TRAINING) ---
# Scegliamo la funzione di costo per la classificazione multi-classe (Cross Entropy)
# Questa funzione misura quanto la previsione del modello si discosta dal valore reale
criterion = nn.CrossEntropyLoss()
# Inizializziamo l'ottimizzatore Adam che aggiornerà i pesi della rete per ridurre l'errore
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("Inizio della fase di addestramento del modello neurale...")
# Iteriamo per 100 cicli completi su tutto il set di addestramento (Epoche)
for epoch in range(100):
    # Passaggio fondamentale: azzeriamo i gradienti precedenti per evitare che si accumulino
    optimizer.zero_grad()

    # Passaggio in avanti (Forward pass): il modello cerca di indovinare le classi basandosi sui dati attuali
    outputs = model(X_train)

    # Calcoliamo l'errore confrontando le previsioni con le etichette reali
    loss = criterion(outputs, y_train)

    # Retropropagazione (Backward pass): calcoliamo di quanto ogni peso deve essere modificato
    loss.backward()

    # Aggiornamento: l'ottimizzatore modifica i pesi della rete neurale
    optimizer.step()

    # Ogni 20 epoche stampiamo il valore dell'errore (loss) per monitorare i progressi
    if (epoch+1) % 20 == 0:
        print(f"Epoca [{epoch+1}/100], Errore Residuo: {loss.item():.4f}")

# --- 4. SALVATAGGIO DEI RISULTATI ---
# Salviamo lo 'state_dict', ovvero i pesi che la rete ha imparato, su un file .pth
# Questo file verrà caricato dall'app di produzione per fare previsioni senza dover ri-addestrare.
torch.save(model.state_dict(), "iris_model.pth")

# Salviamo anche lo 'scaler': questo è critico per l'API di produzione,
# perché i nuovi dati andranno normalizzati con gli stessi parametri (media/varianza) usati qui.
import joblib
joblib.dump(scaler, "scaler.pkl")

print("\nAsset salvati con successo: 'iris_model.pth' e 'scaler.pkl' sono pronti per il deploy!")
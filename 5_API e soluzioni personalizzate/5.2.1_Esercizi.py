import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from flask import Flask, request, jsonify
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Definizione dell'architettura della rete neurale per il dataset MNIST.
# La classe eredita da nn.Module, componente fondamentale di PyTorch per la creazione di modelli.
class MNISTNet(nn.Module):
    def __init__(self):
        """
        Inizializzazione dei livelli della rete.
        Viene utilizzata una struttura sequenziale per definire il flusso dei dati.
        """
        super(MNISTNet, self).__init__()
        self.main = nn.Sequential(
            # Appiattisce l'input (28x28 pixel) in un vettore monodimensionale di 784 elementi.
            nn.Flatten(),
            # Primo livello lineare: trasforma 784 input in 128 caratteristiche latenti.
            nn.Linear(784, 128),
            # Funzione di attivazione ReLU per introdurre non-linearità nel modello.
            nn.ReLU(),
            # Dropout al 20% per prevenire l'overfitting durante la fase di addestramento.
            nn.Dropout(0.2),
            # Secondo livello lineare: riduce le caratteristiche da 128 a 64.
            nn.Linear(128, 64),
            nn.ReLU(),
            # Livello di output: 64 caratteristiche mappate sulle 10 classi (cifre da 0 a 9).
            nn.Linear(64, 10)
        )

    def forward(self, x):
        """
        Definisce il passaggio dei dati attraverso la sequenza di livelli definita nell'init.
        """
        return self.main(x)

# Definizione delle costanti di sistema e dei parametri di configurazione.
MODEL_PATH = "mnist_real_model.pth"
DEVICE = torch.device("cpu") # Esecuzione su CPU per compatibilità universale.

# Parametri statistici del dataset MNIST per la normalizzazione dei dati.
# Questi valori (media e deviazione standard) assicurano che l'input abbia una distribuzione standard.
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# Integrazione del framework Flask per l'esposizione del modello tramite API REST.
app = Flask(__name__)

def train_model():
    """
    Gestisce il download del dataset, l'addestramento della rete e il salvataggio dei pesi.
    Questa funzione interagisce con le classi MNIST di torchvision e DataLoader di torch.utils.data.
    """
    if os.path.exists(MODEL_PATH):
        print("Modello esistente rilevato. Caricamento in corso.")
        return

    print("Download del dataset MNIST e avvio della procedura di addestramento.")
    
    # Definizione della pipeline di trasformazione per i dati di input.
    # Converte le immagini in tensori e applica la normalizzazione Z-score.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])

    # Utilizzo della classe datasets.MNIST per recuperare i dati ufficiali.
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # DataLoader suddivide il dataset in batch per l'elaborazione parallela e ottimizzata.
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    # Istanza del modello e spostamento sul dispositivo di calcolo definito nelle costanti.
    model = MNISTNet().to(DEVICE)
    # CrossEntropyLoss è la funzione di perdita standard per la classificazione multiclasse.
    criterion = nn.CrossEntropyLoss()
    # Adam è l'ottimizzatore che aggiorna i pesi della rete in base ai gradienti calcolati.
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train() # Imposta il modello in modalità addestramento (attiva Dropout).
    
    # Ciclo di addestramento su un'unica epoca.
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()      # Reset dei gradienti accumulati dal passaggio precedente.
        output = model(data)       # Esecuzione del forward pass (predizione).
        loss = criterion(output, target) # Calcolo dell'errore rispetto al target reale.
        loss.backward()            # Backpropagation: calcolo dei gradienti.
        optimizer.step()           # Aggiornamento dei pesi del modello.
        
        if batch_idx % 200 == 0:
            print(f"Progresso: Batch {batch_idx}/{len(train_loader)} - Errore (Loss): {loss.item():.4f}")

    # Esportazione dei pesi addestrati nel file di sistema.
    torch.save(model.state_dict(), MODEL_PATH)
    print("Salvataggio dei parametri ottimizzati completato.")

def preprocess_image(data_list):
    """
    Trasforma i dati grezzi ricevuti tramite API in un formato compatibile con la rete.
    Garantisce la coerenza tra i dati di inviati dall'utente e quelli usati in addestramento.
    """
    # Conversione della lista pixel in array NumPy e normalizzazione nel range [0, 1].
    img = np.array(data_list, dtype=np.float32) / 255.0
    
    if img.size != 784:
        raise ValueError(f"Dimensione input errata: richiesti 784 pixel, rilevati {img.size}")
    
    # Applicazione dei parametri di normalizzazione (Media e Deviazione Standard).
    img = (img - MNIST_MEAN) / MNIST_STD
    
    # Conversione in tensore PyTorch con reshape (Batch Size=1, Canale=1, Altezza=28, Larghezza=28).
    return torch.from_numpy(img).view(1, 1, 28, 28)

# Sequenza di inizializzazione all'avvio del server.
train_model()
model = MNISTNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval() # Imposta il modello in modalità valutazione (disattiva Dropout).

@app.route('/predict', methods=['POST'])
def predict():
    """
    Gestore dell'endpoint API per la classificazione in tempo reale.
    Interagisce con la funzione di preprocessing e i metodi del modello PyTorch.
    """
    try:
        # Recupero del payload JSON dalla richiesta HTTP POST.
        data = request.get_json()
        if not data or 'input_data' not in data:
            return jsonify({"error": "Parametro 'input_data' mancante nel corpo della richiesta."}), 400

        # Passaggio dei dati alla funzione di normalizzazione tecnica.
        input_tensor = preprocess_image(data['input_data'])

        # Disabilitazione del calcolo dei gradienti per risparmiare risorse durante l'inferenza.
        with torch.no_grad():
            logits = model(input_tensor)        # Generazione dei valori grezzi di output.
            probs = torch.softmax(logits, dim=1) # Conversione dei logits in probabilità (0-1).
            confidence, prediction = torch.max(probs, dim=1) # Selezione della classe più probabile.

        # Generazione della risposta strutturata in formato JSON.
        return jsonify({
            "prediction": int(prediction.item()),
            "confidence": f"{confidence.item():.2%}",
            "status": "Successo"
        }), 200

    except Exception as e:
        # Log dell'errore e restituzione di un feedback generico al client.
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Esecuzione dell'applicazione server sulla rete locale.
    app.run(host='0.0.0.0', port=5000)
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify

# Definizione della struttura del modello neurale.
# Questa classe eredita da nn.Module, la base per tutti i moduli di rete neurale in PyTorch.
class HealthClassifier(nn.Module):
    def __init__(self):
        # Inizializzazione della classe base.
        super(HealthClassifier, self).__init__()
        # Definiamo uno strato lineare (fully connected).
        # Prende in input 4 caratteristiche e restituisce 2 valori di uscita (le classi).
        self.linear = nn.Linear(4, 2) 

    def forward(self, x):
        # Il metodo forward definisce come i dati fluiscono attraverso la rete.
        # Qui i dati passano semplicemente attraverso lo strato lineare.
        return self.linear(x)

# Preparazione dell'ambiente di esecuzione: usiamo la CPU per l'inferenza.
device = torch.device("cpu")

# Creazione di un'istanza del modello e spostamento sulla memoria del dispositivo (CPU).
model = HealthClassifier().to(device)

# Predisposizione del modello per la fase di valutazione.
# Questo passaggio è fondamentale perché disabilita comportamenti specifici dell'addestramento 
# come il Dropout o la Batch Normalization, garantendo risultati costanti.
model.eval()

# Inizializzazione dell'applicazione Flask per creare il server web.
app = Flask(__name__)

# Definizione dell'endpoint API che riceverà le richieste di classificazione.
# Accetta solo richieste di tipo POST.
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Estrazione del corpo della richiesta in formato JSON.
        data = request.get_json()
        
        # Validazione minimale: verifichiamo che i dati esistano e contengano la chiave corretta.
        if not data or 'features' not in data:
            return jsonify({"error": "Dati mancanti. Inviare un JSON con la chiave 'features'"}), 400

        # Recupero della lista di caratteristiche inviate dall'utente.
        raw_features = data['features']
        
        # Conversione dei dati grezzi in un array NumPy con precisione a 32 bit (float32).
        # PyTorch lavora in modo ottimale con questa precisione numerica.
        input_array = np.array(raw_features, dtype=np.float32)
        
        # Conversione da array NumPy a tensore PyTorch e aggiunta di una dimensione.
        # Il modello si aspetta un input in formato (batch_size, num_features).
        # unsqueeze(0) trasforma un vettore (4) in una matrice (1, 4).
        input_tensor = torch.from_numpy(input_array).unsqueeze(0).to(device)

        # Fase di inferenza (calcolo del risultato).
        # Usiamo torch.no_grad() per evitare di calcolare i gradienti, riducendo l'uso di memoria e aumentando la velocità.
        with torch.no_grad():
            # Passiamo il tensore al modello (chiama internamente il metodo forward).
            logits = model(input_tensor)
            
            # Applichiamo la funzione Softmax per convertire i valori di uscita (logits) in probabilità.
            # dim=1 indica che la somma delle probabilità deve essere 1 per ogni riga del batch.
            probabilities = torch.softmax(logits, dim=1)
            
            # Determiniamo la classe con la probabilità più alta.
            # item() estrae il valore scalare dal tensore di output.
            prediction = torch.argmax(probabilities, dim=1).item()
            
            # Estraiamo il valore di confidenza associato alla previsione.
            confidenza = torch.max(probabilities).item()

        # Mappatura del risultato numerico in una stringa leggibile.
        # In questo scenario: 1 = In Salute, 0 = Affaticato.
        status = "In Salute" if prediction == 1 else "Affaticato"
        
        # Restituzione della risposta al client in formato JSON.
        return jsonify({
            "status": status,
            "confidence": f"{confidenza:.2%}",
            "model_version": "2026.1.0"
        }), 200

    except Exception as e:
        # Gestione generica degli errori per evitare il crash del server.
        # Restituisce il messaggio dell'errore e lo status code 500 (Internal Server Error).
        return jsonify({"error": str(e)}), 500

# Punto di ingresso dello script.
if __name__ == '__main__':
    # Avvio del server Flask in ascolto su tutte le interfacce di rete (0.0.0.0) sulla porta 5000.
    app.run(host='0.0.0.0', port=5000)
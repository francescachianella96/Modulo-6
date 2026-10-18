import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pytorch
import optuna
import logging
import warnings

# -----------------------------------------------------------------------------
# SETUP LOGGING E WARNING
# -----------------------------------------------------------------------------
# Filtriamo i warning per mantenere l'output pulito durante l'esecuzione.
# Molti warning di MLflow sono informativi (es. versioni locali di PyTorch)
# e possono distrarre durante una prima lettura.
logging.getLogger("mlflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. PREPARAZIONE DEI DATI
#
# Questa funzione si occupa di generare un dataset sintetico e di prepararlo
# per l'utilizzo con PyTorch. Esegue tutti i passaggi necessari: generazione,
# split in train/val e conversione in tensori.
# -----------------------------------------------------------------------------
def get_data():
    # Generiamo un dataset "moons" (due semicerchi incastrati), un classico esempio
    # per testare problemi di classificazione non lineare.
    # n_samples=1000: numero totale di punti generati.
    # noise=0.2: aggiunge variabilità ai dati per renderli meno perfetti e più realistici.
    # random_state=42: fissa il seme casuale per garantire che i dati siano identici ad ogni esecuzione.
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

    # Dividiamo i dati in training set (per l'addestramento) e validation set (per la valutazione).
    # test_size=0.2: indica che il 20% dei dati verrà riservato per la validazione.
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    
    # Conversione dei dati da array NumPy a Tensori PyTorch.
    # I modelli PyTorch richiedono dati sotto forma di tensori.
    # FloatTensor viene usato per le feature (numeri decimali), LongTensor per le etichette (interi delle classi).
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

    # Creazione dei DataLoader.
    # I DataLoader gestiscono l'iterazione sui dati, fornendoli in pacchetti (batch) per l'addestramento.
    # batch_size=32: ogni iterazione fornirà un blocco di 32 esempi.
    return DataLoader(train_ds, batch_size=32), DataLoader(val_ds, batch_size=32)

# -----------------------------------------------------------------------------
# 2. DEFINIZIONE DEL MODELLO (ARCHITETTURA DINAMICA)
#
# Definiamo una classe per la rete neurale. Questa classe è progettata per essere flessibile:
# il numero di neuroni nascosti e il numero di strati non sono fissi, ma vengono passati
# come parametri al costruttore.
# Questo design permette a Optuna di testare diverse architetture automaticamente.
# -----------------------------------------------------------------------------
class SimpleNet(nn.Module):
    def __init__(self, n_hidden, n_layers):
        super().__init__()
        layers = []
        in_features = 2  # I dati di input (make_moons) hanno 2 caratteristiche: coordinate x e y.
        
        # Creiamo dinamicamente gli strati nascosti in base al parametro n_layers.
        for _ in range(n_layers):
            # Aggiungiamo un livello lineare che trasforma l'input in 'n_hidden' neuroni.
            layers.append(nn.Linear(in_features, n_hidden))
            # Aggiungiamo una funzione di attivazione ReLU (introduce non-linearità).
            layers.append(nn.ReLU())
            # L'uscita di questo strato diventa l'ingresso (features) per il successivo.
            in_features = n_hidden
            
        # Aggiungiamo lo strato di output finale.
        # Collega l'ultimo strato nascosto ai 2 neuroni di uscita (corrispondenti alle 2 classi: 0 e 1).
        layers.append(nn.Linear(n_hidden, 2))
        
        # nn.Sequential avvolge la lista di strati in un unico modulo che li esegue in sequenza.
        # L'asterisco (*) serve a "spacchettare" la lista layers e passarne i contenuti come argomenti singoli.
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Definisce il flusso dei dati attraverso il modello quando viene chiamato.
        return self.model(x)

# -----------------------------------------------------------------------------
# 3. FUNZIONE OBIETTIVO PER L'OTTIMIZZAZIONE
#
# Questa funzione è il cuore dello script. Viene chiamata ripetutamente da Optuna.
# Ogni chiamata rappresenta un "trial" (un esperimento) che testa una specifica
# combinazione di iperparametri.
# Inoltre, integriamo MLflow per tracciare e registrare i risultati di ogni esperimento.
# -----------------------------------------------------------------------------
def objective(trial):
    # --- DEFINIZIONE DEGLI SPAZI DI RICERCA (IPERPARAMETRI) ---
    # Usiamo l'oggetto 'trial' per suggerire valori per gli iperparametri da testare.
    
    # Range per il learning rate in scala logaritmica (esplora ordini di grandezza diversi).
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    
    # Range per il numero di neuroni nascosti (da 8 a 64).
    n_hidden = trial.suggest_int("n_hidden", 8, 64)
    
    # Range per il numero di strati nascosti (da 1 a 3).
    n_layers = trial.suggest_int("n_layers", 1, 3)

    # --- INIZIO TRACCIAMENTO CON MLFLOW ---
    # Avviamo una "run" di MLflow per registrare questo specifico esperimento.
    # nested=True indica che questa run è parte di un esperimento più grande (o gerarchia).
    with mlflow.start_run(nested=True):
        
        # Registriamo su MLflow i parametri che Optuna ha scelto per questo tentativo.
        mlflow.log_params(trial.params)
        
        # Configurazione del dispositivo di calcolo (GPU se disponibile, altrimenti CPU).
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Istanziamo il modello SimpleNet utilizzando i parametri n_hidden e n_layers suggeriti.
        model = SimpleNet(n_hidden, n_layers).to(device)
        
        # Definiamo l'ottimizzatore (Adam) configurato con il learning rate suggerito.
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Definiamo la funzione di perdita per classificazione multiclasse (CrossEntropyLoss).
        criterion = nn.CrossEntropyLoss()
        
        # Otteniamo i dataloader per addestramento e validazione.
        train_loader, val_loader = get_data()

        # --- CICLO DI ADDESTRAMENTO ---
        # Eseguiamo l'addestramento per un numero fissato di epoche (es. 10).
        for epoch in range(10):
            model.train()  # Imposta il modello in modalità addestramento (abilita dropout, batchnorm, ecc.).
            
            # Iteriamo sui batch di dati di training.
            for X_batch, y_batch in train_loader:
                # Spostiamo i dati sul device (CPU o GPU).
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # Azzeriamo i gradienti accumulati nel passo precedente.
                optimizer.zero_grad()
                
                # Calcoliamo la loss: differenza tra le predizioni del modello e i valori reali.
                loss = criterion(model(X_batch), y_batch)
                
                # Calcoliamo i gradienti (backpropagation).
                loss.backward()
                
                # Aggiorniamo i pesi del modello basandoci sui gradienti calcolati.
                optimizer.step()

            # --- VALIDAZIONE ---
            model.eval()  # Imposta il modello in modalità valutazione (disabilita dropout, ecc.).
            correct = 0
            
            # Disabilitiamo il calcolo dei gradienti per risparmiare memoria e calcoli durante la validazione.
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    outputs = model(X_val)
                    
                    # Verifichiamo quante predizioni sono corrette.
                    # argmax(1) restituisce l'indice della classe con la probabilità più alta.
                    correct += (outputs.argmax(1) == y_val).sum().item()
            
            # Calcoliamo l'accuratezza su tutto il validation set per questa epoca.
            accuracy = correct / len(val_loader.dataset)
            
            # Registriamo l'accuratezza come metrica su MLflow, associandola all'epoca corrente.
            mlflow.log_metric("accuracy", accuracy, step=epoch)
            
            # --- PRUNING (POTATURA) ---
            # Comunichiamo a Optuna l'accuratezza intermedia ottenuta in questa epoca.
            trial.report(accuracy, epoch)
            
            # Optuna valuta se vale la pena continuare questo trial.
            # Se le prestazioni sono molto peggiori rispetto ad altri trial già eseguiti,
            # interrompiamo subito l'esecuzione (Pruning) per risparmiare tempo e risorse.
            if trial.should_prune():
                mlflow.set_tag("status", "pruned")  # Segnaliamo su MLflow che il trial è stato potato.
                raise optuna.exceptions.TrialPruned()

        # Se il trial arriva alla fine (non viene potato), salviamo il modello completo su MLflow.
        mlflow.pytorch.log_model(model, "model")
        
        # Ritorniamo l'accuratezza finale. Optuna userà questo valore per capire se ha trovato un buon set di parametri.
        return accuracy

# -----------------------------------------------------------------------------
# 4. ESECUZIONE DELLO STUDIO
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    
    # Configurazione MLflow
    # Impostiamo il tracking URI in modo assoluto.
    # In questo modo, i log verranno salvati nella cartella "mlruns" all'interno
    # della directory di questo script, indipendentemente da dove viene lanciato il comando.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mlruns_uri = "file:///" + os.path.join(current_dir, "mlruns").replace("\\", "/")
    mlflow.set_tracking_uri(mlruns_uri)
    
    mlflow.set_experiment("Deep_Learning_Best_Practices")

    # Creazione Studio Optuna
    study = optuna.create_study(
        direction="maximize", 
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Avviamo il processo di ottimizzazione.
    # Optuna eseguirà la funzione 'objective' per 20 volte (n_trials=20), cercando i migliori iperparametri.
    study.optimize(objective, n_trials=20)

    # Al termine, stampiamo i risultati migliori ottenuti.
    print(f"Miglior accuratezza: {study.best_value:.4f}")
    print(f"Migliori iperparametri: {study.best_params}")
    
    print("\n----------------------------------------------------------------")
    print("PER VEDERE LA DASHBOARD DI MLFLOW:")
    print("1. Apri un nuovo terminale")
    print(f"2. Esegui il comando: mlflow ui --backend-store-uri {mlruns_uri}")
    print("3. Apri il browser all'indirizzo http://127.0.0.1:5000")
    print("----------------------------------------------------------------")
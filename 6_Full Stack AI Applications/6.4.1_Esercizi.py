import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch
import optuna
import logging
import warnings
import os

# -----------------------------------------------------------------------------
# SETUP LOGGING E WARNING
# -----------------------------------------------------------------------------
# Filtriamo i warning per mantenere l'output pulito durante l'esecuzione.
# Molti warning di MLflow sono informativi (es. versioni locali di PyTorch)
# e possono distrarre durante una prima lettura.
logging.getLogger("mlflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. PREPARAZIONE DEI DATI (MNIST)
#
# In questa versione modificata, utilizziamo il dataset MNIST (immagini 28x28 di cifre scritte a mano).
# -----------------------------------------------------------------------------
def get_data():
    # Trasformazione: converte l'immagine in tensore e normalizza i pixel tra -1 e 1 (approssimativamente).
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Scarichiamo il dataset MNIST di training.
    # download=True scaricherà i dati nella cartella './data' se non sono già presenti.
    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Dividiamo il dataset di training in Training Set effettivo e Validation Set.
    # Usiamo 50.000 immagini per il training e 10.000 per la validazione.
    train_size = 50000
    val_size = 10000
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    # Creazione dei DataLoader.
    # batch_size=64: elaboriamo 64 immagini alla volta.
    return DataLoader(train_dataset, batch_size=64, shuffle=True), DataLoader(val_dataset, batch_size=64)

# -----------------------------------------------------------------------------
# 2. DEFINIZIONE DEL MODELLO (ARCHITETTURA DINAMICA PER IMMAGINI)
#
# Adattiamo SimpleNet per gestire input immagine (28x28 = 784 pixel) e output multiclasse (10 cifre).
# -----------------------------------------------------------------------------
class SimpleNet(nn.Module):
    def __init__(self, n_hidden, n_layers):
        super().__init__()
        layers = []
        
        # Primo strato: Appiattisce l'immagine 28x28 in un vettore di 784 elementi.
        layers.append(nn.Flatten())
        
        in_features = 28 * 28  # 784 input
        
        # Creiamo dinamicamente gli strati nascosti in base al parametro n_layers.
        for _ in range(n_layers):
            layers.append(nn.Linear(in_features, n_hidden))
            layers.append(nn.ReLU())
            in_features = n_hidden
            
        # Aggiungiamo lo strato di output finale.
        # Collega l'ultimo strato nascosto ai 10 neuroni di uscita (cifre 0-9).
        layers.append(nn.Linear(n_hidden, 10))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# -----------------------------------------------------------------------------
# 3. FUNZIONE OBIETTIVO PER L'OTTIMIZZAZIONE
# -----------------------------------------------------------------------------
def objective(trial):
    # --- DEFINIZIONE DEGLI SPAZI DI RICERCA (IPERPARAMETRI) ---
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    n_hidden = trial.suggest_int("n_hidden", 32, 128) # Aumentato range per MNIST
    n_layers = trial.suggest_int("n_layers", 1, 3)

    # --- INIZIO TRACCIAMENTO CON MLFLOW ---
    # Creiamo una run annidata.
    with mlflow.start_run(nested=True):
        
        # Log dei parametri scelti da Optuna.
        mlflow.log_params(trial.params)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Istanziamo il modello e l'ottimizzatore.
        model = SimpleNet(n_hidden, n_layers).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        train_loader, val_loader = get_data()

        # --- CICLO DI ADDESTRAMENTO ---
        # Riduciamo a 5 epoche per velocità, dato che MNIST è più pesante di make_moons.
        for epoch in range(5):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()

            # --- VALIDAZIONE ---
            model.eval()
            correct = 0
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    outputs = model(X_val)
                    correct += (outputs.argmax(1) == y_val).sum().item()
            
            accuracy = correct / len(val_loader.dataset)
            
            # Registriamo l'accuratezza come metrica su MLflow.
            mlflow.log_metric("accuracy", accuracy, step=epoch)
            
            # --- PRUNING (POTATURA) ---
            trial.report(accuracy, epoch)
            if trial.should_prune():
                mlflow.set_tag("status", "pruned")
                raise optuna.exceptions.TrialPruned()

        # Salvataggio del modello finale.
        mlflow.pytorch.log_model(model, "model")
        
        return accuracy

# -----------------------------------------------------------------------------
# 4. ESECUZIONE DELLO STUDIO
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    
    # Configurazione MLflow con percorso assoluto nella cartella corrente (mlruns_mnist).
    # Usiamo un nome diverso per la cartella per non mischiarla con l'esempio precedente.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mlruns_uri = "file:///" + os.path.join(current_dir, "mlruns_mnist").replace("\\", "/")
    mlflow.set_tracking_uri(mlruns_uri)
    
    mlflow.set_experiment("MNIST_Classification_Optuna")

    # Creazione Studio Optuna.
    study = optuna.create_study(
        direction="maximize", 
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Eseguiamo 10 trial come richiesto.
    study.optimize(objective, n_trials=10)

    print(f"Miglior accuratezza: {study.best_value:.4f}")
    print(f"Migliori iperparametri: {study.best_params}")
    
    print("\n----------------------------------------------------------------")
    print("PER VEDERE LA DASHBOARD DI MLFLOW:")
    print("1. Apri un nuovo terminale")
    print(f"2. Esegui il comando: mlflow ui --backend-store-uri {mlruns_uri}")
    print("3. Apri il browser all'indirizzo http://127.0.0.1:5000")
    print("----------------------------------------------------------------")
# mlflow server --host 127.0.0.1 --port 5000


import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# 1. ARCHITETTURA DEL MODELLO
# La classe SempliceClassificatore eredita da nn.Module (standard PyTorch).
# Definisce la struttura della rete neurale che verrà poi serializzata e gestita da MLflow.
class SempliceClassificatore(nn.Module):
    def __init__(self):
        """
        Costruttore della classe. Inizializza un singolo livello lineare.
        Input: 10 caratteristiche (feature) numeriche.
        Output: 2 valori (logit) corrispondenti alle classi di classificazione.
        """
        super(SempliceClassificatore, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        """
        Definisce il flusso dei dati attraverso la rete (forward pass).
        Riceve il tensore x e restituisce le predizioni del livello lineare.
        """
        return self.fc(x)

# CONFIGURAZIONE AMBIENTE MLFLOW
# Specifichiamo dove risiede il server di tracciamento e il nome dell'esperimento.
mlflow.set_tracking_uri("http://127.0.0.1:5000")
experiment_name = "Esercitazione_Model_Registry"
mlflow.set_experiment(experiment_name)

# Nome univoco sotto cui il modello verrà registrato nel Model Registry globale.
model_name = "ClassificatoreResiliente"

def train_and_register(version_note, lr=0.01):
    """
    Funzione ausiliaria che incapsula l'intero processo di:
    1. Creazione di una run MLflow.
    2. Addestramento simulato del modello SempliceClassificatore.
    3. Logging di parametri e metriche.
    4. Registrazione automatica della versione nel Model Registry.
    """
    with mlflow.start_run() as run:
        # Istanza del modello: interagisce con MLflow durante la fase di logging (log_model).
        model = SempliceClassificatore()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        
        # Salvataggio dei metadati del training per garantire la riproducibilità.
        mlflow.log_param("learning_rate", lr)
        
        # Simuliamo una metrica di accuratezza variabile in base all'iperparametro lr.
        accuracy = 0.85 if lr == 0.01 else 0.92
        mlflow.log_metric("accuracy", accuracy)
        
        # log_model salva i pesi del modello come artefatti.
        # registered_model_name innesca la logica del Model Registry: se il modello non esiste,
        # lo crea; se esiste già, aggiunge una nuova versione incrementale (v1, v2, ecc.).
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=model_name
        )
        print(f"Esecuzione completata e modello inviato al registry: {version_note}")

# --- FASE 1: GESTIONE DEL CICLO DI VITA (LIFECYCLE MANAGEMENT) ---

# Creazione della prima versione. Questa operazione registra il modello e gli assegna la versione 1.
train_and_register("Versione 1: Configurazione base stabile", lr=0.01)

# Creazione della seconda versione. MLflow incrementa automaticamente il numero di versione a 2.
train_and_register("Versione 2: Ottimizzazione performance", lr=0.001)

# MlflowClient è la classe principale per interagire programmaticamente con il Model Registry
# senza passare necessariamente dalle run di tracciamento.
client = MlflowClient()

# Ogni versione nel registry può trovarsi in uno 'Stage' specifico.
# Spostiamo la v1 in 'Staging' per indicare che è pronta per i test pre-produzione.
client.transition_model_version_stage(
    name=model_name,
    version=1,
    stage="Staging"
)

# Promozione della v2 direttamente in 'Production'. 
# In un ambiente aziendale, questo passaggio avverrebbe solo dopo il superamento di test automatizzati.
client.transition_model_version_stage(
    name=model_name,
    version=2,
    stage="Production"
)

# --- FASE 2: RESILIENZA OPERATIVA E ROLLBACK ---

def handle_rollback(faulty_version, stable_version):
    """
    Gestisce una situazione di emergenza. Se la versione attualmente in produzione (faulty_version)
    presenta problemi, viene archiviata e sostituita istantaneamente dalla versione precedente (stable_version).
    """
    print(f"\nAVVISO: Rilevato comportamento anomalo nella versione {faulty_version}!")
    
    # Rimuoviamo la versione problematica dallo stage di produzione spostandola in Archivio.
    client.transition_model_version_stage(
        name=model_name,
        version=faulty_version,
        stage="Archived"
    )
    
    # Operazione di ripristino: promuoviamo la versione stabile precedente a Production.
    client.transition_model_version_stage(
        name=model_name,
        version=stable_version,
        stage="Production"
    )
    print(f"Procedura di rollback ultimata. La versione {stable_version} è ora attiva in produzione.")

# Simulazione di un errore in produzione sulla v2 e ritorno immediato alla v1.
handle_rollback(faulty_version=2, stable_version=1)

# --- FASE 3: TRACCIABILITÀ, METADATI E LINEAGE ---

# È fondamentale documentare perché una certa versione è in un determinato stato.
# Aggiorniamo la descrizione della v1 per riflettere l'avvenuto rollback.
client.update_model_version(
    name=model_name,
    version=1,
    description="Versione ripristinata come fallback dopo criticità su v2. Baseline: dataset 2026_V1."
)

# I Tag permettono di aggiungere metadati chiave-valore per ricerche o audit rapidi.
# In questo caso specifichiamo il team responsabile per la governance del modello.
client.set_model_version_tag(
    name=model_name,
    version=1,
    key="proprietario",
    value="Team_MLOps_Core"
)

print("\nFlusso di lavoro completato con successo.")
print("La tracciabilità e la documentazione del modello sono state aggiornate nel registry.")
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- COMPONENTI DELLA ARCHITETTURA ---

class VisionEncoder(nn.Module):
    """
    Questa classe si occupa di estrarre le caratteristiche salienti (feature) da una immagine.
    Rappresenta il primo stadio della pipeline, trasformando i pixel grezzi in un vettore numerico.
    """
    def __init__(self):
        super().__init__()
        # Definiamo la sequenza di trasformazioni per processare l'immagine
        self.features = nn.Sequential(
            # Applica una convoluzione per estrarre pattern spaziali (bordi, forme)
            nn.Conv2d(3, 16, kernel_size=3),
            # Riduce le dimensioni spaziali a 1x1, mantenendo solo l'essenza delle feature
            nn.AdaptiveAvgPool2d((1, 1)),
            # Trasforma il tensore multidimensionale in un vettore piatto (vettore di feature)
            nn.Flatten()
        )
    
    def forward(self, x):
        # Riceve l'immagine in input e restituisce il vettore di feature estratto
        return self.features(x)

class TextClassifier(nn.Module):
    """
    Questa classe funge da classificatore. Riceve in input le feature estratte dal VisionEncoder
    e le utilizza per determinare la categoria di appartenenza (es. Positivo/Negativo).
    """
    def __init__(self):
        super().__init__()
        # Trasforma il vettore di 16 feature in 2 valori (logits) corrispondenti alle classi di output
        self.classifier = nn.Linear(16, 2)
    
    def forward(self, x):
        # Calcola i punteggi (logits) per ogni classe partendo dalle feature ricevute
        return self.classifier(x)

# --- GESTORE CENTRALIZZATO (ORCHESTRATORE) ---

class RobustSystem:
    """
    Questa classe orchestra l'intero processo: coordina il modello di visione,
    gestisce un ensemble di classificatori e applica logiche di sicurezza (fallback).
    """
    def __init__(self, vision_model, nlp_models, confidence_threshold=0.8):
        # Il modello che estrae le feature dalle immagini
        self.vision = vision_model
        # Una lista di modelli che lavorano insieme per migliorare la precisione (Ensemble)
        self.nlp_ensemble = nn.ModuleList(nlp_models)
        # Livello minimo di confidenza richiesto per accettare una previsione automatica
        self.threshold = confidence_threshold
        # Valore restituito se il sistema non è abbastanza sicuro della previsione
        self.fallback_value = "RICHIESTA_REVISIONE_UMANA"

    def predict(self, image_tensor):
        """
        Esegue l'intero flusso di lavoro: Visione -> Integrazione -> Decisione.
        """
        # Disabilita il calcolo del gradiente per risparmiare memoria durante l'inferenza
        with torch.inference_mode():
            
            # FASE 1: Estrazione delle caratteristiche visive tramite il VisionEncoder
            # Il sistema passa l'immagine al componente di visione per ottenere il vettore di feature
            visual_features = self.vision(image_tensor)
            
            # FASE 2: Classificazione tramite Ensemble (Media Pesata)
            # Interroghiamo tutti i modelli presenti nella lista per ottenere più "opinioni"
            all_probs = []
            for model in self.nlp_ensemble:
                # Ogni modello riceve le stesse feature visive estratte in precedenza
                logits = model(visual_features)
                # Trasformiamo i punteggi grezzi in probabilità (che sommano a 1)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs)
            
            # Calcoliamo la media delle probabilità fornite da tutti i modelli dell'ensemble
            # Questo riduce il rischio che un singolo errore di un modello comprometta il risultato
            avg_probs = torch.stack(all_probs).mean(dim=0)
            
            # Identifichiamo la classe con la probabilità più alta e il relativo valore di confidenza
            conf, pred = torch.max(avg_probs, dim=1)
            
            # FASE 3: Logica di Controllo Qualità (Fallback)
            # Se la confidenza è inferiore alla soglia impostata, attiviamo il fallback umano
            if conf.item() < self.threshold:
                return self.fallback_value, conf.item()
            
            # Altrimenti restituiamo la previsione del sistema
            return pred.item(), conf.item()

# --- ESEMPIO DI ESECUZIONE ---

# 1. Creazione dei componenti fondamentali
modello_visione = VisionEncoder()

# 2. Preparazione dell'ensemble: utilizziamo più istanze del classificatore per maggiore robustezza
# Qui creiamo una lista di due modelli indipendenti che verranno consultati dal sistema
lista_modelli_classificazione = [TextClassifier(), TextClassifier()]

# 3. Inizializzazione del sistema robusto con una soglia di attendibilità del 75%
# L'orchestratore riceve il vision encoder e la lista dei classificatori
sistema = RobustSystem(modello_visione, lista_modelli_classificazione, confidence_threshold=0.75)

# 4. Generazione di un dato di esempio (simula un'immagine a colori di 32x32 pixel)
# La forma [1, 3, 32, 32] rappresenta: [batch_size, canali_RGB, altezza, larghezza]
immagine_esempio = torch.randn(1, 3, 32, 32)

# 5. Esecuzione della previsione attraverso l'intero sistema orchestrato
# La funzione predict gestisce internamente il passaggio di dati tra visione e classificazione
risultato, punteggio_confidenza = sistema.predict(immagine_esempio)

# 6. Visualizzazione dell'output finale
# Mostriamo se il sistema ha deciso in autonomia o se è necessario l'intervento umano
print(f"Esito del sistema: {risultato} (Livello di confidenza: {punteggio_confidenza:.2f})")
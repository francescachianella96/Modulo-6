import random

def get_dirty_ocr_output():
    """
    Simula l'output di un modello OCR che restituisce una stringa 
    con rumore visivo (caratteri speciali e spazi extra).
    """
    sample_texts = ["Fattura #123", "Codice Prodotto: AB-99", "Totale: 150.00 EUR", "Data: 01/01/2024"]
    base_text = random.choice(sample_texts)
    noise = ["#", "!", "@", "  ", "??"]
    # Aggiunge caratteri di rumore in posizioni casuali
    return f"{random.choice(noise)} {base_text} {random.choice(noise)}"

def model_optimistic():
    """
    Modello fittizio 'Ottimista': tende a restituire punteggi di confidenza elevati.
    """
    # Genera un valore casuale tra 0.4 e 1.0
    return random.uniform(0.4, 1.0)

def model_conservative():
    """
    Modello fittizio 'Conservativo': tende a essere più prudente nelle stime.
    """
    # Genera un valore casuale tra 0.1 e 0.7
    return random.uniform(0.1, 0.7)

def ensemble_predict(conf_optimistic, conf_conservative, w_opt=0.4, w_cons=0.6):
    """
    Esegue una media pesata delle confidenze fornite dai due modelli.
    Viene data maggiore importanza al modello conservativo (peso 0.6).
    """
    weighted_avg = (conf_optimistic * w_opt) + (conf_conservative * w_cons)
    
    # Logica di fallback: se la confidenza combinata è bassa, richiede intervento umano
    if weighted_avg < 0.5:
        return weighted_avg, "Revisione Umana Necessaria"
    
    return weighted_avg, "Previsione Accettata"

def run_simulation(num_iterations=10):
    """
    Esegue la pipeline su un set di esempi e conta gli interventi umani necessari.
    """
    human_intervention_count = 0
    
    print(f"--- Inizio Simulazione su {num_iterations} esempi ---\n")
    
    for i in range(1, num_iterations + 1):
        # 1. Simulazione output OCR sporco
        ocr_text = get_dirty_ocr_output()
        
        # 2. Ottenimento predizioni (confidenza) dai modelli fittizi
        conf_opt = model_optimistic()
        conf_cons = model_conservative()
        
        # 3. Ensemble e logica di fallback
        avg_conf, status = ensemble_predict(conf_opt, conf_cons)
        
        # 4. Conteggio interventi
        if status == "Revisione Umana Necessaria":
            human_intervention_count += 1
            
        print(f"Esempio {i}:")
        print(f"  OCR Output: '{ocr_text}'")
        print(f"  Confidenza Ensemble: {avg_conf:.2f}")
        print(f"  Stato: {status}\n")
        
    print("--- Risultati Finali ---")
    print(f"Totale interventi umani richiesti: {human_intervention_count} su {num_iterations}")

if __name__ == "__main__":
    run_simulation(10)
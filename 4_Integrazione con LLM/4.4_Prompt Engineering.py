def genera_prompt_analista(recensione_utente):
    """
    Genera un prompt strutturato per l'analisi del sentiment
    utilizzando tecniche di role-prompting e delimitatori.
    """
    
    # Definizione della Persona e del Task
    persona = "Agisci come un esperto Analista Marketing specializzato nel comportamento dei consumatori."
    
    # Definizione del formato di output richiesto
    formato_output = {
        "sentiment": "positivo | negativo | neutro",
        "motivi": ["lista di stringhe"]
    }

    # Costruzione del prompt tramite f-string
    prompt = f"""
{persona}

Il tuo compito è analizzare la recensione del cliente delimitata da triple virgolette.

### ISTRUZIONI:
1. Analizza il tono emotivo e identifica i motivi principali citati.
2. Restituisci il risultato esclusivamente in formato JSON con le seguenti chiavi: "sentiment" e "motivi".
3. **CLAUSOLA DI SALVAGUARDIA**: Se il testo della recensione è ambiguo, privo di senso, troppo breve o non contiene abbastanza informazioni per un'analisi accurata, restituisci categoricamente:
   {{"sentiment": "non classificabile", "motivi": []}}

### RECENSIONE:
\"\"\"{recensione_utente}\"\"\"

### OUTPUT ATTESO (JSON):
"""
    return prompt

# Esempio di utilizzo:
testo_recensione = "Il prodotto è arrivato in tempo, ma la qualità dei materiali è pessima. Non lo ricomprerei."
prompt_finale = genera_prompt_analista(testo_recensione)

print(prompt_finale)
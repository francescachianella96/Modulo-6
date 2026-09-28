import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

# 1. CONFIGURAZIONE DEL MODELLO E DEL TOKENIZER
# Inizializziamo il modello GPT-2, un sistema di linguaggio pre-addestrato ottimizzato per la generazione di testo.
# Il tokenizer ha il compito di convertire il testo in sequenze numeriche (token) elaborabili dal modello.
# AutoModelForCausalLM carica i pesi e l'architettura necessaria per prevedere il token successivo.
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. DEFINIZIONE DEI VINCOLI GRAMMATICALI: FILTRO PER LE LETTERE MAIUSCOLE
# Questa classe eredita da LogitsProcessor e interviene direttamente nel processo decisionale del modello.
class NoUppercaseProcessor(LogitsProcessor):
    """
    Il processore modifica la distribuzione di probabilità (logits) prima che venga scelto il token successivo.
    L'obiettivo è forzare il modello a evitare qualsiasi parola che contenga lettere maiuscole.
    """
    def __call__(self, input_ids, scores):
        # Analizziamo l'intero dizionario dei token disponibili per identificare quelli non desiderati.
        for token_id in range(len(scores[0])):
            # Convertiamo l'ID numerico del token in testo per verificarne la capitalizzazione.
            token_text = tokenizer.decode([token_id])
            
            # Se il token contiene almeno un carattere maiuscolo, assegniamo un punteggio di meno infinito.
            # Questo rende impossibile la selezione di quel token durante la fase di campionamento.
            if any(char.isupper() for char in token_text):
                scores[:, token_id] = float('-inf') 
        return scores

# 3. DEFINIZIONE DEI CRITERI DI STOP: INTERRUZIONE AL PUNTO FERMO
# Questa classe eredita da StoppingCriteria e definisce una condizione logica di uscita.
class StopAtPeriodCriteria(StoppingCriteria):
    """
    Il criterio monitora il flusso di generazione e invia un segnale di arresto quando viene prodotto un punto.
    Questo meccanismo è essenziale per garantire che l'output si fermi in modo naturale a fine frase.
    """
    def __call__(self, input_ids, scores, **kwargs):
        # Recuperiamo l'ultimissimo token generato dal modello in questo passaggio.
        last_token = tokenizer.decode(input_ids[0][-1])
        
        # Se il token contiene un punto fermo, restituiamo True per terminare immediatamente il ciclo.
        return "." in last_token

# 4. PREPARAZIONE DELLA GENERAZIONE IN STREAMING
# Definiamo la frase di partenza (prompt) e la trasformiamo in tensori PyTorch comprensibili dal modello.
prompt = "Some thoughts about technology, Apple vs Microsoft:"
inputs = tokenizer(prompt, return_tensors="pt")

# TextStreamer gestisce la visualizzazione progressiva del testo sul terminale (token per token).
# Questo approccio riduce la latenza percepita, permettendo di leggere l'output mentre viene creato.
streamer = TextStreamer(tokenizer, skip_prompt=True)

# 5. COORDINAZIONE E INTERAZIONE TRA I COMPONENTI
# In questa fase assembliamo le classi create sopra nelle strutture dati richieste dal metodo generate().
# - LogitsProcessorList: implementa NoUppercaseProcessor per controllare COSA viene scritto (solo minuscolo).
# - StoppingCriteriaList: implementa StopAtPeriodCriteria per controllare QUANDO fermarsi (al primo punto).
logits_processors = LogitsProcessorList([NoUppercaseProcessor()])
stopping_criteria = StoppingCriteriaList([StopAtPeriodCriteria()])

print(f"Prompt iniziale: {prompt}")
print("Risposta (Generazione vincolata in minuscolo e streaming):")

# 6. ESECUZIONE DEL PROCESSO DI GENERAZIONE
# Il metodo generate() orchestra l'interazione tra modello, filtri e criteri di stop:
# 1. Calcola le probabilità dei token.
# 2. Applica NoUppercaseProcessor per filtrare le maiuscole.
# 3. Seleziona un token (do_sample).
# 4. Invia il token allo streamer per la visualizzazione immediata.
# 5. Interroga StopAtPeriodCriteria per decidere se continuare o fermarsi.
with torch.no_grad():
    model.generate(
        **inputs,
        streamer=streamer,                  # Visualizza il testo in tempo reale.
        logits_processor=logits_processors,     # Forza il vincolo sulle minuscole.
        stopping_criteria=stopping_criteria,     # Gestisce la chiusura automatica della frase.
        max_new_tokens=100,                 # Numero massimo di nuovi termini generabili.
        do_sample=True,                     # Introduce varietà nella risposta.
        temperature=0.7,                    # Bilancia fedeltà e creatività.
        pad_token_id=tokenizer.eos_token_id
    )
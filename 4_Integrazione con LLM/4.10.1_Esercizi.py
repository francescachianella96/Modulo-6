# Esercizio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

# 1. CONFIGURAZIONE DEL MODELLO E DEL TOKENIZER
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. SFIDA: VINCOLO MANUALE ANTI-NUMERI (LogitsProcessor)
class AvoidNumbersProcessor(LogitsProcessor):
    """
    Questo processore implementa il Logit Masking. 
    Azzera la probabilità di ogni token che contenga una cifra (0-9).
    """
    def __call__(self, input_ids, scores):
        # Iteriamo sull'intero vocabolario per identificare i token numerici
        for token_id in range(len(scores[0])):
            token_text = tokenizer.decode([token_id])
            
            # Se il token contiene un numero, lo rendiamo impossibile da scegliere
            if any(char.isdigit() for char in token_text):
                scores[:, token_id] = float('-inf') 
        return scores

# 3. CRITERIO DI STOP: PUNTO ESCLAMATIVO (StoppingCriteria)
class StopAtExclamationCriteria(StoppingCriteria):
    """
    Interrompe la generazione non appena viene rilevato un punto esclamativo.
    Utile per controllare la terminazione e risparmiare risorse computazionali.
    """
    def __call__(self, input_ids, scores, **kwargs):
        # Decodifichiamo l'ultimo token prodotto nell'ultimo passo di generazione
        last_token = tokenizer.decode(input_ids[0][-1])
        
        # Restituiamo True per fermare il loop se troviamo '!'
        return "!" in last_token

# 4. PREPARAZIONE DELLA GENERAZIONE IN STREAMING
prompt = "Tell me a short story about the number 42:"
inputs = tokenizer(prompt, return_tensors="pt")

# TextStreamer visualizza i token sulla console in tempo reale (Emissione asincrona)
streamer = TextStreamer(tokenizer, skip_prompt=True)

# 5. COORDINAZIONE DEI COMPONENTI
# Inseriamo i nostri vincoli nelle liste dedicate
logits_processors = LogitsProcessorList([AvoidNumbersProcessor()])
stopping_criteria = StoppingCriteriaList([StopAtExclamationCriteria()])

print(f"Prompt iniziale: {prompt}")
print("-" * 30)
print("Risposta in streaming (Senza numeri e stop al primo '!'):")

# 6. ESECUZIONE DEL PROCESSO DI GENERAZIONE
with torch.no_grad():
    model.generate(
        **inputs,
        streamer=streamer,
        logits_processor=logits_processors,
        stopping_criteria=stopping_criteria,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id
    )
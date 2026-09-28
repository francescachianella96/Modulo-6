import torch
import re
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig

# 1. DEFINIZIONE E PREPARAZIONE DEI DATI
# Il dataset viene creato inizialmente come una lista di dizionari che rappresentano esempi 
# di istruzioni, contesti e risposte attese. Questa struttura è fondamentale per guidare
# il modello verso un comportamento specifico attraverso il fine-tuning supervisionato.
raw_data = [
    {"instruction": "Cos'è un tensore?", "context": "Deep Learning", "response": "Un array multidimensionale. Scrivimi a marco@tensor.it"},
    {"instruction": "Come salvo un modello?", "context": "PyTorch", "response": "Usa torch.save(). Info su docs@pytorch.org"},
    {"instruction": "Definisci il Gradiente.", "context": "Matematica", "response": "Il vettore delle derivate parziali."}
]

def preprocess_and_clean(data):
    """
    Questa funzione si occupa della pulizia dei dati per garantire la privacy e la sicurezza.
    Viene utilizzata un'espressione regolare per identificare gli indirizzi email e sostituirli
    con un testo generico [REDACTED], impedendo al modello di memorizzare dati sensibili
    durante la fase di addestramento.
    """
    email_regex = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    for entry in data:
        # La sostituzione avviene solo nel campo 'response' che contiene il testo generato.
        entry["response"] = re.sub(email_regex, "[REDACTED]", entry["response"])
    return data

# Dopo la pulizia, i dati vengono convertiti nell'oggetto Dataset della libreria Hugging Face.
# Questo oggetto permette una gestione efficiente della memoria e un'integrazione nativa
# con le classi di addestramento utilizzate successivamente.
cleaned_data = preprocess_and_clean(raw_data)
dataset = Dataset.from_list(cleaned_data)

# 2. CARICAMENTO DEL TOKENIZER
# Carichiamo il tokenizer corrispondente al modello GPT-2. Il tokenizer ha il compito di
# scomporre il testo in unità elementari (token) e convertirle in indici numerici integrabili dal modello.
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# GPT-2 non dispone di un token dedicato per il padding. Per gestire batch di testi con 
# lunghezze differenti, impostiamo il token di fine sequenza (eos_token) come token di padding.
tokenizer.pad_token = tokenizer.eos_token 

# 3. CONFIGURAZIONE LoRA (Low-Rank Adaptation)
# LoRA è una tecnica di efficienza che permette di addestrare solo delle matrici aggiuntive
# a basso rango (r=8) anziché tutti i parametri del modello. Questo riduce drasticamente
# l'uso della memoria video (VRAM) pur mantenendo alte le prestazioni.
peft_config = LoraConfig(
    r=8,                       # Rango delle matrici adattive: determina la complessità dell'adattamento.
    lora_alpha=32,             # Fattore di scala che bilancia l'influenza dei pesi LoRA sui pesi originali.
    target_modules=["c_attn"], # Specifica i layer del modello GPT-2 su cui applicare le matrici LoRA.
    lora_dropout=0.05,         # Percentuale di neuroni disattivati per evitare che il modello memorizzi i dati a memoria.
    bias="none",               # Indica di non modificare i parametri di bias per massimizzare l'efficienza.
    task_type="CAUSAL_LM",     # Definisce il compito come modellazione del linguaggio casuale (generazione testo).
    fan_in_fan_out=True        # Parametro tecnico necessario per la corretta gestione dei pesi nei layer Conv1d di GPT-2.
)

# 4. DEFINIZIONE DELLA FORMATTAZIONE DEI PROMPT
def formatting_prompts_func(example):
    """
    Questa funzione riceve un singolo elemento del dataset e lo trasforma in una stringa di testo.
    La formattazione strutturata (Istruzione-Contesto-Risposta) insegna al modello a riconoscere
    il confine tra la domanda dell'utente e il corpo della risposta generata.
    """
    text = (f"### Istruzione: {example['instruction']}\n"
            f"### Contesto: {example['context']}\n"
            f"### Risposta: {example['response']}{tokenizer.eos_token}")
    return text

# 5. CONFIGURAZIONE DEL TRAINING (SFTConfig)
# SFTConfig contiene tutti i parametri che guidano il processo di addestramento.
# Questi parametri influenzano direttamente la velocità e la qualità dell'apprendimento.
sft_config = SFTConfig(
    output_dir="./gpt2-sft-trl",      # Cartella dove verranno salvati i log e i checkpoint intermedi.
    per_device_train_batch_size=1,     # Numero di esempi elaborati per ogni iterazione sulla GPU.
    gradient_accumulation_steps=4,      # Accumula i gradienti per simulare un batch più grande e stabile.
    learning_rate=2e-4,                # Velocità con cui il modello aggiorna i propri pesi.
    logging_steps=1,                   # Intervallo di passi per mostrare l'andamento della perdita (loss).
    num_train_epochs=3,                # Numero totale di volte in cui il modello vedrà l'intero dataset.
    save_strategy="epoch",             # Salva lo stato del modello alla fine di ogni epoca di addestramento.
    fp16=True,                         # Utilizza precisione a 16 bit per velocizzare il calcolo e risparmiare VRAM.
    push_to_hub=False,                 # Evita il caricamento automatico su repository remoti durante il test.
    max_length=512,                    # Lunghezza massima consentita per ogni sequenza di input.
    packing=False,                     # Mantiene ogni esempio separato per evitare interferenze tra risposte diverse.
    dataset_text_field="text"          # Identifica il campo testuale all'interno del dataset elaborato dal trainer.
)

# 6. INIZIALIZZAZIONE E AVVIO DEL PROCESSO DI TRAINING
# SFTTrainer è l'orchestratore principale: collega il modello base, il dataset preparato,
# la configurazione LoRA e i parametri di training stabiliti in SFTConfig.
trainer = SFTTrainer(
    model=model_id,
    train_dataset=dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    args=sft_config,
)

# Avvio effettivo del ciclo di addestramento basato sui dati forniti.
trainer.train()

# 6.1 SALVATAGGIO DEFINITIVO DEGLI ADATTATORI
# Al termine del training, è fondamentale salvare i pesi addestrati (adapter) in una cartella specifica.
# Questo permette di ricaricare solo le modifiche apprese senza dover rielaborare l'intero modello.
final_adapter_path = "gpt2_lora_final"
trainer.save_model(final_adapter_path) 
print(f"Modello salvato correttamente in: {final_adapter_path}")

# 7. CARICAMENTO DEL MODELLO E TEST DI INFERENZA
# In questa fase verifichiamo che il modello abbia effettivamente appreso dai dati forniti.
# Per farlo, carichiamo il modello originale e applichiamo gli "adapter" LoRA appena creati.
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(model_id)

# La classe PeftModel si occupa di "iniettare" gli adattatori caricati sopra il modello base.
model = PeftModel.from_pretrained(base_model, final_adapter_path)
model.to("cuda" if torch.cuda.is_available() else "cpu") # Utilizza la GPU se disponibile per la generazione.
model.eval() # Disattiva i layer di dropout e normalizzazione per una generazione coerente.

# Creiamo un prompt di prova per osservare come il modello risponde a una nuova interrogazione.
prompt = "### Istruzione: Cos'è un tensore?\n### Contesto: Deep Learning\n### Risposta:"

# Trasformiamo il prompt in tensori e chiediamo al modello di generare il completamento del testo.
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    output_tokens = model.generate(
        **inputs, 
        max_new_tokens=50,       # Numero massimo di nuovi token da generare.
        temperature=0.7,         # Livello di creatività della generazione (0.7 è un buon equilibrio).
        do_sample=True,          # Abilita il campionamento probabilistico per variare le risposte.
        pad_token_id=tokenizer.eos_token_id
    )

# Decodifichiamo i token numerici prodotti dal modello in testo leggibile dall'uomo.
generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print("\n" + "="*30)
print("--- TEST INFERENZA SUCCESSIVO AL TRAINING ---")
print(generated_text)
print("="*30)
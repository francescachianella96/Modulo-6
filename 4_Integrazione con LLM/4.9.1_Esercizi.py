import torch
import re
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig


# --- 1. DATASET IN INGLESE ---
# I pesi pre-addestrati di GPT-2 sono ottimizzati per l'inglese. Aspettiamoci una migliore coerenza.
raw_data = [
    {"instruction": "What is a tensor?", "context": "Deep Learning", "response": "A multi-dimensional array. Contact me at marco@tensor.it"},
    {"instruction": "How do I save a model?", "context": "PyTorch", "response": "Use torch.save(). Check docs at docs@pytorch.org"},
    {"instruction": "Define Gradient.", "context": "Mathematics", "response": "The vector of partial derivatives."}
]

def preprocess_and_clean(data):
    email_regex = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    for entry in data:
        entry["response"] = re.sub(email_regex, "[REDACTED]", entry["response"])
    return data

cleaned_data = preprocess_and_clean(raw_data)
dataset = Dataset.from_list(cleaned_data)

# --- 2. TOKENIZER E MODELLO ---
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token 

# --- 3. CONFIGURAZIONE LoRA AGGIORNATA ---
# Abbiamo aumentato lora_alpha a 64 per rendere gli aggiornamenti dell'adapter più aggressivi.
peft_config = LoraConfig(
    r=16,               # Rango aumentato per una maggiore capacità
    lora_alpha=64,      # Alpha più alto = maggiore influenza dei pesi LoRA
    target_modules=["c_attn"],
    lora_dropout=0.1,   # Dropout aumentato per una migliore generalizzazione
    bias="none",
    task_type="CAUSAL_LM",
    fan_in_fan_out=True
)

# --- 4. TEMPLATE DEL PROMPT IN INGLESE ---
def formatting_prompts_func(example):
    text = (f"### Instruction: {example['instruction']}\n"
            f"### Context: {example['context']}\n"
            f"### Response: {example['response']}{tokenizer.eos_token}")
    return text

# --- 5. SFTConfig OTTIMIZZATO ---
sft_config = SFTConfig(
    output_dir="./gpt2-sft-en",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-4,       # Learning rate aumentato per un adattamento più rapido
    logging_steps=1,
    num_train_epochs=10,      # Epoche aumentate a 10 per padroneggiare meglio il dataset
    save_strategy="epoch",
    fp16=True,
    max_length=512, 
    packing=False,
    dataset_text_field="text"
)

# --- 6. ADDESTRAMENTO ---
trainer = SFTTrainer(
    model=model_id,
    train_dataset=dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    args=sft_config,
)

trainer.train()

# Salvataggio degli adapter finali in inglese
final_adapter_path = "gpt2_lora_en_final"
trainer.save_model(final_adapter_path) 

# --- 7. TEST DI INFERENZA ---
base_model = AutoModelForCausalLM.from_pretrained(model_id)
model = PeftModel.from_pretrained(base_model, final_adapter_path)
model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# Test con un prompt simile al dataset in inglese
test_prompt = "### Instruction: What is a tensor?\n### Context: Deep Learning\n### Response:"

inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    output_tokens = model.generate(
        **inputs, 
        max_new_tokens=40, 
        temperature=0.5, # Temperatura più bassa per un output più fattuale
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

print("\n" + "="*40)
print("--- TEST DI INFERENZA IN INGLESE ---")
print(tokenizer.decode(output_tokens[0], skip_special_tokens=True))
print("="*40)
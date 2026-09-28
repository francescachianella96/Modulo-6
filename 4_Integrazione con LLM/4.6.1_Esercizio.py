# Esercizio

from transformers import pipeline, set_seed

# Fissiamo il seed per rendere i risultati riproducibili (fondamentale nel DL)
set_seed(42)

# 1. Inizializzazione della Pipeline
# Carichiamo gpt2.
# perfetto per studiare questi meccanismi perché è leggero e trasparente.
generator = pipeline('text-generation', model='gpt2', device=0, framework='pt')

prompt = "Il futuro dell'intelligenza artificiale è"

# 2. Generazione Conservativa (Temperature bassa)
output_safe = generator(
    prompt, 
    max_new_tokens=40, 
    temperature=0.2, 
    do_sample=True,
    truncation=True
)[0]['generated_text']

# 3. Generazione Creativa (Temperature alta + Top-p)
output_creative = generator(
    prompt, 
    max_new_tokens=40, 
    temperature=1.5, 
    top_p=0.9, 
    do_sample=True,
    truncation=True
)[0]['generated_text']

print(f"--- OUTPUT CONSERVATIVO (T=0.2) ---\n{output_safe}\n")
print(f"--- OUTPUT CREATIVO (T=1.5, Top-p=0.9) ---\n{output_creative}")
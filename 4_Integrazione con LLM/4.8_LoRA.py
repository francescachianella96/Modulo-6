import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

# Configurazione dell'hardware: viene data priorità alla GPU (CUDA) se disponibile.
# L'uso di una GPU è fondamentale per garantire tempi di generazione accettabili con modelli complessi come SDXL.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Identificativo del modello base di Stable Diffusion XL 1.0.
model_id = "stabilityai/stable-diffusion-xl-base-1.0"

# --- 1. CARICAMENTO DEL MODELLO BASE ---
# La classe DiffusionPipeline gestisce l'intero workflow di generazione (modello UNet, VAE, text encoders).
# Utilizziamo torch.float16 (precisione dimezzata) per ridurre drasticamente il consumo di memoria video (VRAM)
# senza compromettere visibilmente la qualità dell'immagine finale.
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# Spostiamo l'intera pipeline sulla memoria della GPU (o CPU) per l'esecuzione.
pipe.to(device)

# Definizione del prompt: la descrizione testuale che guida il processo di diffusione.
prompt = "A futuristic city with flying cars, sunset, cinematic lighting"

print("--- GENERAZIONE CON MODELLO BASE ---")
# Il metodo __call__ della pipeline avvia il processo di denoising iterativo.
# num_inference_steps definisce quanti passaggi di campionamento verranno effettuati (valore tipico per SDXL è tra 30 e 50).
# .images[0] estrae la prima immagine generata dalla lista dei risultati.
image_base = pipe(prompt, num_inference_steps=30).images[0]

# Salvataggio dell'immagine prodotta dal modello originale sul disco.
image_base.save("city_base.png")

# --- 2. INTEGRAZIONE DEL MODULO LORA (Low-Rank Adaptation) ---
# LoRA permette di modificare il comportamento del modello iniettando piccoli pesi addizionali (matrici A e B)
# negli strati di attenzione (Attention Layers), senza dover ricaricare l'intero modello pesante.
# Questo esempio carica uno stile specifico (Pixel Art).
lora_model_id = "nerjs/pixel-art-xl"

# Il metodo load_lora_weights scarica i pesi LoRA e li applica dinamicamente alla pipeline esistente.
# Questa operazione è molto più veloce rispetto al caricamento di un nuovo modello completo.
pipe.load_lora_weights(lora_model_id)

print("\n--- GENERAZIONE CON LORA ATTIVO ---")
# Eseguiamo nuovamente la generazione con lo stesso prompt. 
# La pipeline ora utilizzerà i pesi base "corretti" dai pesi LoRA, applicando lo stile pixel art al contenuto.
image_lora = pipe(prompt, num_inference_steps=30).images[0]

# Salvataggio della versione stilizzata con LoRA.
image_lora.save("city_pixel_art.png")

# --- 3. GESTIONE DELLA MODULARITA ---
# Una delle caratteristiche principali di questo approccio è la possibilità di scaricare i pesi aggiuntivi.
# Il metodo unload_lora_weights rimuove i pesi applicati in precedenza, ripristinando lo stato originale del modello.
# Questo permette di passare da uno stile all'altro mantenendo lo stesso oggetto pipeline in memoria.
pipe.unload_lora_weights()
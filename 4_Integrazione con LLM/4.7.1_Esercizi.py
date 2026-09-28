import torch
import torch.nn.functional as F
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# 1. Setup e caricamento del modello CLIP
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

# Prepariamo l'immagine del gatto e le tre frasi
url = "http://images.cocodataset.org/val2017/000000039769.jpg" # Foto di gatti
image = Image.open(requests.get(url, stream=True).raw)
labels = ["un gatto", "un cane", "un tramonto"]

# 2. Estrazione degli Embedding
inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    # CLIP restituisce già i vettori proiettati nello spazio condiviso
    image_embeds = outputs.image_embeds  # Shape: [1, 512]
    text_embeds = outputs.text_embeds    # Shape: [3, 512]

# 3. Calcolo della Similitudine Coseno
# Usiamo torch.nn.functional per calcolare la similitudine tra l'immagine e ogni testo
# Espandiamo il vettore immagine per confrontarlo con tutti i testi contemporaneamente
similarities = F.cosine_similarity(image_embeds, text_embeds)

# Risultati
print(f"Risultati della similitudine:")
for label, score in zip(labels, similarities):
    print(f" - {label}: {score.item():.4f}")

# Identificazione del vincitore
best_idx = similarities.argmax().item()
print(f"\nLa frase con lo score più alto è: '{labels[best_idx]}'")
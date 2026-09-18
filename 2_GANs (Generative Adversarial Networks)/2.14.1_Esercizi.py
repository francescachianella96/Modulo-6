# Esercizio
# Utilizzando un modello pre-addestrato caricare un modello di inpainting
# Espandere l'immagine e riempire il nuovo spazio vuoto, valutando visivamente la coerenza del risultato

import torch
import requests
from PIL import Image, ImageDraw, ImageFilter
from io import BytesIO
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline


# 1. Configurazione del dispositivo e inizializzazione del modello
device = "cuda" if torch.cuda.is_available() else "cpu"

# Utilizziamo il modello pre-addestrato specifico per l'inpainting fornito da RunwayML.
model_id = "runwayml/stable-diffusion-inpainting"

# Inizializziamo la pipeline di inpainting. 
# Se disponibile CUDA, carichiamo il modello in float16 (mezza precisione) per risparmiare memoria VRAM.
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)

# 2. Acquisizione dell'immagine originale
# Scarichiamo un'immagine di esempio da Unsplash, la convertiamo in spazio colore RGB 
# e la ridimensioniamo a una risoluzione standard di 512x512 pixel.
url = "https://images.unsplash.com/photo-1476900543704-4312b78632f8?q=80&w=512&auto=format&fit=crop"
init_image = Image.open(BytesIO(requests.get(url).content)).convert("RGB").resize((512, 512))

# 3. Preparazione della base per l'outpainting (Extended Background)
# Per ottenere un'estensione coerente, non usiamo uno sfondo nero, ma creiamo una "guida cromatica".
# Definiamo la nuova dimensione (768px di larghezza per 512px di altezza).
new_width, height = 768, 512

# Creiamo lo sfondo esteso partendo dall'immagine originale, ridimensionandola e applicando 
# un filtro di sfocatura (Gaussian Blur). Questo fornisce al modello suggerimenti sui colori
# e le forme da generare nelle nuove aree.
expanded_bg = init_image.resize((new_width, height), Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(radius=20))

# Incolliamo l'immagine originale nitida al centro dello sfondo sfocato.
# Lo shift di 128 pixel centra l'immagine originale di 512 pixel all'interno della base da 768.
expanded_bg.paste(init_image, (128, 0))

# 4. Creazione della Maschera di Inpainting (Soft Mask)
# La maschera indica al modello quali parti dell'immagine devono essere rigenerate.
# Il valore 255 (bianco) indica aree da generare, 0 (nero) aree da preservare.
mask_image = Image.new("L", (new_width, height), 255)
draw = ImageDraw.Draw(mask_image)

# Definiamo l'area centrale da preservare (dove si trova l'immagine nitida).
# Lasciamo un piccolo margine di sovrapposizione (overlap) di 10 pixel rispetto ai bordi
# dell'immagine originale per facilitare la fusione (blending) tra vecchio e nuovo.
draw.rectangle([138, 0, 128 + 512 - 10, height], fill=0)

# Applichiamo una sfocatura alla maschera. Questo crea una transizione graduale nei valori
# dei pixel tra l'area preservata e quella generata, evitando stacchi netti visibili.
mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=10))

# 5. Esecuzione del processo di generazione
# Definiamo un prompt testuale che descrive la scena completa che vogliamo ottenere.
prompt = "landscape photography, a wide road continuing into a dense forest, high mountains on the horizon, cinematic lighting, ultra-realistic"

# Interazione con il modello:
# - image: Passiamo lo sfondo esteso che funge da guida iniziale (expanded_bg).
# - mask_image: Indica dove il modello deve effettivamente intervenire.
# - strength: Determina quanto il modello può discostarsi dall'immagine di input nelle aree mascherate.
# - guidance_scale: Controlla quanto la generazione deve aderire strettamente al prompt testuale.
output = pipe(
    prompt=prompt,
    image=expanded_bg,
    mask_image=mask_image,
    strength=0.85,
    guidance_scale=7.5
).images[0]

# 6. Salvataggio del risultato finale
output.save("outpainting_match_perfetto.png")
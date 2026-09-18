import torch
import requests
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline


# Configurazione del dispositivo
# Viene selezionata la GPU se disponibile (cuda) per massimizzare le prestazioni,
# altrimenti si utilizza la CPU (che sarà significativamente più lenta).
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Caricamento del modello di Inpainting
# Viene utilizzata la classe StableDiffusionInpaintPipeline dalla libreria diffusers.
# Questa classe gestisce l'intero processo di generazione, scaricando i pesi del modello pre-addestrato.
# model_id indica il repository su Hugging Face da cui scaricare il modello.
model_id = "runwayml/stable-diffusion-inpainting"

# from_pretrained scarica e inizializza il modello.
# torch_dtype=torch.float16 viene utilizzato per ridurre l'occupazione di memoria (half precision),
# utile soprattutto se si utilizza una GPU con memoria limitata.
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
).to(device)

# 2. Preparazione dell'immagine originale
# Scarichiamo un'immagine di esempio da Unsplash (una strada immersa nella natura).
# Unsplash è un'ottima fonte per immagini di alta qualità gratuite.
url = "https://images.unsplash.com/photo-1476900543704-4312b78632f8?q=80&w=512&auto=format&fit=crop"
response = requests.get(url)

# Verifichiamo che il download sia andato a buon fine
if response.status_code != 200:
    raise Exception(f"Errore nel scaricare l'immagine. Status code: {response.status_code}")

# Carichiamo l'immagine dai byte scaricati usando PIL (Python Imaging Library).
# .convert("RGB") assicura che l'immagine abbia 3 canali di colore (Rosso, Verde, Blu).
# .resize((512, 512)) ridimensiona l'immagine a 512x512 pixel, che è la risoluzione standard
# per questo modello specifico di Stable Diffusion.
init_image = Image.open(BytesIO(response.content)).convert("RGB").resize((512, 512))

# 3. Creazione della Maschera (Mask)
# La maschera è un'immagine in scala di grigi che indica al modello quale parte dell'immagine originale
# deve essere rigenerata. I pixel bianchi (valore 255) rappresentano l'area da modificare,
# mentre i pixel neri (valore 0) rappresentano l'area da mantenere intatta.

# Creiamo un'immagine nera di dimensioni 512x512. "L" indica la modalità a scala di grigi (Luminance).
mask_image = Image.new("L", (512, 512), 0)

# Utilizziamo ImageDraw per disegnare sulla maschera.
draw = ImageDraw.Draw(mask_image)

# Disegniamo un rettangolo bianco al centro dell'immagine.
# Le coordinate sono [x_min, y_min, x_max, y_max].
# Questa area bianca sarà quella che il modello andrà a riempire e modificare.
draw.rectangle([150, 150, 350, 350], fill=255)

# 4. Esecuzione dell'Inpainting
# Definiamo il prompt testuale che guiderà la generazione nella zona mascherata.
# Il modello cercherà di generare un contenuto che si integri con l'immagine circostante
# seguendo la descrizione fornita.
prompt = "a professional photo of a smooth road, seamless integration"

# Chiamiamo la pipeline 'pipe' passando:
# - prompt: la descrizione di cosa generare.
# - image: l'immagine originale di partenza.
# - mask_image: la maschera che definisce l'area di intervento.
# La chiamata restituisce un oggetto che contiene una lista di immagini generate (qui prendiamo la prima con [0]).
output = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]

# 5. Valutazione Quantitativa (MSE)
# Definiamo una funzione per calcolare l'Errore Quadratico Medio (MSE) tra l'immagine originale e quella generata,
# considerando solo l'area che è stata modificata (il "buco").
def calculate_hole_diff(original, generated, mask):
    # Convertiamo le immagini PIL in array NumPy per poter eseguire calcoli matematici.
    # astype(np.float32) converte i valori interi (0-255) in numeri decimali per maggiore precisione.
    orig_arr = np.array(original).astype(np.float32)
    gen_arr = np.array(generated).astype(np.float32)
    
    # La maschera viene normalizzata dividendo per 255.0, ottenendo valori tra 0 (nero) e 1 (bianco).
    mask_arr = np.array(mask).astype(np.float32) / 255.0
    
    # La maschera originale ha shape (512, 512), ma le immagini sono RGB (512, 512, 3).
    # Dobbiamo espandere la maschera per avere la stessa dimensionalità delle immagini
    # in modo da poterla applicare a tutti e tre i canali colore.
    # np.newaxis aggiunge una dimensione, np.repeat ripete i valori lungo quella dimensione.
    mask_3d = np.repeat(mask_arr[:, :, np.newaxis], 3, axis=2)
    
    # Calcoliamo la differenza pixel per pixel tra l'originale e la generata.
    # Moltiplichiamo per la maschera: i pixel fuori dal buco (mask=0) diventeranno 0,
    # quindi non influiranno sul calcolo dell'errore. Consideriamo solo l'area modificata.
    diff = (orig_arr - gen_arr) * mask_3d
    
    # Calcoliamo la media del quadrato delle differenze (MSE).
    mse = np.mean(np.square(diff))
    return mse

# Calcoliamo l'MSE utilizzando l'immagine iniziale, l'output generato e la maschera.
mse_value = calculate_hole_diff(init_image, output, mask_image)

# Stampiamo il risultato a video.
print(f"Errore Quadratico Medio (MSE) nell'area ricostruita: {mse_value:.2f}")

# 6. Salvataggio dei Risultati
# Salviamo le tre immagini (risultato, originale, maschera) su disco per permettere
# un confronto visivo diretto.
output.save("risultato_inpainting.png")
init_image.save("originale.png")
mask_image.save("maschera.png")
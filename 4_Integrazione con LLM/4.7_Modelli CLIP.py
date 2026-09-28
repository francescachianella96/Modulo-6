import torch
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# CONFIGURAZIONE DEL MODELLO
# Carichiamo CLIP (Contrastive Language-Image Pre-training), un modello in grado di 
# mappare testo e immagini in uno spazio vettoriale condiviso.
model_id = "openai/clip-vit-base-patch32"

# CLIPModel: La classe principale che contiene i pesi del modello neurale.
# CLIPProcessor: Gestisce la preparazione dei dati (tokenizzazione testo e pre-processamento immagini).
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

# PREPARAZIONE DEL DATABASE DI IMMAGINI
# Definiamo una lista di URL che rappresentano le immagini nelle quali vogliamo effettuare la ricerca.
urls = [
    "http://images.cocodataset.org/val2017/000000039769.jpg", # Rappresenta gatti
    "http://images.cocodataset.org/val2017/000000281759.jpg", 
]

# Scarichiamo le immagini tramite requests e le convertiamo in oggetti PIL (Python Imaging Library).
# Questa conversione è necessaria perché il CLIPProcessor accetta oggetti PIL o array NumPy.
images = [Image.open(requests.get(url, stream=True).raw) for url in urls]

# Definizione della query di ricerca testuale.
query_testuale = "una foto accogliente di piccoli felini domestici"

# ESTRAZIONE DELLE CARATTERISTICHE (EMBEDDING)
# Il CLIPProcessor trasforma il testo "grezzo" in token e le immagini in tensori normalizzati.
# return_tensors="pt" indica che vogliamo ottenere tensori PyTorch come output.
inputs = processor(text=[query_testuale], images=images, return_tensors="pt", padding=True)

# Disabilitiamo il calcolo dei gradienti (torch.no_grad) per risparmiare memoria durante l'inferenza.
with torch.no_grad():
    # Passiamo gli input processati al modello CLIP. 
    # L'operatore ** esegue lo scompattamento del dizionario 'inputs' negli argomenti corretti del modello.
    outputs = model(**inputs)
    
    # Il modello produce due vettori (embedding) fondamentali:
    # image_embeds: Rappresentazione numerica delle caratteristiche visive [NumeroImmagini, 512]
    # text_embeds: Rappresentazione numerica del significato semantico del testo [1, 512]
    image_embeds = outputs.image_embeds 
    text_embeds = outputs.text_embeds   

# CALCOLO DELLA SIMILITUDINE SEMANTICA
# Per confrontare i vettori, dobbiamo prima normalizzarli (portare la loro lunghezza a 1).
# Questo permette di utilizzare il prodotto scalare come misura di Cosine Similarity.
image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

# Calcolo dell'affinità tramite prodotto matriciale (@). 
# Moltiplichiamo il vettore di testo per la trasposta del vettore delle immagini (.T).
# Risultato: un punteggio che indica quanto il testo è "vicino" a ciascuna immagine nello spazio semantico.
scores = (text_embeds @ image_embeds.T) * 100 

# OUTPUT DEI RISULTATI
print(f"Query: '{query_testuale}'")
for i, score in enumerate(scores[0]):
    # Mostriamo il punteggio di affinità per ogni immagine caricata.
    print(f"Immagine {i+1} - Score di affinità: {score.item():.2f}")
import os
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIConnectionError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 1. CONFIGURAZIONE AMBIENTE
load_dotenv()

base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENAI_MODEL_NAME")
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(base_url=base_url, api_key=api_key)

# 2. LOGICA DI RETRY AVANZATA
# Estendiamo il retry anche agli errori di connessione fisica, non solo al rate limit.
@retry(
    retry=(retry_if_exception_type(RateLimitError) | retry_if_exception_type(APIConnectionError)),
    wait=wait_exponential(multiplier=1, min=4, max=20), # Attesa crescente
    stop=stop_after_attempt(3),
    before_sleep=lambda retry_state: print(f"Tentativo fallito. Riprovo tra pochi secondi... (N. {retry_state.attempt_number})")
)
def chiamata_api_riassunto(testo):
    """
    Interroga il modello con identità 'Redattore ANSA' e vincoli di formattazione.
    """
    return client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system", 
                "content": (
                    "Sei un esperto Redattore dell'agenzia ANSA. "
                    "Il tuo stile è giornalistico, asciutto, autorevole e oggettivo. "
                    "Restituisci sempre il riassunto in formato Markdown utilizzando "
                    "esattamente tre bullet point (punti elenco)."
                )
            },
            {
                "role": "user", 
                "content": f"Redigi un lancio d'agenzia sintetico per il seguente testo:\n\n{testo}"
            }
        ],
        temperature=0.4, # Leggermente più alto per fluidità giornalistica
        max_tokens=300
    )

def esegui_riassunto(testo):
    print(f"--- Comunicato ANSA in elaborazione con {model_name} ---")
    try:
        risposta = chiamata_api_riassunto(testo)
        return risposta.choices[0].message.content
    except Exception as e:
        return f"Errore critico durante la redazione: {e}"

# 3. TEST SU TESTO LUNGO (>500 parole)
if __name__ == "__main__":
    # Testo espanso sulla rivoluzione dell'IA nel 2026
    testo_lungo = """
    L'anno 2026 segna un punto di svolta senza precedenti nella storia dell'integrazione tecnologica globale. 
    Mentre i decenni precedenti sono stati caratterizzati da una digitalizzazione frammentata, il 2026 vede l'ascesa 
    di sistemi IA ubiqui e interconnessi che operano in tempo reale su scala planetaria. Questo fenomeno, 
    spesso definito dagli analisti come 'La Grande Convergenza', non riguarda solo la potenza di calcolo pura, 
    ma la capacità delle macchine di collaborare senza l'intervento umano diretto in settori critici come la 
    logistica, la sanità e la gestione energetica. 
    
    Nel settore industriale, le fabbriche 'Lights Out' (completamente automatizzate) sono passate dall'essere 
    una rarità a diventare lo standard produttivo nelle economie avanzate. Queste strutture utilizzano reti 
    neurali locali che comunicano con le catene di approvvigionamento globali, prevedendo carenze di materie 
    prime con mesi di anticipo. Tuttavia, questa efficienza estrema porta con sé sfide geopolitiche nuove: 
    il controllo sui nodi di calcolo e sulle infrastrutture di raffreddamento dei data center è diventato 
    il nuovo terreno di scontro diplomatico, sostituendo in parte la dipendenza dai combustibili fossili.
    
    Sul fronte della vita quotidiana, l'integrazione ha raggiunto il livello del linguaggio naturale. Gli 
    assistenti virtuali non sono più semplici esecutori di comandi, ma mediatori proattivi. Un cittadino medio 
    nel 2026 interagisce con circa 150 istanze diverse di IA ogni ora, spesso senza accorgersene: dai sistemi 
    di regolazione del traffico urbano che minimizzano le emissioni dei veicoli autonomi, fino ai protocolli 
    medici personalizzati che monitorano i segnali biomarcatori tramite dispositivi indossabili invisibili. 
    
    La sicurezza informatica è diventata, di conseguenza, la priorità assoluta dei governi. La protezione 
    delle 'pipeline di dati' è ora considerata una questione di sicurezza nazionale equiparabile alla difesa 
    dei confini terrestri. L'integrazione sicura non è più un optional tecnico, ma un requisito legale 
    stringente. Le aziende che non rispettano i protocolli di crittografia post-quantistica rischiano 
    l'esclusione immediata dai mercati digitali internazionali. In questo contesto, l'etica dell'IA ha 
    smesso di essere una discussione filosofica per trasformarsi in un insieme di algoritmi di auditing 
    che verificano costantemente l'assenza di bias discriminatori nei processi decisionali automatizzati. 
    Infine, il mercato del lavoro sta vivendo la sua più grande trasformazione dalla prima rivoluzione industriale. 
    Mentre molti ruoli ripetitivi sono scomparsi, sono nate nuove professioni come l'Ingegnere della Fiducia 
    Algoritmica e l'Orchestratore di Ecosistemi Digitali, figure chiave per mantenere l'equilibrio tra 
    efficienza tecnologica e supervisione umana.
    """

    risultato = esegui_riassunto(testo_lungo)
    
    print("\nLANCIO ANSA (FORMATO MARKDOWN):")
    print("-" * 40)
    print(risultato)
    print("-" * 40)
import os
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# PUNTO DI INGRESSO DELLA CONFIGURAZIONE
# Carichiamo le variabili di ambiente definite nel file esterno '.env'.
# Questa pratica è fondamentale per la sicurezza e la flessibilità: 
# permette di modificare parametri come l'indirizzo del server o il modello 
# senza dover alterare il codice logico del programma.
load_dotenv()

# ACQUISIZIONE DEI PARAMETRI DI CONNESSIONE
# Utilizziamo la libreria 'os' per leggere i valori caricati da '.env'.
# - base_url: L'indirizzo di rete dove LM Studio espone le proprie API (es. localhost:1234).
# - model_name: Il nome identificativo del modello che abbiamo caricato e attivato in LM Studio.
# - api_key: Sebbene LM Studio spesso non richieda autenticazione, la libreria OpenAI 
#   necessita di una stringa (anche casuale) per inizializzare il client correttamente.
base_url = os.getenv("OPENAI_BASE_URL")
model_name = os.getenv("OPENAI_MODEL_NAME")
api_key = os.getenv("OPENAI_API_KEY")

# INIZIALIZZAZIONE DEL CLIENT DI COMUNICAZIONE
# Qui creiamo un'istanza della classe 'OpenAI'. Il parametro fondamentale è 'base_url',
# che istruisce il client a non contattare i server ufficiali di OpenAI, ma a reindirizzare 
# ogni richiesta verso il nostro server locale gestito da LM Studio.
client = OpenAI(
    base_url=base_url,
    api_key=api_key
)

# LOGICA DI RETRY ESPONENZIALE CON TENACITY
# Utilizziamo un decoratore per gestire automaticamente i tentativi di riconnessione.
# - retry_if_exception_type: Interveniamo solo se l'errore è un RateLimitError (429).
# - wait_exponential: Applichiamo un'attesa che cresce esponenzialmente (es. 1, 2, 4 secondi).
# - stop_after_attempt: Limitiamo il numero massimo di tentativi per evitare cicli infiniti.
@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(5),
    before_sleep=lambda retry_state: print(f"Server occupato. Nuovo tentativo tra pochi secondi... (Tentativo n. {retry_state.attempt_number})")
)
def chiamata_api_riassunto(testo):
    """
    Esegue la chiamata effettiva all'API. Questa funzione è isolata per permettere
    al decoratore 'tenacity' di monitorare specificamente questa operazione.
    """
    return client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system", 
                "content": "Sei un assistente specializzato nel riassumere testi complessi in punti chiave chiari e precisi."
            },
            {
                "role": "user", 
                "content": f"Analizza e riassumi il seguente testo:\n\n{testo}"
            }
        ],
        temperature=0.3,
        max_tokens=500
    )

def esegui_riassunto(testo):
    """
    Funzione principale che coordina la richiesta di riassunto.
    Gestisce il risultato della chiamata API e le eventuali eccezioni finali.
    """
    
    # Informiamo l'utente sull'avvio del processo.
    print(f"Elaborazione avviata con il modello '{model_name}'...")
    
    try:
        # Invochiamo la funzione protetta dal sistema di retry.
        risposta = chiamata_api_riassunto(testo)
        
        # NAVIGAZIONE NELL'OGGETTO DI RISPOSTA
        # Estraiamo il contenuto testuale dalla struttura restituita dal modello.
        return risposta.choices[0].message.content
        
    except RateLimitError:
        return "Errore: Il server è rimasto sovraccarico dopo tutti i tentativi di recupero."
    except Exception as e:
        # GESTIONE DELLE ECCEZIONI GENERICHE
        # Catturiamo errori critici per evitare il crash del programma.
        return f"Si è verificato un errore imprevisto: {e}"

# BLOCCO DI ESECUZIONE PRINCIPALE
if __name__ == "__main__":
    # Testo sorgente per il test della funzione.
    testo_per_analisi = (
        "L'intelligenza artificiale (IA) è il ramo dell'informatica che si occupa dello sviluppo di "
        "macchine in grado di eseguire compiti che normalmente richiedono l'intelligenza umana. "
        "Questi compiti includono l'apprendimento, il ragionamento, la risoluzione di problemi, la "
        "percezione e la comprensione del linguaggio naturale. Negli ultimi anni, grazie alla "
        "disponibilità di grandi quantità di dati e alla potenza di calcolo crescente, l'IA ha "
        "compiuto passi da gigante, diventando parte integrante di molte tecnologie quotidiane."
    )
    
    # Esecuzione del processo di riassunto.
    risultato_sintesi = esegui_riassunto(testo_per_analisi)
    
    # PRESENTAZIONE DEI RISULTATI A VIDEO
    # Utilizziamo una formattazione semplice e pulita per il terminale.
    print("\n" + "-"*50)
    print("ESITO DELL'ELABORAZIONE (RIASSUNTO):")
    print("-"*50)
    print(risultato_sintesi)
    print("-"*50)
from openai import OpenAI
import time

# Inizializzazione del client per la connessione al server locale LM Studio.
# Viene specificato l'indirizzo base del server e una chiave API fittizia richiesta dalla libreria.
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Definizione del messaggio di sistema (System Message).
# Questo dizionario stabilisce il comportamento e il tono che l'intelligenza artificiale deve mantenere.
# Viene inserito all'inizio di ogni richiesta per garantire coerenza nelle risposte.
system_message = {
    "role": "system", 
    "content": "Sei un esperto di ecologia e sostenibilità. Rispondi in modo educativo, ironico e incoraggiante."
}

# Inizializzazione della memoria della sessione.
# Questa lista conterrà i messaggi scambiati tra l'utente e l'assistente per mantenere il contesto.
chat_history = []

def get_chat_response(messages_to_send):
    """
    Invia la lista dei messaggi al modello locale e restituisce il testo della risposta.
    Gestisce eventuali errori di connessione al server.
    """
    try:
        # Creazione della richiesta di completamento della chat.
        completion = client.chat.completions.create(
            model="gpt-oss-20b",
            messages=messages_to_send,
            temperature=0.7, # Bilancia la precisione con una moderata creatività.
            max_tokens=150   # Limita la lunghezza della risposta per ottimizzare i tempi di elaborazione.
        )
        # Estrazione del contenuto testuale dalla risposta ricevuta.
        return completion.choices[0].message.content
    except Exception as e:
        # Messaggio di errore in caso di mancata risposta del server.
        return f"Errore di connessione a LM Studio: {e}"

# Ciclo principale di interazione con l'utente via terminale.
print("Benvenuto nell'Eco-Chat (Scrivi 'esci' per terminare)")
print("-" * 50)

while True:
    # Acquisizione dell'input testuale dall'utente.
    user_input = input("\nTu: ")
    
    # Condizione di uscita: se l'utente digita una parola chiave, il programma termina.
    if user_input.lower() in ["esci", "quit", "exit"]:
        print("Chiusura del programma in corso.")
        break

    # Registrazione del messaggio dell'utente nella cronologia della conversazione.
    chat_history.append({"role": "user", "content": user_input})

    # Implementazione della logica 'Sliding Window' (Finestra Scorrevole).
    # Per evitare di superare il limite di memoria (Context Window) del modello,
    # vengono presi in considerazione solo gli ultimi 5 messaggi della cronologia.
    last_messages = chat_history[-5:]
    
    # Costruzione del pacchetto di messaggi (payload) da inviare all'IA.
    # Si concatena il messaggio di sistema (che definisce l'identità) con la finestra degli ultimi messaggi.
    payload = [system_message] + last_messages
    
    # Richiesta della risposta al modello tramite la funzione definita in precedenza.
    assistant_res = get_chat_response(payload)
    
    # Visualizzazione della risposta a video.
    print(f"\nAI: {assistant_res}")
    
    # Salvataggio della risposta dell'assistente nella cronologia per mantenere il filo del discorso.
    chat_history.append({"role": "assistant", "content": assistant_res})
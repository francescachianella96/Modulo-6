from openai import OpenAI

# Inizializzazione del client per la comunicazione con il server locale.
# LM Studio emula l'interfaccia di OpenAI, quindi usiamo la stessa libreria.
# Il 'base_url' punta all'indirizzo locale predefinito di LM Studio (porta 1234).
# L'API key è una stringa fissa poiché il server locale non richiede autenticazione reale.
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Definizione della memoria della conversazione.
# Questa lista di dizionari rappresenta la "storia" che passiamo al modello.
# Ogni elemento ha un 'role' (ruolo) e un 'content' (contenuto del messaggio).
context = [
    {
        "role": "system", 
        "content": "Sei un esperto di ecologia e sostenibilità. Rispondi in modo educativo e incoraggiante."
    },
    {
        "role": "user", 
        "content": "Ciao, vorrei ridurre la mia impronta di carbonio a casa."
    },
    {
        "role": "assistant", 
        "content": "Ottima iniziativa! Puoi iniziare riducendo gli sprechi energetici e migliorando l'isolamento termico."
    }
]

# Aggiungiamo un nuovo messaggio dell'utente alla cronologia esistente.
# Il modello leggerà tutto il 'context' per capire di cosa stiamo parlando.
context.append({"role": "user", "content": "Qual è il primo passo più semplice?"})

# Funzione per generare una risposta deterministica dal modello.
# Riceve in ingresso la lista dei messaggi (cronologia).
def generate_stable_response(messages_history):
    # 'completions.create' invia la richiesta al server e attende la risposta.
    return client.chat.completions.create(
        model="gpt-oss-20b", # Identificativo del modello caricato in LM Studio.
        messages=messages_history, # Passiamo l'intera cronologia dei messaggi.
        temperature=0,  # Impostato a 0 per avere risposte identiche ogni volta (niente creatività).
        seed=1234,      # Parametro opzionale per aumentare ulteriormente la riproducibilità.
        max_tokens=100  # Limite massimo di parole/token nella risposta.
    )

# Eseguiamo la prima chiamata al modello.
print("Avvio della prima generazione...")
# Estraiamo il contenuto testuale della risposta dalla struttura dati restituita (choices[0].message.content).
res1 = generate_stable_response(context).choices[0].message.content
print(f"Risposta 1: {res1}\n")

# Eseguiamo la seconda chiamata per verificare che il risultato sia lo stesso.
print("Avvio della seconda generazione per verifica determinismo...")
res2 = generate_stable_response(context).choices[0].message.content
print(f"Risposta 2: {res2}")

# Confronto tra le due risposte per assicurarci che siano identiche parola per parola.
# Questo conferma che i parametri di stabilità (temperature=0) stanno funzionando.
assert res1 == res2, "Errore: Le risposte non sono identiche!"
print("\nSuccesso: Le risposte sono identiche parola per parola.")
def genera_prompt_zeroshot(commento):
    return f"""
    Classifica il tono del seguente commento come 'Sarcastico' o 'Non Sarcastico'.
    Rispondi solo con la categoria e una breve spiegazione.

    Commento: "{commento}"
    Risposta:"""

# Test con un caso ambiguo
test_commento = "Oh, fantastico! Il pacco è arrivato distrutto, proprio quello che desideravo."

print(genera_prompt_zeroshot(test_commento))

def genera_prompt_fewshot(commento_input):
    prompt_multilinea = f"""
Agisci come un esperto di analisi linguistica. Classifica il commento finale come 'Sarcastico' o 'Non Sarcastico'.

Esempio 1 (Iperbole):
Commento: "Adoro aspettare 40 minuti al telefono, è il mio hobby preferito!"
Categoria: Sarcastico

Esempio 2 (Genovino/Diretto):
Commento: "Il servizio è stato pessimo e l'operatore non è stato d'aiuto."
Categoria: Non Sarcastico

Esempio 3 (Ironia Sottile):
Commento: "Un altro aggiornamento software che rompe le funzioni base. Ottimo lavoro."
Categoria: Sarcastico

---
Ora classifica il seguente commento:
Commento: "{commento_input}"
Categoria:"""
    return prompt_multilinea

# Test
input_utente = "Grazie per avermi risposto dopo tre settimane, la vostra velocità è leggendaria."
print(genera_prompt_fewshot(input_utente))
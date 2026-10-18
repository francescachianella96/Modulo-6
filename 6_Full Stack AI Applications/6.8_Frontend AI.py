<!DOCTYPE html>
<html lang="it">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analizzatore di Logica e Codice</title>

    <!-- Importazione delle librerie esterne tramite CDN -->
    <!-- Axios: Gestisce le richieste HTTP asincrone verso il server LLM -->
    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
    <!-- Marked: Converte le stringhe in formato Markdown restituite dall'AI in HTML leggibile -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <!-- Highlight.js: Libreria per la colorazione della sintassi dei blocchi di codice (CSS + JS) -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>

    <style>
        /* Definizione delle variabili CSS per mantenere coerenza nel design system */
        :root {
            --colore-primario: #4f46e5;
            --sfondo-pagina: #0f172a;
            --sfondo-card: #1e293b;
            --testo-chiaro: #f8fafc;
            --bordo-input: #334155;
            --testo-codice: #38bdf8;
        }

        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background-color: var(--sfondo-pagina);
            color: var(--testo-chiaro);
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2rem;
            margin: 0;
        }

        /* Contenitore principale: centra il contenuto e definisce la larghezza massima del layout */
        .container {
            width: 100%;
            max-width: 900px;
            background: var(--sfondo-card);
            padding: 2.5rem;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
        }

        h1 {
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
            color: var(--testo-chiaro);
        }

        p {
            color: #94a3b8;
            margin-bottom: 1.5rem;
        }

        /* Area di inserimento del codice: utilizza un font monospace per facilitare la lettura */
        textarea {
            width: 100%;
            height: 180px;
            background: #0f172a;
            color: var(--testo-codice);
            border: 1px solid var(--bordo-input);
            border-radius: 8px;
            padding: 1rem;
            font-family: 'Fira Code', 'Cascadia Code', monospace;
            font-size: 0.9rem;
            resize: vertical;
            margin-bottom: 1.2rem;
            outline: none;
            transition: border-color 0.2s;
        }

        textarea:focus {
            border-color: var(--colore-primario);
        }

        /* Pulsante di invio: gestisce anche lo stato 'disabled' durante il caricamento */
        button {
            background: var(--colore-primario);
            color: white;
            border: none;
            padding: 0.8rem 2.5rem;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            font-size: 1rem;
            transition: background 0.2s, transform 0.1s;
        }

        button:hover {
            background: #4338ca;
        }

        button:active {
            transform: scale(0.98);
        }

        button:disabled {
            background: #6366f1;
            opacity: 0.6;
            cursor: wait;
        }

        /* Area dove verrà mostrata la spiegazione generata dal modello linguistico */
        #output {
            margin-top: 2rem;
            border-top: 1px solid var(--bordo-input);
            padding-top: 1.5rem;
            line-height: 1.7;
            font-size: 1.05rem;
        }

        /* Messaggio di attesa visualizzato tra l'invio e la ricezione dei dati */
        .loading-spinner {
            display: none;
            color: #94a3b8;
            margin-top: 1.2rem;
            font-size: 0.9rem;
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0% {
                opacity: 0.5;
            }

            50% {
                opacity: 1;
            }

            100% {
                opacity: 0.5;
            }
        }

        /* Formattazione specifica per i blocchi di codice all'interno della risposta */
        pre code {
            border-radius: 8px;
            padding: 1rem !important;
            font-size: 0.9rem;
        }
    </style>
</head>

<body>

    <div class="container">
        <h1>Analizzatore di Codice</h1>
        <p>Inserisci una porzione di codice o un algoritmo per ottenere una spiegazione dettagliata del suo
            funzionamento.</p>

        <!-- Campo di input per il codice dell'utente -->
        <textarea id="inputArea"
            placeholder="Esempio: def calcola_media(numeri): return sum(numeri) / len(numeri)"></textarea>

        <!-- Interfaccia di controllo -->
        <button id="runBtn">Analizza Codice</button>
        <div id="loader" class="loading-spinner">Elaborazione in corso, attendere...</div>

        <!-- Div di destinazione per la risposta formattata -->
        <div id="output"></div>
    </div>

    <script>
        /**
         * LOGICA DI INTERAZIONE CON IL MODELLO DI LINGUAGGIO (LLM)
         */

        // 1. Configurazione del client HTTP Axios
        // Questo oggetto centralizza le impostazioni per connettersi a LM Studio o altri sistemi compatibili con OpenAI API.
        const analizzatoreAPI = axios.create({
            baseURL: 'http://127.0.0.1:1234/v1', // Indirizzo locale del server LLM
            timeout: 90000 // Timeout esteso per gestire i tempi di generazione dei token più lunghi
        });

        // 2. Inizializzazione del parser Markdown (Marked)
        // Integriamo Highlight.js all'interno del processo di parsing per colorare automaticamente il codice presente nel testo.
        marked.setOptions({
            highlight: function (code, lang) {
                // Riconoscimento automatico del linguaggio per applicare la colorazione corretta
                return hljs.highlightAuto(code).value;
            }
        });

        // Riferimenti agli elementi del DOM (Document Object Model)
        const btnAnalizza = document.getElementById('runBtn');
        const campoTesto = document.getElementById('inputArea');
        const areaRisultato = document.getElementById('output');
        const indicatoreAttesa = document.getElementById('loader');

        /**
         * Gestore dell'evento click sul pulsante di analisi.
         * Coordina l'intero flusso: UI -> Richiesta API -> Parsing Risultato -> UI
         */
        btnAnalizza.addEventListener('click', async () => {
            const codiceInserito = campoTesto.value.trim();

            // Validazione base dell'input
            if (!codiceInserito) {
                alert("Per favore, inserisci del codice da analizzare.");
                return;
            }

            // Preparazione dell'interfaccia: disabilitiamo i controlli per evitare richieste multiple
            btnAnalizza.disabled = true;
            indicatoreAttesa.style.display = 'block';
            areaRisultato.innerHTML = ""; // Pulizia di eventuali analisi precedenti

            try {
                /**
                 * Invio della richiesta al server LLM.
                 * Utilizziamo l'endpoint standard per le conversazioni (Chat Completions).
                 */
                const risposta = await analizzatoreAPI.post('/chat/completions', {
                    "model": "gpt-oss-20b", // Identificativo del modello caricato nel server
                    messages: [
                        {
                            role: "system",
                            content: "Sei un programmatore esperto. Il tuo compito è analizzare il codice fornito dall'utente spiegandolo riga per riga e descrivendo come interagiscono i componenti. Usa Markdown per la formattazione e blocchi di codice dove necessario. Rispondi in modo professionale ed essenziale."
                        },
                        {
                            role: "user",
                            content: `Analizza il seguente codice:\n\n${codiceInserito}`
                        }
                    ],
                    temperature: 0.2 // Valore basso per garantire risposte più deterministiche e precise
                });

                // Estrazione del testo in Markdown dalla struttura della risposta API
                const testoMarkdown = risposta.data.choices[0].message.content;

                /**
                 * Trasformazione del Markdown in HTML.
                 * Il metodo 'parse' di Marked trasforma la stringa in elementi DOM validi.
                 */
                areaRisultato.innerHTML = marked.parse(testoMarkdown);

                /**
                 * Post-processing: Forziamo il ri-rendering degli stili di evidenziazione del codice
                 * per assicurarci che anche i blocchi iniettati dinamicamente siano colorati correttamente.
                 */
                document.querySelectorAll('pre code').forEach(hljs.highlightElement);

            } catch (errore) {
                // Gestione robusta degli errori di rete o di risposta del modello
                console.error("Errore durante l'analisi:", errore);
                areaRisultato.innerHTML = `
                    <div style="background: #450a0a; border: 1px solid #991b1b; padding: 1rem; border-radius: 8px; color: #fca5a5;">
                        <strong>Si è verificato un errore:</strong><br>
                        ${errore.message}<br><br>
                        <small>Assicurati che il server locale dei modelli sia attivo all'indirizzo http://localhost:1234.</small>
                    </div>`;
            } finally {
                // Ripristino dello stato dell'interfaccia indipendentemente dall'esito della richiesta
                btnAnalizza.disabled = false;
                indicatoreAttesa.style.display = 'none';
            }
        });
    </script>
</body>

</html>
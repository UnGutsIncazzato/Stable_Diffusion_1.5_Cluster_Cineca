# Generazione di Immagini con Stable Diffusion sul Cluster HPC CINECA Leonardo

Questo progetto contiene il codice, le dipendenze e la documentazione completa per eseguire la generazione di immagini da testo (Text-to-Image) utilizzando il modello **Stable Diffusion 1.5** sul supercomputer **Leonardo** del CINECA.

L'uso di un ambiente High Performance Computing (HPC) come Leonardo è cruciale per gestire la complessità e la mole di calcolo richieste dai modelli di diffusione (come Stable Diffusion), che eccedono le capacità di un normale hardware casalingo[cite: 3, 114, 115].

**La guida completa e dettagliata per tutti i passaggi è inclusa nel file:** `GuidaCinecaLeonardo.docx`[cite: 1, 4].

## Contenuto del Repository

| File/Cartella | Descrizione |
| :--- | :--- |
| `GuidaCinecaLeonardo.docx` | **DOCUMENTAZIONE COMPLETA:** Guida passo passo per la configurazione dell'ambiente, l'installazione delle librerie e l'esecuzione del codice sul cluster Leonardo[cite: 1, 2, 4]. |
| `generate.py` | Lo script Python principale contenente la logica per caricare il modello Stable Diffusion e generare un'immagine dal prompt[cite: 162, 164]. |
| `sd_generate.sh` | Lo script di lancio SLURM (`sbatch`) necessario per eseguire il codice sui nodi di calcolo di Leonardo, specificando risorse come GPU e tempo di esecuzione[cite: 152, 153, 233, 246]. |
| `generated_image.png` | Esempio di immagine risultato ottenuta dopo l'esecuzione del processo sul cluster. |
| **`requirements.txt`** | (Da aggiungere) Elenco delle dipendenze Python necessarie per l'ambiente virtuale. |

## Panoramica del Flusso di Lavoro (Consultare la Guida Dettagliata)

Per eseguire il progetto, sono necessari diversi step di configurazione e preparazione, tutti documentati nel file `GuidaCinecaLeonardo.docx`.

### 1. Preparazione dell'Accesso al CINECA

Questo step riguarda l'ottenimento delle credenziali e l'accesso remoto al cluster:
* Creazione dell'account utente sul portale UserDB del CINECA[cite: 5, 6].
* Completamento e convalida del profilo utente e richiesta di accesso HPC[cite: 12, 13, 14].
* Configurazione della procedura di autenticazione a due fattori (2FA) con codice OTP[cite: 18, 20].
* Installazione e configurazione del client `smallstep` per l'accesso SSH (istruzioni separate per Windows e Linux/macOS)[cite: 24, 25].

### 2. Setup dell'Ambiente sul Cluster

Una volta effettuato l'accesso SSH, l'ambiente di sviluppo deve essere preparato:
* Caricamento dei moduli di sviluppo necessari, ad esempio `cineca-ai`[cite: 88, 89].
* Definizione e attivazione di un ambiente virtuale Python (venv) per isolare le dipendenze[cite: 91, 92, 93].

### 3. Installazione e Trasferimento dei Modelli

Dato che i nodi di calcolo non accedono a Internet[cite: 116, 117], i modelli devono essere scaricati in locale e poi trasferiti:
* Installazione delle dipendenze Python necessarie (`diffusers`, `transformers`, `torch` con CUDA, ecc.)[cite: 122, 125, 127, 207, 209].
* Autenticazione con Hugging Face e download dei file di Stable Diffusion 1.5 (`v1-5-pruned.ckpt`) e del VAE (`vae-ft-mse-840000-ema-pruned.safetensors`)[cite: 138, 144, 147, 148, 149].
* Trasferimento dei modelli scaricati dal PC locale al cluster tramite comandi SCP[cite: 95, 99, 103, 119, 120].

### 4. Esecuzione del Codice

Il codice Python viene eseguito tramite il gestore di coda SLURM:
* Creazione del file Python (`generate.py`) con la logica di Stable Diffusion[cite: 162, 163].
* Creazione dello script SLURM (`sd_generate.sh`) per allocare risorse GPU e lanciare il file Python[cite: 152, 154].
* Invio del job alla coda di esecuzione tramite `sbatch`[cite: 246].

***

## Risultato Esempio

L'immagine `generated_image.png` mostra un risultato ottenuto da questo processo.

# Judith — LLM Fine-Tuning for Car Dealership Customer Service

Fine-tuning **LLaMA 3.2** to act as a bilingual AI sales assistant ("Judith") for a Spanish-speaking car dealership. The project covers the full ML pipeline: raw audio calls → speech-to-text → prompt engineering → fine-tuning → LLM-as-a-Judge evaluation.

---

## Pipeline Overview

```
WAV Audio Files
      │
      ▼
[1] Audio Transcription (AssemblyAI)
      │  Speaker-labeled transcripts
      ▼
[2] Answer Machine Filter
      │  Removes voicemail-only recordings
      ▼
[3] Prompt Engineering (GPT-4o-mini)
      │  Structures transcripts: Asesor / Cliente tags + <soc> tokens
      ▼
[4] Q&A Pair Extraction
      │  data.json  →  {question, answer} pairs
      ▼
[5] Tokenization + Train/Test Split
      │  Tokenized dataset (90/10 split)
      ▼
[6] Fine-Tuning (LLaMA 3.2)
      │  Fine-tuned model checkpoints
      ▼
[7] LLM-as-a-Judge Evaluation (GPT-4o-mini + cosine similarity)
```

---

## Project Structure

```
├── audio_files/                  Raw WAV call recordings
├── audio_files_processed/        Processed audio + transcripts
│   └── text_files1/
│       ├── transcripts/          GPT-structured transcripts
│       └── procesados/           Already-processed raw transcripts
├── answer_machine_files/         Voicemail recordings (filtered out)
│
├── audio_processing/
│   ├── audio_transcriber.py      AssemblyAI speech-to-text pipeline
│   ├── identify_answer_machine.py  Voicemail detection and filtering
│   └── plain_text.py             Text utility class (fuzzy word/sentence lookup)
│
└── src/
    ├── data/
    │   ├── clean_data.py         GPT prompt engineering — structures raw transcripts
    │   ├── preprocess.py         Extracts Q&A pairs from structured transcripts
    │   ├── tokenizer.py          LLaMA tokenizer wrapper + dataset split
    │   └── data.json             Final training dataset
    │
    ├── models/
    │   ├── trainer.py            Custom HuggingFace Trainer (FLOP tracking, smoothed loss)
    │   └── inference.py          Inference utility function
    │
    ├── train/
    │   ├── train.py              Main training script (3B model, adafactor)
    │   ├── train_30epoch_1bs_earlyS.py  1B model, cosine LR, early stopping
    │   ├── train_30epoch_1bs.py  1B model, 30 epochs
    │   ├── train_2epoch_1bs.py   1B model, 2 epochs
    │   └── train_5200.py         1B model, 5200 steps
    │
    ├── evaluation/
    │   ├── llm_eval.ipynb        LLM-as-a-Judge evaluation notebook
    │   ├── respuestas_llm.json   Raw model answers on the test set
    │   ├── respuestas_llm_depurado.json  Cleaned model answers
    │   └── judge_veredict_list_gpt4o_mini.json  Judge verdicts + similarity scores
    │
    ├── tools/
    │   └── helpers.py            OpenAI API wrappers + file I/O utilities
    │
    ├── eval_model.py             Quick single-question inference test
    └── get_metrics_finetuned_model.py  Compute eval loss on the fine-tuned model
```

---

## Stage 1 — Audio Transcription (`audio_processing/audio_transcriber.py`)

Uses the **AssemblyAI** API to transcribe `.wav` call recordings with **speaker diarization** (Spanish, `language_code='es'`). Only calls longer than 30 seconds are kept; shorter ones are skipped (voicemails, dropped calls). Output is a text file per call with `Speaker A: ...` / `Speaker B: ...` lines.

**Key config:**
- `speaker_labels=True` — diarizes speakers automatically
- `language_code='es'` — Spanish transcription model

---

## Stage 2 — Answer Machine Filtering (`audio_processing/identify_answer_machine.py`)

Scans every transcript for known voicemail phrases (in English and Spanish). Recordings that match are moved to `answer_machine_files/` and excluded from training data.

Detection phrases:
- `"Please leave your message"`
- `"Por favor deje su mensaje"`

Uses the `PlainText` class (`plain_text.py`), which supports fuzzy word matching via `difflib.SequenceMatcher` and exact normalized sentence lookup.

---

## Stage 3 — Prompt Engineering (`src/data/clean_data.py`)

Raw transcripts have unlabeled speakers (`Speaker A`, `Speaker B`). This stage sends each transcript to **GPT-4o-mini** with a structured prompt that instructs it to:

1. Identify and label each speaker as **Asesor** (sales advisor) or **Cliente** (customer)
2. Insert a `<soc>` token at the start of each speaker turn
3. Correct spelling and grammar
4. Enrich vehicle references with make and model where missing

Processed transcripts are saved to `transcripts/` and the originals are moved to `procesados/`.

> **Note:** Set `gptize_switch = True` in `clean_data.py` to enable processing. It's `False` by default to prevent re-processing.

---

## Stage 4 — Q&A Pair Extraction (`src/data/preprocess.py`)

Parses GPT-structured transcripts using the `<soc>` delimiter to extract conversation turns. Builds `{question, answer}` pairs by tracking speaker alternation:

- **Cliente** turns → `question`
- **Asesor** turns → `answer`

When the speaker switches from Asesor back to Cliente, the accumulated pair is saved. Output is `data.json` — the training dataset.

---

## Stage 5 — Tokenization (`src/data/tokenizer.py`)

`TokenizadorLlamadas` wraps the HuggingFace `AutoTokenizer` for LLaMA. For each Q&A pair:

1. Concatenates `question + answer`
2. Appends the EOS token manually (required for causal LM training)
3. Truncates from the **left** to `max_length=2048` tokens
4. Sets `labels = input_ids` (standard causal LM objective)

The dataset is split 90% train / 10% test with `seed=123`.

---

## Stage 6 — Fine-Tuning (`src/train/`)

### Base models
| Script | Base Model | Key Config |
|--------|-----------|------------|
| `train.py` | `meta-llama/Llama-3.2-3B-Instruct` | adafactor, 2596 steps, 5 epochs |
| `train_30epoch_1bs_earlyS.py` | `meta-llama/Llama-3.2-1B-Instruct` | AdamW, cosine LR, early stopping (patience=10), fp16 |
| `train_30epoch_1bs.py` | `meta-llama/Llama-3.2-1B-Instruct` | adafactor, 30 epochs, fp16 |

### Training setup
- **Batch size:** 1 per device with `gradient_accumulation_steps=4` (effective batch = 4)
- **Gradient checkpointing:** enabled (saves VRAM at cost of speed)
- **Best model selection:** saved by lowest `eval_loss`
- **Learning rate:** `1e-5` with cosine or constant schedule

### Custom `Trainer` (`src/models/trainer.py`)
Extends `transformers.Trainer` with:
- Per-step FLOP computation and throughput logging
- Smoothed loss history (EMA with window=100)
- Remaining time estimation
- Guard against empty batches (`numel() == 0`)

### Best performing model
`judith_llama3.2_1b_15epoch1_bSizeAdamW` — LLaMA 3.2-1B fine-tuned for 15 epochs with AdamW optimizer, batch size 1, on the Q&A dataset.

---

## Stage 7 — LLM-as-a-Judge Evaluation (`src/evaluation/llm_eval.ipynb`)

### Step 1 — Generate answers
Run the fine-tuned model on every sample in the test split and save `{question, real_answer, llm_answer}` to `respuestas_llm.json`.

### Step 2 — GPT-4o-mini Judge
Each generated answer is evaluated by **GPT-4o-mini** using a structured prompt that returns a parseable JSON verdict:

```
Relevancia: "NO_RELEVANTE" | "PARCIALMENTE_RELEVANTE" | "RELEVANTE"
Explicacion: "<brief explanation>"
```

### Step 3 — Cosine similarity
Uses `sentence-transformers/all-MiniLM-L6-v2` via LangChain's `pairwise_embedding_distance` evaluator to compute the cosine distance between the real and generated answer embeddings.

### Results (150 test samples)

| Metric | Value |
|--------|-------|
| Mean cosine similarity | 0.416 |
| Median cosine similarity | 0.444 |
| Responses with similarity > 0.5 | 52 / 150 (34.7%) |
| Dominant judge label | PARCIALMENTE_RELEVANTE |

---

## Setup

### Requirements

```bash
pip install openai assemblyai transformers datasets torch langchain langchain-huggingface sentence-transformers pandas seaborn matplotlib
```

### Environment Variables

The project reads all secrets from environment variables. Set the following before running any script:

| Variable | Required by | Where to get it |
|----------|-------------|-----------------|
| `OPENAI_API_KEY` | `src/tools/helpers.py` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `ASSEMBLYAI_API_KEY` | `audio_processing/audio_transcriber.py` | [assemblyai.com](https://www.assemblyai.com/) |

**Linux / macOS:**

```bash
export OPENAI_API_KEY="sk-..."
export ASSEMBLYAI_API_KEY="..."
```

**Windows (PowerShell):**

```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:ASSEMBLYAI_API_KEY = "..."
```

**Persistent (recommended) — add to your shell profile (`.bashrc`, `.zshrc`, or PowerShell `$PROFILE`):**

```bash
export OPENAI_API_KEY="sk-..."
export ASSEMBLYAI_API_KEY="..."
```

Never commit API keys to the repository.

### HuggingFace Access

LLaMA 3.2 requires accepting Meta's license on HuggingFace and authenticating:

```bash
huggingface-cli login
```

---

## Running the Pipeline

```bash
# 1. Transcribe audio files
python audio_processing/audio_transcriber.py

# 2. Filter voicemails
python audio_processing/identify_answer_machine.py

# 3. Structure transcripts with GPT (set gptize_switch=True first)
python src/data/clean_data.py

# 4. Extract Q&A pairs → data.json
python src/data/preprocess.py

# 5. Fine-tune (from src/ directory)
cd src
python train/train_30epoch_1bs_earlyS.py

# 6. Evaluate
python eval_model.py
python get_metrics_finetuned_model.py
# Or run the full LLM-as-a-Judge notebook:
jupyter notebook evaluation/llm_eval.ipynb
```

---

## External Services

| Service | Purpose | API |
|---------|---------|-----|
| AssemblyAI | Speech-to-text with speaker diarization | `assemblyai` Python SDK |
| OpenAI GPT-4o-mini | Transcript structuring (prompt engineering) | `openai` Python SDK |
| OpenAI GPT-4o-mini | LLM-as-a-Judge evaluation | `openai` Python SDK |
| HuggingFace | LLaMA 3.2 base model weights | `transformers` |
| sentence-transformers | Cosine similarity evaluation | `sentence-transformers` |

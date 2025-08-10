## Fashion Assistant Chatbot

An end‑to‑end RAG chatbot that helps users find fashion products from a product corpus. It uses:

- ChromaDB as a persistent vector store
- OpenAI embeddings (`text-embedding-ada-002`) for indexing and retrieval
- A Cross-Encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for better relevance
- OpenAI GPT‑4o for grounded generation
- Optional short‑term conversation memory variant

### Demo (CLI)
- Run `FashionChatbot.py` for single‑turn search with moderation
- Run `FashionChatbot_withMemory.py` for multi‑turn chats with context and a `new` command to reset

---

### Project structure

- `FashionChatbot.py`: CLI chatbot with retrieval, rerank, moderation, and GPT‑4o response
- `FashionChatbot_withMemory.py`: Same as above, but keeps brief conversation context between turns
- `embedding.py`: Builds the ChromaDB collection from a CSV dataset
- `search_and_generate.py`: Simple script to query, rerank, and generate a response; writes `top_5_RAG.csv`
- `ChromaDB_Data_v2/`: Persistent Chroma collection directory (can be reused across runs)
- `config.yaml`: Holds your `OPENAI_API_KEY` (do not commit your real key)
- `environment.yml`: Conda environment with pinned versions
- `top_5_RAG.csv`: Example output produced by `search_and_generate.py`
- `Fashion Assistant Chatbot Documentation.pdf`: Extended write‑up (optional)

---

### Requirements

- Python 3.9
- macOS or Linux
- OpenAI API key

Note on GPU: The project works on CPU. If you have a CUDA‑enabled PyTorch, reranking can be faster.

---

### Setup

Option A — venv + pip (recommended on macOS):

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install openai==1.58.1 chromadb==0.6.0 sentence-transformers==3.3.1 pandas==2.2.3 pyyaml==6.0.2 numpy==2.0.2 onnxruntime==1.19.2 scikit-learn==1.6.0 torch==2.5.1
```

Option B — conda (Linux friendly, may require tweaks on macOS):

```bash
conda env create -f environment.yml
conda activate llm
```

If `environment.yml` fails on macOS due to Linux‑specific pins, prefer Option A.

---

### Configuration

Set your OpenAI API key in `config.yaml`:

```yaml
OPENAI_API_KEY: "your_openai_api_key"
```

The scripts will read `config.yaml` and export `OPENAI_API_KEY` to the environment at runtime.

Do not commit real keys. For GitHub, keep `config.yaml` locally with a placeholder value or add it to `.gitignore`.

---

### Vector database (ChromaDB)

This repo includes a `ChromaDB_Data_v2/` directory intended for a prebuilt collection named `RAG_on_Fashion`.

You have two options:

- Use the existing collection in `ChromaDB_Data_v2/` (fastest to get started)
- Rebuild the collection from a CSV dataset using `embedding.py`

Rebuilding from CSV:

1) Obtain a product dataset with at least the following columns:
   - `p_id`, `name`, `products`, `brand`, `colour`, `price`, `avg_rating`, `ratingCount`, `p_attributes`, `description`, `img`

2) Update the `file_path` in `embedding.py` (bottom of the file) to point to your CSV, e.g.:

```python
file_path = "Fashion Dataset v2.csv"  # change to your CSV path
```

3) Build the index:

```bash
python embedding.py
```

This will:
- Concatenate relevant fields into a `Combined` text field
- Create metadata objects for each product
- Add documents, ids (`p_id`), and metadata to the collection `RAG_on_Fashion` in `ChromaDB_Data_v2/`

To reset the index, delete `ChromaDB_Data_v2/` and rerun `embedding.py`.

Data source example: see the Myntra dataset on Kaggle (referenced in the documentation PDF).

---

### Running the chatbots

Single‑turn chatbot:

```bash
python FashionChatbot.py
```

Multi‑turn chatbot with lightweight memory:

```bash
python FashionChatbot_withMemory.py
```

Controls:
- Type your query and press Enter
- Type `quit` to exit
- In the memory version, type `new` to start a fresh search context

Both versions:
- Perform retrieval from `RAG_on_Fashion`
- Rerank via `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Run an OpenAI moderation check on inputs
- Generate structured product suggestions with full metadata (name, brand, price, color, rating, rating_count, p_id, img, attributes)

On first run, the reranker and sentence‑transformers models will download automatically.

---

### Quick test without chat (RAG + generate)

```bash
python search_and_generate.py
```

This will run a sample query, produce top‑5 retrieved+reranked items, generate an answer, and write `top_5_RAG.csv`.

---

### How it works (high level)

1) Indexing (`embedding.py`)
   - Build `Combined` textual representation per product
   - Upsert to Chroma (`RAG_on_Fashion`) with metadata and `p_id` IDs

2) Retrieval + Reranking
   - Query Chroma with OpenAI embeddings
   - Rerank with `cross-encoder/ms-marco-MiniLM-L-6-v2`

3) Grounded Generation
   - Prompt GPT‑4o with the top results and strict output guidelines
   - Return structured suggestions including all metadata fields

4) Safety
   - Inputs are checked with OpenAI Moderation API

---

### Troubleshooting

- OpenAI auth error: Ensure `OPENAI_API_KEY` is correctly set in `config.yaml` and that your key is active.
- Empty/old results: Rebuild the index by running `python embedding.py` with the correct CSV path.
- Model downloads are slow: The reranker downloads on first run; ensure internet access and retry.
- Mac build issues with conda: Prefer the pip/venv setup; the provided `environment.yml` has Linux‑pinned packages.
- Torch/CUDA warnings: Safe to ignore on CPU‑only machines.

---

### GitHub hygiene

- Never commit your real `config.yaml` with secrets. Keep the placeholder value or add `config.yaml` to `.gitignore`.
- The `ChromaDB_Data_v2/` folder can be large. Consider excluding it from the repo and documenting rebuild steps.

Example `.gitignore` additions:

```gitignore
config.yaml
ChromaDB_Data_v2/
.venv/
__pycache__/
*.csv
```

---

### Acknowledgements

- OpenAI API (embeddings, moderation, GPT‑4o)
- ChromaDB vector database
- Sentence‑Transformers Cross‑Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- Dataset inspiration: Myntra fashion product dataset on Kaggle


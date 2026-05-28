# RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline using BM25S for retrieval and HuggingFace Transformers for generation.

## Features

- **Two RAG modes**:
  - **Simple RAG**: Single retrieval + generation
  - **Fix-sentence RAG**: Performs retrieval after each generated sentence for more context-aware generation

- **BM25S Retriever**: Fast BM25 retrieval with tokenized corpus caching
- **HuggingFace Integration**: Uses Transformers and Datasets libraries
- **Latency Tracking**: Comprehensive metrics for all pipeline stages
- **Interactive Mode**: Allows continuous querying with mode switching

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage (Interactive Mode)

```bash
# Default settings (BeIR/hotpotqa corpus, Llama-3.2-1B model, simple mode)
python rag_pipeline.py

# Custom corpus and model
python rag_pipeline.py --corpus BeIR/nfcorpus --model meta-llama/Llama-3.2-3B

# Fix-sentence mode
python rag_pipeline.py --mode fix-sentence
```

### Single Question Mode

```bash
python rag_pipeline.py --question "What is the capital of France?"
```

### Command Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--corpus` | `-c` | `BeIR/hotpotqa` | HuggingFace dataset name for corpus |
| `--model` | `-m` | `meta-llama/Llama-3.2-1B` | HuggingFace model name for generation |
| `--mode` | | `simple` | RAG mode: `simple` or `fix-sentence` |
| `--initial-k` | `-k` | `64` | Number of documents to retrieve initially |
| `--sentence-k` | | `3` | Documents to retrieve per sentence (fix-sentence mode) |
| `--max-tokens` | | `256` | Maximum new tokens to generate |
| `--cache-dir` | | `./cache` | Directory to cache tokenized corpus |
| `--question` | `-q` | | Single question (bypasses interactive mode) |

### Interactive Mode Commands

While in interactive mode:
- Type your question to get a response
- `metrics` - Show latency metrics
- `mode simple` or `mode fix-sentence` - Switch RAG mode
- `quit` or `exit` - Exit the program

## RAG Modes

### Simple RAG

1. User enters a question
2. BM25S retrieves top-k relevant documents (default k=64)
3. Documents are concatenated with the query as context
4. LLM generates the answer

### Fix-sentence RAG

1. User enters a question
2. BM25S retrieves top-k relevant documents (default k=64)
3. LLM starts generating token by token
4. After each sentence ends (`.`, `!`, `?`, `\n`):
   - The generated sentence is used as a new query
   - BM25S retrieves additional documents (default k=3)
   - New documents are appended to the context
   - Generation continues with enriched context

## Corpus Caching

The first run will:
1. Download the corpus from HuggingFace
2. Tokenize all documents using BM25S
3. Build the BM25 index
4. Save to `./cache/` (or specified cache directory)

Subsequent runs will load directly from the cache, significantly reducing startup time.

## Latency Metrics

The pipeline tracks and reports:
- Corpus loading time
- Tokenization time
- Index building time
- Cache loading/saving time
- Model loading time
- Per-query retrieval time
- Per-query generation time
- Token generation time (fix-sentence mode)
- Total query processing time

## Example Output

```
2026-02-01 10:00:00 - INFO - ============================================================
2026-02-01 10:00:00 - INFO - PROCESSING QUERY (mode: simple)
2026-02-01 10:00:00 - INFO - Question: What is machine learning?
2026-02-01 10:00:00 - INFO - ============================================================
2026-02-01 10:00:00 - INFO - Retrieving top-64 documents for query: 'What is machine learning?...'
2026-02-01 10:00:01 - INFO - Retrieval completed in 15.23 ms
2026-02-01 10:00:01 - INFO - Retrieved 64 documents
2026-02-01 10:00:01 - INFO - Generating response (simple RAG)...
2026-02-01 10:00:05 - INFO - Generation completed in 4523.45 ms
2026-02-01 10:00:05 - INFO - Generated 128 tokens
2026-02-01 10:00:05 - INFO - Total query time: 4538.68 ms
```

## Supported Corpora

Any HuggingFace dataset can be used, but the following BeIR datasets are well-tested:
- `BeIR/hotpotqa` (default)
- `BeIR/nfcorpus`
- `BeIR/fiqa`
- `BeIR/scidocs`
- `BeIR/msmarco`

## Notes

- GPU is used automatically if available (CUDA)
- Model uses FP16 on GPU for memory efficiency
- Large corpora may require significant memory for indexing
- First-time model download requires internet connection and HuggingFace login for gated models like Llama

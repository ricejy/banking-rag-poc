# Banking AI Search — PoC

A proof-of-concept AI-powered chatbot designed to make mobile banking frictionless. Acting as a **digital ambassador** embedded inside a banking app, the chatbot guides customers to the right product or service — and straight into it via a deep link — while answering general banking questions using an internal knowledge base grounded in real OCBC content.

---

## Motivation

Navigating a mobile banking app to find the right product, complete a transaction, or resolve an issue can be unintuitive. This PoC explores whether a conversational AI layer can remove that friction — understanding what the customer wants in natural language and either:

1. **Guiding them directly into the relevant section of the app** (e.g. open the Precious Metals trading screen), or
2. **Answering their question on the spot** using a RAG pipeline over official banking content.

The goal is a customer experience where the chatbot behaves like a knowledgeable, always-available digital concierge: understanding intent, surfacing relevant information, and acting as the bridge between the customer and the right part of the app.

---

## Demo

The PoC is served as a Streamlit web app that simulates the in-app chatbot experience.

![OCBC Logo](langgraph/resources/images/ocbc_logo.png)

---

## Architecture Overview

```
Customer query
      │
      ▼
┌─────────────┐
│   Router    │  ← Classifies query: in-app product/service vs. general FAQ
└──────┬──────┘
       │
  ┌────┴─────┐
  │          │
  ▼          ▼
┌──────┐  ┌──────┐
│In-App│  │ FAQ  │  ← Both nodes query ChromaDB (hybrid dense + BM25 search)
│ Node │  │ Node │
└──┬───┘  └──┬───┘
   │          │
   ▼          ▼
Response + (optional) Deep Link button
```

### LangGraph Pipeline

The chatbot is built as a [LangGraph](https://github.com/langchain-ai/langgraph) state graph with three nodes:

| Node | Role |
|------|------|
| **Router** | Classifies whether the query targets an in-app product/service or is a general FAQ. Uses structured LLM output (JSON schema enforced). |
| **In-App Node** | Identifies the specific product/service, retrieves relevant knowledge-base docs, generates a response, and emits a deep link button to open the product in the app. |
| **FAQ Node** | Answers general banking questions using only retrieved documents. Compliant with financial safety and compliance rules — no advice, no invented facts. |

### Retrieval (RAG)

Documents are stored in a local [ChromaDB](https://www.trychroma.com/) vector store. Retrieval uses a **hybrid search** strategy:

- **Dense vector search** — query embedded with `BAAI/bge-large-en-v1.5`, matched against indexed chunks via approximate nearest-neighbour search.
- **BM25 lexical scoring** — computed in-memory over the dense candidates to reward keyword overlap.
- **Score fusion** — `0.65 × dense_normalized + 0.35 × BM25_normalized`.

### LLM

All LLM calls go through `langchain_openai.ChatOpenAI`. A caching layer (`langgraph/cache/cache.json`, keyed by SHA-256 of the full prompt) avoids redundant API calls during development and demos.

---

## Repository Structure

```
.
├── langgraph/                        # Core chatbot application
│   ├── streamlit_app.py              # Streamlit UI — entry point
│   ├── chatbot_nodes.py              # LangGraph graph, nodes, retrieval logic
│   ├── llm_utils.py                  # LLM factory with caching and structured output
│   ├── llm_local.py                  # Experimental local inference via MLX (Mac only)
│   ├── cache/cache.json              # LLM response cache
│   ├── prompt_files/                 # All LLM prompts (loaded at runtime)
│   │   ├── node_router_system_prompt.txt
│   │   ├── node_router_user_prompt.txt
│   │   ├── identify_prod_service_system_prompt.txt
│   │   ├── identify_prod_service_user_prompt.txt
│   │   ├── prod_service_generation_system_prompt.txt
│   │   └── faq_generation_system_prompt.txt
│   └── resources/images/             # Static assets (OCBC logo)
│
├── scraping/                         # Data ingestion pipeline
│   ├── scrape.py                     # Scrapy spider — crawls OCBC website
│   ├── process_html.py               # Converts scraped HTML → Markdown
│   ├── process_pdf.py                # Converts scraped PDFs → Markdown
│   ├── chunk_and_index.py            # Chunks Markdown, embeds, upserts to ChromaDB
│   ├── config.py                     # Chunking parameters (MAX_WORDS, WORD_OVERLAP)
│   ├── scraped_html_files/           # Raw HTML from crawler
│   ├── scraped_pdf_files/            # Raw PDFs from crawler
│   └── processed_markdown_files/     # Cleaned Markdown ready for indexing
│
└── chromadb/
    ├── data/                         # Persistent ChromaDB vector store (runtime)
    └── test_query.py                 # Standalone script to test vector search
```

---

## Supported In-App Products & Services

The router classifies queries against the following list. When matched, the chatbot responds with product information **and** a deep link button that opens the relevant screen inside the app:

| Product / Service | Deep Link Destination |
|---|---|
| Precious Metals | Buy/sell precious metals screen |
| Unit Trust | Unit trust products screen |
| Insurance | Bancassurance screen |
| Dispute Card Transactions | Card dispute screen |

---

## Setup

### Prerequisites

- Python 3.10+
- An OpenAI API key

### Install dependencies

```bash
pip install streamlit langgraph langchain-openai chromadb sentence-transformers \
            streamlit-float scrapy markdownify beautifulsoup4 pdfplumber pypdf \
            python-dotenv
```

### Environment variables

Create a `.env` file at the repo root:

```
API_KEY=<your_openai_api_key>
```

---

## Running the App

```bash
cd langgraph
streamlit run streamlit_app.py
```

---

## Data Pipeline

Run these steps in order to rebuild the knowledge base from scratch.

### 1. Crawl the OCBC website

```bash
cd scraping
python scrape.py
```

Outputs HTML files to `scraping/scraped_html_files/` and PDFs to `scraping/scraped_pdf_files/`. Crawl depth is capped at 3, respects `robots.txt`, and uses polite throttling.

### 2. Convert HTML to Markdown

```bash
python process_html.py
```

Filters scraped HTML files by the URL prefixes configured in `process_html.py` (`URL_PREFIXES` list) and converts them to Markdown. Output goes to `scraping/processed_markdown_files_1/`.

### 3. Convert PDFs to Markdown

```bash
python process_pdf.py
```

Converts all PDFs in `scraping/scraped_pdf_files/` to Markdown. Output goes to `scraping/processed_markdown_files/`.

### 4. Chunk and index into ChromaDB

```bash
python chunk_and_index.py
```

Reads all Markdown files in `scraping/processed_markdown_files/`, applies a two-pass chunking strategy (section-level split by headings → sliding word window, 400 words / 60-word overlap), embeds each chunk with `BAAI/bge-large-en-v1.5`, and upserts into the `poc_collection` ChromaDB collection at `chromadb/data/`.

> **Note:** If you change the embedding model, delete the existing ChromaDB collection first to avoid dimension conflicts (see commented line in `chunk_and_index.py`).

### 5. Test retrieval

```bash
cd ../chromadb
python test_query.py
```

Runs a sample query and prints the top-10 ranked chunks with distances and metadata.

---

## Key Design Decisions

**Hybrid retrieval over pure dense search** — BM25 catches exact keyword matches (e.g. product names, fee amounts) that dense embeddings can miss, while dense search handles semantic paraphrasing. The 65/35 blend was chosen as a starting point for tuning.

**Structured LLM output for routing and product identification** — JSON schema is enforced at the OpenAI API level, so the graph never receives malformed routing decisions.

**Prompt files over inline strings** — keeping prompts in `.txt` files makes them easy to iterate on without touching Python code.

**Response caching** — during development and PoC demos, identical queries are served from a local JSON cache to avoid API latency and cost.

**Deep links as the primary CTA** — rather than describing multi-step navigation, the chatbot emits a button that takes the customer directly to the relevant screen, minimising friction.

---

## Limitations (PoC Scope)

- Jailbreak detection is a simple keyword blocklist (`"die"`, `"suicide"`, `"unfiltered"`); the LLM-based guard is commented out.
- Available products/services and their deep links are hardcoded in `chatbot_nodes.py`.
- Conversation history is held in memory (`MemorySaver`) and lost on page refresh.
- The local MLX inference path (`llm_local.py`) is experimental and not wired into the main pipeline. (Was experimented with previously)

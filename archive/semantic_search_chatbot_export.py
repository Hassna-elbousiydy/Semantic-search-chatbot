#!/usr/bin/env python
# coding: utf-8

# # Semantic Search Chatbot Notebook
# This notebook contains the step-by-step implementation of the NLP Q&A system.

# # Data Download & Preprocessing (SQuAD v1.1)
# 
# This section downloads the SQuAD v1.1 dataset, extracts contexts as documents, cleans them, chunks them into token-aware segments, and saves:
# - Full processed chunks to `data/processed/` (ignored by git),
# - Small samples to `data/sample/` (committed) for demo & CI.

# In[1]:


import os
import json
import math
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parents[1]
# If paths look wrong in VS Code, fallback to repo root manually:
REPO = Path.cwd().parents[1] if Path.cwd().name == "notebooks" else Path.cwd()
DATA_RAW = REPO / "data" / "raw"
DATA_PROCESSED = REPO / "data" / "processed"
DATA_SAMPLE = REPO / "data" / "sample"

for p in [DATA_RAW, DATA_PROCESSED, DATA_SAMPLE]:
    p.mkdir(parents=True, exist_ok=True)

print("Repo:", REPO)
print("Raw:", DATA_RAW)
print("Processed:", DATA_PROCESSED)
print("Sample:", DATA_SAMPLE)


# In[2]:


# Download SQuAD v1.1 (train/dev)
dataset = load_dataset("squad")
train = dataset["train"]
dev = dataset["validation"]

print(train)
print(dev)
print("Example item keys:", train[0].keys())


# In[3]:


def clean_text(txt: str) -> str:
    # lightweight cleaning; keep it minimal to avoid losing meaning
    txt = txt.replace("\xa0", " ").replace("\t", " ").replace("\r", " ")
    txt = " ".join(txt.split())  # normalize whitespace
    return txt.strip()

# Extract unique contexts from train + dev
all_contexts = [clean_text(x["context"]) for x in train] + [clean_text(x["context"]) for x in dev]
unique_contexts = list(dict.fromkeys(all_contexts))  # preserve order, remove duplicates
len_unique = len(unique_contexts)
len_all = len(all_contexts)

print(f"All contexts: {len_all:,} | Unique contexts: {len_unique:,}")
print("Sample context:", unique_contexts[0][:300], "...")


# In[4]:


# Use the same tokenizer family you’ll use for embeddings later
# MiniLM works well on CPU and is small.
TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

MAX_TOKENS = 240
STRIDE = 60  # overlap

def chunk_by_tokens(text: str, max_tokens: int = MAX_TOKENS, stride: int = STRIDE):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        sub = tokens[start:end]
        chunk_text = tokenizer.decode(sub, skip_special_tokens=True)
        chunks.append(chunk_text)
        if end == len(tokens):
            break
        start = max(0, end - stride)
    return chunks

# quick sanity test
test_chunks = chunk_by_tokens(unique_contexts[0])
len(test_chunks), sum(len(tokenizer.encode(c, add_special_tokens=False)) for c in test_chunks)


# In[5]:


def build_chunk_records(docs: list[str]):
    records = []
    for doc_id, doc in enumerate(docs):
        chunks = chunk_by_tokens(doc)
        for i, ch in enumerate(chunks):
            rec = {
                "doc_uuid": f"doc-{doc_id:06d}",
                "chunk_id": i,
                "text": ch,
                "n_tokens": len(tokenizer.encode(ch, add_special_tokens=False)),
                "source": "squad_v1.1"
            }
            records.append(rec)
    return records

chunk_records = build_chunk_records(unique_contexts)
len(chunk_records), chunk_records[0]


# In[7]:


# FULL processed dump (ignored by git)
full_jsonl = DATA_PROCESSED / "squad_chunks.jsonl"
with full_jsonl.open("w", encoding="utf-8") as f:
    for rec in chunk_records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# SMALL sample for repo/demo (commit this!)
sample_n = 200  # small but useful
sample_jsonl = DATA_SAMPLE / "squad_chunks_sample.jsonl"
with sample_jsonl.open("w", encoding="utf-8") as f:
    for rec in chunk_records[:sample_n]:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("Saved:", full_jsonl)
print("Saved sample:", sample_jsonl, "(", sample_n, "records )")


# In[8]:


def pick_questions(ds, limit=100):
    rows = []
    for i in range(min(limit, len(ds))):
        rows.append({
            "id": ds[i]["id"],
            "title": ds[i]["title"],
            "question": clean_text(ds[i]["question"])
        })
    return rows

questions_sample = pick_questions(dev, limit=100)

q_jsonl = DATA_SAMPLE / "squad_questions_sample.jsonl"
with q_jsonl.open("w", encoding="utf-8") as f:
    for q in questions_sample:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")

print("Saved:", q_jsonl, "(", len(questions_sample), "records )")


# In[10]:


import matplotlib.pyplot as plt

lens = [rec["n_tokens"] for rec in chunk_records[:5000]]  # sample for speed
plt.hist(lens, bins=30)
plt.title("Chunk token length distribution (sample)")
plt.xlabel("tokens"); plt.ylabel("count")
plt.show()

print("Avg tokens:", np.mean(lens), "| 95th pct:", np.percentile(lens, 95))


# ## Data Card (SQuAD v1.1)
# 
# - **Source:** SQuAD v1.1 (train + dev)
# - **Docs:** Context paragraphs treated as documents (deduplicated)
# - **Preprocessing:** whitespace normalization; token-aware chunking (max 240 tokens, stride 60)
# - **Artifacts:**
#   - `data/processed/squad_chunks.jsonl` — full processed chunks (git-ignored)
#   - `data/sample/squad_chunks_sample.jsonl` — 200-chunk sample (committed)
#   - `data/sample/squad_questions_sample.jsonl` — 100-question sample (committed)
# - **Use:** The chunks feed the embedding + vector store stage; sample files enable quick demo & CI.

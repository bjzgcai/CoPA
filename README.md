# CoPA Software Pipeline

This repository contains the core pipeline used to run retrieval-based personalization, LLM response generation, and factor-aware evaluation on the CoPA-style StackExchange datasets.

The project is organized as a practical end-to-end workflow:

1. Prepare a domain-specific dataset in Parquet format.
2. Optionally reorder each user's historical profile with semantic retrieval.
3. Generate responses with one of the supported personalization strategies.
4. Evaluate the generated responses against the factor labels in the dataset.

The current repository documents the runnable scripts that are present here, while the pipeline description reflects the full intended workflow.

## Repository Structure

```text
Software/
├── agent/
│   ├── generation/
│   │   ├── baseline.py
│   │   ├── profileGenerate.py
│   │   └── profileQA.py
│   ├── prompt/
│   │   ├── system/
│   │   └── user/
│   └── utils/
├── dataset/
│   ├── engineering_tools.parquet
│   ├── leisure_fandom.parquet
│   ├── lifestyle_society.parquet
│   └── science_theory.parquet
├── evaluation/
│   ├── evaluator.py
│   ├── llm_client.py
│   └── utils.py
├── retrieval/
│   └── rag_rank.py
└── requirements.txt
```

## What the Data Looks Like

Each Parquet file in `dataset/` currently contains 10 top-level columns:

- `user_id`
- `question`
- `question_text`
- `profile`
- `answers`
- `domain`
- `category`
- `creation_date`
- `factor_labels`
- `random_factor_labels`

In practice:

- `question` and `question_text` define the current query to answer.
- `profile` stores the user's historical question traces.
- `factor_labels` stores the structured factor annotations used in evaluation.
- `random_factor_labels` is preserved as an auxiliary label field.

## Setup

Run all commands below from the repository root:

```bash
cd /home/tosdata/Software
mkdir -p outputs
```


### 1. Create or activate a Python environment

A minimal dependency list is provided in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 2. Configure API access

The generation and evaluation scripts call an OpenAI-compatible chat completion endpoint.

You can configure the environment with either shell variables or a `.env` file.

Required variables:

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4o-mini"
export OPENAI_MAX_TOKENS="1000"
export OPENAI_TEMPERATURE="0.7"
```

Notes:

- `OPENAI_BASE_URL` can point to any OpenAI-compatible endpoint.
- Both `agent/utils/llm_client.py` and `evaluation/llm_client.py` load environment variables via `python-dotenv`.
- Prompt templates must remain in `agent/prompt/system/` and `agent/prompt/user/`.

## Pipeline Overview

### Stage 1: Optional Retrieval-Based Reordering

Script: `retrieval/rag_rank.py`

Purpose:

- Re-rank each row's `profile` field by semantic similarity to the current question.
- Produce a reordered Parquet file for retrieval-based personalization.

This stage is useful when you want the model to consume the most relevant historical questions first instead of using the original order.

Example:

```bash
python retrieval/rag_rank.py \
  --input_file dataset/engineering_tools.parquet \
  --output_file outputs/engineering_tools_ranked.parquet \
  --model_name sentence-transformers/all-MiniLM-L6-v2 \
  --profile_col profile \
  --batch_size 32 \
  --device cuda:0
```

Important arguments:

- `--input_file`: input Parquet dataset.
- `--output_file`: destination Parquet with reordered `profile`.
- `--model_name`: SentenceTransformer checkpoint.
- `--profile_col`: profile column to sort, usually `profile`.
- `--save_original`: optionally keeps a backup copy of the original profile column.
- `--device`: target device such as `cpu`, `cuda:0`, or `mps`.

Output:

- A Parquet file with the same rows as the input.
- The `profile` column is reordered by retrieval similarity.

### Stage 2: Response Generation

There are two supported generation workflows.

#### Option A: Baseline / Prompted Personalization

Script: `agent/generation/baseline.py`

This script supports three generation modes:

- `No-Personalization`
- `Rag-Personalization`
- `Time-Personalization`

Behavior:

- Reads one Parquet file.
- Builds prompts from the current question and user history.
- Calls the configured LLM endpoint.
- Writes outputs incrementally to a JSON file keyed by `user_id`.

Example: no personalization

```bash
python agent/generation/baseline.py \
  -i dataset/engineering_tools.parquet \
  -o outputs/engineering_tools_no_personalization.json \
  -t No-Personalization
```

Example: retrieval-based personalization

```bash
python agent/generation/baseline.py \
  -i outputs/engineering_tools_ranked.parquet \
  -o outputs/engineering_tools_rag_personalization.json \
  -t Rag-Personalization
```

Example: time-based personalization with factor-aware prompting

```bash
python agent/generation/baseline.py \
  -i dataset/engineering_tools.parquet \
  -o outputs/engineering_tools_time_factor.json \
  -t Time-Personalization \
  -f
```

How the modes differ:

- `No-Personalization`: uses only the current question.
- `Rag-Personalization`: uses the reordered history from retrieval.
- `Time-Personalization`: sorts the history by recency.
- `-f / --factor`: appends the six-factor personalization instruction to the system prompt.

#### Option B: ProfileQA-Style Generation

Script: `agent/generation/profileQA.py`

This pipeline is more expensive but more structured.

Behavior:

- Iterates through a user's recent history.
- Infers a domain for each historical question.
- Builds domain-level and global user profiles with the LLM.
- Generates the final answer conditioned on the synthesized profile.

Example:

```bash
python agent/generation/profileQA.py \
  -i dataset/engineering_tools.parquet \
  -o outputs/engineering_tools_profileqa.json
```

Factor-aware version:

```bash
python agent/generation/profileQA.py \
  -i dataset/engineering_tools.parquet \
  -o outputs/engineering_tools_profileqa_factor.json \
  -f
```

When to use it:

- Use `baseline.py` for fast experiments and ablations.
- Use `profileQA.py` when you want a richer profile synthesis pipeline.

### Stage 3: Factor-Aware Evaluation

Script: `evaluation/evaluator.py`

Purpose:

- Compare generated answers with the factor annotations stored in `factor_labels`.
- Ask the evaluator model to score six dimensions:
  - Cognitive Trust
  - Situational Anchoring
  - Schema Consistency
  - Cognitive Load Management
  - Metacognitive Scaffolding
  - Affective and Motivational Resonance

Example:

```bash
python evaluation/evaluator.py \
  -i dataset/engineering_tools.parquet \
  -o outputs/engineering_tools_rag_personalization.json \
  -r outputs/engineering_tools_rag_personalization_eval.json
```

Inputs:

- `-i / --input-file`: source Parquet with `factor_labels`.
- `-o / --output-file`: generated response JSON.
- `-r / --result-file`: evaluation result JSON.

Output format:

```json
{
  "score": 0.0,
  "per_question_scores": [
    {
      "id": "user_id",
      "score": 0.0,
      "details": {
        "Cognitive Trust": {
          "score": 0,
          "reasoning": "..."
        }
      }
    }
  ]
}
```

## Recommended End-to-End Runs

### A. Fast baseline pipeline

```bash
python agent/generation/baseline.py \
  -i dataset/engineering_tools.parquet \
  -o outputs/engineering_tools_baseline.json \
  -t No-Personalization

python evaluation/evaluator.py \
  -i dataset/engineering_tools.parquet \
  -o outputs/engineering_tools_baseline.json \
  -r outputs/engineering_tools_baseline_eval.json
```

### B. Retrieval-based personalization pipeline

```bash
python retrieval/rag_rank.py \
  --input_file dataset/engineering_tools.parquet \
  --output_file outputs/engineering_tools_ranked.parquet \
  --model_name sentence-transformers/all-MiniLM-L6-v2 \
  --profile_col profile \
  --batch_size 32 \
  --device cuda:0

python agent/generation/baseline.py \
  -i outputs/engineering_tools_ranked.parquet \
  -o outputs/engineering_tools_rag.json \
  -t Rag-Personalization

python evaluation/evaluator.py \
  -i outputs/engineering_tools_ranked.parquet \
  -o outputs/engineering_tools_rag.json \
  -r outputs/engineering_tools_rag_eval.json
```

### C. Profile synthesis pipeline

```bash
python agent/generation/profileQA.py \
  -i dataset/engineering_tools.parquet \
  -o outputs/engineering_tools_profileqa.json \
  -f

python evaluation/evaluator.py \
  -i dataset/engineering_tools.parquet \
  -o outputs/engineering_tools_profileqa.json \
  -r outputs/engineering_tools_profileqa_eval.json
```

## Output Conventions

### Generation output

The generation scripts save results incrementally as a JSON dictionary keyed by `user_id`.

Typical shape:

```json
{
  "12345@stackexchange": [
    {
      "output": "generated answer text"
    }
  ]
}
```

### Evaluation output

The evaluator stores:

- a global average score
- one record per question/user
- per-factor reasoning and scores

## Practical Notes

- The scripts are resumable: both generation and evaluation load existing JSON files and skip processed `user_id`s.
- `profileQA.py` is significantly slower than `baseline.py` because it performs multiple LLM calls per user.
- `rag_rank.py` can use GPU if available; otherwise set `--device cpu`.
- `pandas.read_parquet` requires a Parquet backend, which is why `pyarrow` is included in `requirements.txt`.
- If you change prompt templates, keep the filenames expected by `agent/prompt/prompt_formetters.py`.

## Current Scope

This repository currently contains:

- cleaned domain-specific Parquet datasets
- retrieval ranking for profile history
- baseline and profile-synthesis response generation
- factor-aware evaluation

It is best used as an experimental pipeline for comparing personalization strategies on StackExchange-style question answering data.

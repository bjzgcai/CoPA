# CoPA

CoPA is a benchmark for personalized question answering (QA). This dataset is introduced in our paper, [CoPA: Benchmarking Personalized Question Answering with Data-Informed Cognitive Factors](https://arxiv.org/abs/2604.14773).

## Overview

Personalized question answering is not only about giving a correct answer. A useful response should also match how a specific user prefers to receive information: what they trust, how much detail they need, how much structure helps them, and what kind of contextual grounding they expect.

CoPA is designed for this setting. The dataset is constructed from the Stack Exchange question answering platform, using community QA data to pair each target user question with that user's historical profile, candidate community answers, and factor annotations that capture user-specific cognitive preferences. This makes CoPA suitable for evaluating whether a model can generate responses that are not just generally good, but personally appropriate for a given user.

This dataset comes from our arXiv paper:

- Paper: https://arxiv.org/abs/2604.14773


## Dataset Structure

The dataset contains 4 Parquet files, each corresponding to a broad category:

- `engineering_tools.parquet` - Engineering and Tools - 864 examples
- `leisure_fandom.parquet` - Leisure and Fandom - 252 examples
- `lifestyle_society.parquet` - Lifestyle and Society - 413 examples
- `science_theory.parquet` - Science and Theory - 456 examples

In total, CoPA contains:

- **1,985 examples / user profiles**
- **158 domains**
- **4 broad categories**

Each row represents one personalized QA example built around a target user and a target question.

### Main Fields

Below are the main fields in each example and what they mean:

| Field | Description |
| --- | --- |
| `user_id` | A unique identifier for the user in the source community QA domain. |
| `question` | A short version of the target question. |
| `question_text` | The full natural-language text of the target question to be answered. |
| `profile` | The user's historical question-answering profile. This is a list of previous questions associated with the same user, together with contextual metadata and community answers. It is used to infer the user's preferences and behavior patterns. |
| `answers` | Candidate answers for the target question collected from the source community QA data. |
| `domain` | The source domain or community from which the example is drawn, such as a specific Stack Exchange site. |
| `category` | The broad category grouping the example, such as Engineering and Tools or Science and Theory. |
| `creation_date` | The timestamp associated with the target question. |
| `factor_labels` | Structured annotations for the six cognitive factors used in CoPA. These labels describe how an ideal personalized response should align with the user's preferences. |
| `random_factor_labels` | An auxiliary factor-style annotation field included as an additional reference field. |

### About the `profile` Field

The `profile` field is one of the core parts of CoPA. It contains the user's historical interaction record and typically includes:

- previous questions asked by the user
- question text and identifiers
- domain and category metadata
- timestamps
- associated community answers

This historical profile provides the context needed to model user-specific preferences rather than treating each question in isolation.

### About the `answers` Field

The `answers` field contains candidate answers for the target question. These are useful for:

- benchmarking answer selection or reranking
- studying personalization in answer generation
- comparing model outputs against community-provided responses

### About the `factor_labels` Field

The `factor_labels` field is the key signal that makes CoPA different from standard QA datasets. It captures six cognitive factors used for factor-level personalized evaluation:

- **Cognitive Trust**
- **Situational Anchoring**
- **Schema Consistency**
- **Cognitive Load Management**
- **Metacognitive Scaffolding**
- **Affective and Motivational Resonance**

For each factor, the dataset provides structured descriptions that help characterize what kind of response would better align with the user's cognitive preferences.

### Example Schema (Simplified)

```python
{
  "user_id": "10422@3dprinting",
  "question": "What should I use to clean buildtak(knock off)?",
  "question_text": "...",
  "profile": [
    {
      "question_id": "...",
      "question": "...",
      "question_text": "...",
      "domain": "...",
      "category": "...",
      "creation_date": "...",
      "answers": [...]
    }
  ],
  "answers": [
    {
      "Score": 4,
      "body": "...",
      "id": 11746,
      "isaccept": false,
      "timestep": "2020-01-12T01:28:50.690"
    }
  ],
  "domain": "3dprinting",
  "category": "Engineering and Tools",
  "creation_date": "...",
  "factor_labels": [
    {
      "Cognitive Trust": {
        "description": "...",
        "explanation": "..."
      },
      "Situational Anchoring": {
        "description": "...",
        "explanation": "..."
      },
      "Schema Consistency": {
        "description": "...",
        "explanation": "..."
      },
      "Cognitive Load Management": {
        "description": "...",
        "explanation": "..."
      },
      "Metacognitive Scaffolding": {
        "description": "...",
        "explanation": "..."
      },
      "Affective and Motivational Resonance": {
        "description": "...",
        "explanation": "..."
      }
    }
  ]
}
```

## Intended Uses

CoPA can be used for:

- benchmarking personalized question answering systems
- evaluating factor-aware response generation
- studying user modeling and preference-aware alignment
- analyzing how LLM outputs match user-specific cognitive preferences
- comparing personalized QA methods beyond generic lexical metrics

## Citation and Links

If you use CoPA in your research, please cite our paper.

- arXiv paper: https://arxiv.org/abs/2604.14773

### BibTeX

```bibtex
@misc{su2026copabenchmarkingpersonalizedquestion,
  title={CoPA: Benchmarking Personalized Question Answering with Data-Informed Cognitive Factors},
  author={Hang Su and Zequn Liu and Chen Hu and Xuesong Lu and Yingce Xia and Zhen Liu},
  year={2026},
  eprint={2604.14773},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2604.14773}
}
```

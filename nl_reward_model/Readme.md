# Natural Language Reward Model (NL_Reward_Model)

This repository contains implementations of reward models trained on different datasets to provide natural language feedback and critiques for language model outputs.

## Overview

The NL_Reward_Model project trains specialized models that can:
- Evaluate the quality of language model responses
- Provide detailed textual feedback
- Identify specific spans of text that are particularly good or problematic
- Generate structured critique outputs in JSON format

## Requirements


## Datasets

The repository includes training scripts and configurations for three different datasets:

### 1. UltraFeedback

A comprehensive dataset for training reward models that provide detailed critiques of language model outputs. The model identifies good and poor spans within responses and provides textual feedback.

**Key Features:**
- Span-level feedback (good_spans and poor_spans)
- Detailed textual critiques
- JSON-structured output format

**Training:**
```bash
cd RLAIF/TEXT2GRAD/NL_Reward_Model/UltraFeedback
bash deepspeed_train.sh
```


### 2. SLF5K

A dataset focused on evaluating and critiquing summaries against original posts using word-level scoring based on span-based quality assessment.


**Key Features:**
- Word-level scoring (1 for good spans, -1 for poor spans, 0 for neutral)
- Textual feedback on summary quality
- JSON-structured output format

**Training:**
```bash
cd RLAIF/TEXT2GRAD/NL_Reward_Model/SLF5K
bash deepspeed_train.sh
```

### 3. KodCode

A dataset for evaluating and providing feedback on programming solutions to coding problems.

**Key Features:**
- Code-specific feedback on correctness, efficiency, readability, and completeness
- Identification of problematic code snippets (wrong_code)
- Suggestions for code improvements (improvement_code)
- JSON-structured output format

**Training:**
```bash
cd RLAIF/TEXT2GRAD/NL_Reward_Model/KodCode
bash deepspeed_train.sh
```

## Model Architecture
All implementations use:
- Base LLM: Llama-3.1-8B-Instruct
- LoRA fine-tuning for parameter efficiency
- DeepSpeed for distributed training
- Structured JSON output format






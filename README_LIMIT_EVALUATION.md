# TriStage-RAG LIMIT Dataset Evaluation Guide

This guide explains how to evaluate the 3-stage TriStage-RAG pipeline on the LIMIT dataset using MTEB.

## Overview

The TriStage-RAG pipeline combines three models:
1. **Stage 1**: `google/embeddinggemma-300m` - Fast candidate generation
2. **Stage 2**: `lightonai/GTE-ModernColBERT-v1` - Multi-vector rescoring
3. **Stage 3**: `cross-encoder/ms-marco-MiniLM-L6-v2` - Cross-encoder reranking

## Setup

### 1. Install Dependencies

```bash
pip install mteb datasets numpy
```

### 2. Download LIMIT Dataset

```bash
# Clone the LIMIT dataset repository
git clone https://github.com/google-deepmind/limit.git
mv limit limit_dataset
```

### 3. Dataset Structure

The LIMIT dataset should be organized as:
```
limit_dataset/
├── bookqa/
│   ├── queries.json
│   ├── corpus.json
│   └── qrels.json
├── hotpotqa/
│   ├── queries.json
│   ├── corpus.json
│   └── qrels.json
├── nq/
│   ├── queries.json
│   ├── corpus.json
│   └── qrels.json
└── triviaqa/
    ├── queries.json
    ├── corpus.json
    └── qrels.json
```

## Running Evaluation

### Basic Evaluation

Run the combined evaluation on both MTEB and LIMIT datasets:

```bash
python benchmark/run_combined_evaluation.py
```

### Specific Evaluation Types

```bash
# Run only MTEB evaluation
python benchmark/run_combined_evaluation.py --eval-type mteb

# Run only LIMIT evaluation
python benchmark/run_combined_evaluation.py --eval-type limit

# Run stage comparison
python benchmark/run_combined_evaluation.py --eval-type comparison

# Run all evaluations
python benchmark/run_combined_evaluation.py --eval-type all
```

### Custom Parameters

```bash
# Custom dataset path
python benchmark/run_combined_evaluation.py --limit-dataset-path /path/to/limit

# Smaller sample size for faster evaluation
python benchmark/run_combined_evaluation.py --sample-size 500

# Specific task for comparison
python benchmark/run_combined_evaluation.py --task LIMIT-HotpotQA

# Debug logging
python benchmark/run_combined_evaluation.py --log-level DEBUG
```

## Evaluation Modes

### 1. Full Pipeline Evaluation
Evaluates the complete 3-stage pipeline on all LIMIT tasks:
- LIMIT-BookQA
- LIMIT-HotpotQA
- LIMIT-NaturalQuestions
- LIMIT-TriviaQA

### 2. Stage-by-Stage Comparison
Compares individual stages against the full pipeline:
- Stage 1 only (embedding model)
- Stage 1+2 (embedding + ColBERT)
- Full 3-stage pipeline

### 3. MTEB Standard Tasks
Evaluates on standard MTEB retrieval and clustering tasks for baseline comparison.

## Results

Results are saved to `benchmark/results/` with timestamps:
- `mteb_results_YYYYMMDD_HHMMSS.json` - MTEB evaluation results
- `limit_results_YYYYMMDD_HHMMSS.json` - LIMIT dataset results

### Metrics Tracked

Each evaluation tracks:
- **NDCG@10**: Normalized Discounted Cumulative Gain
- **Recall@10**: Recall at 10 documents
- **MAP@10**: Mean Average Precision at 10
- **MRR@10**: Mean Reciprocal Rank at 10
- **Evaluation time**: Time taken for evaluation
- **Pipeline configuration**: Model and settings used

## Expected Performance

Based on the 3-stage architecture:

1. **Stage 1**: Fast but lower accuracy (~0.4-0.6 NDCG@10)
2. **Stage 1+2**: Moderate speed, good accuracy (~0.6-0.8 NDCG@10)  
3. **Full Pipeline**: Slower but highest accuracy (~0.7-0.9 NDCG@10)

The pipeline should show significant improvement over individual stages, especially on complex tasks like HotpotQA.

## Troubleshooting

### Common Issues

1. **Dataset not found**: Ensure LIMIT dataset is downloaded and path is correct
2. **Memory issues**: Reduce sample size with `--sample-size 500`
3. **Model loading errors**: Check internet connection for model downloads
4. **MTEB not found**: Install with `pip install mteb`

### Performance Tips

- Use smaller sample sizes for development/testing
- Enable GPU acceleration if available
- Monitor memory usage during evaluation
- Consider using the "small" version of LIMIT dataset for faster iterations

## Interpreting Results

The evaluation will show:
- **Individual task scores** for each LIMIT task
- **Overall statistics** across all tasks
- **Stage comparison** showing incremental improvements
- **Performance interpretation** (excellent/good/moderate/needs improvement)

Look for:
- Consistent performance across different tasks
- Significant improvement from Stage 1 to full pipeline
- Competitive scores compared to MTEB baselines
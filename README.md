# Multilingual Receipt Item Classification

A comprehensive machine learning project exploring multiple approaches for classifying receipt items into categories, including embedding models, Top-k similarity, TF-IDF, and ensemble methods.

## Project Overview

This project focuses on classifying receipt items into distinct categories based on item names. The project systematically compares multiple machine learning approaches including traditional TF-IDF methods, modern embedding models, Top-k similarity strategies, and ensemble techniques to achieve optimal categorization performance.

## Current Performance

**Best Model Performance:**
- **Model:** TF-IDF Ensemble Method
- **F1 Score:** 0.9290 (best single run)
- **Consistent Range:** 0.902-0.929 (ensemble methods)
- **Runtime:** Fast execution
- **Key Finding:** Ensemble approach dramatically outperforms individual TF-IDF (0.929 vs 0.259)

## Experimental Results

### Model Comparison

| Approach | F1 Score | Runtime | Notes |
|----------|----------|---------|-------|
| **TF-IDF Ensemble (Run 3)** | **0.9290** | Fast | **🏆 BEST PERFORMING - Ensemble method** |
| TF-IDF Only (Run 1) | 0.2588 | Fast | Traditional TF-IDF without ensemble |
| TF-IDF Only (Run 2) | 0.2629 | Fast | Traditional TF-IDF without ensemble |
| Qwen3 Embedding (Top-3) | 0.5187 | 466 sec | Modern embedding approach |
| Qwen3 Embedding (Top-5) | 0.5870 | 440 sec | Top-k similarity strategy |
| Dummy Baseline | 0.0225 | Fast | Random classification baseline |

### Key Insights

- **Ensemble Method Superiority:** TF-IDF ensemble dramatically outperforms individual methods (0.929 vs 0.259)
- **Traditional vs Modern:** Ensemble TF-IDF outperforms embedding approaches by significant margin
- **Method Diversity:** Project successfully implements and compares 4 distinct ML approaches
- **Performance Range:** Results span from 0.0225 (baseline) to 0.929 (ensemble), showing clear method differentiation
- **Speed-Accuracy Balance:** Ensemble methods achieve best accuracy while maintaining fast inference

## Technical Architecture

### Core Components

- **Embedding Models:** Qwen3-Embedding-0.6B for semantic representation
- **Similarity Computation:** Cosine similarity between query and document embeddings
- **Evaluation Strategy:** Top-k classification with F1 score calculation
- **Configuration Management:** JSON-based model configuration system

### Model Pipeline

1. **Text Preprocessing:** Clean and normalize item names
2. **Embedding Generation:** Generate semantic embeddings for items and classes
3. **Similarity Scoring:** Compute cosine similarity between embeddings
4. **Top-k Classification:** Select top-k most similar classes
5. **F1 Evaluation:** Calculate weighted F1 scores for performance assessment

## Key Improvements

1. **Comprehensive Method Comparison:** Systematic evaluation of 4 distinct approaches (Embedding, Top-k, TF-IDF, Ensemble)
2. **Ensemble Method Development:** TF-IDF ensemble achieves 0.929 F1 score - dramatically outperforming individual methods
3. **Multi-Strategy Evaluation:** Top-k similarity strategies with embedding models
4. **Performance Benchmarking:** Clear differentiation between traditional and modern ML approaches
5. **Robust Experimental Design:** Multiple runs validate consistency and method effectiveness

## Next Steps

### Model Enhancement
- [ ] Optimize ensemble method parameters to push beyond 0.929 F1 score
- [ ] Investigate hybrid ensemble combining TF-IDF + embedding approaches
- [ ] Fine-tune embedding models specifically for receipt classification task
- [ ] Develop weighted ensemble strategies based on confidence scores

### Evaluation Optimization
- [ ] **Dynamic k Selection:** Optimize k value based on confidence scores
- [ ] **Threshold-based Classification:** Implement confidence thresholds for predictions
- [ ] **Cross-validation:** Implement robust evaluation with multiple data splits

### Performance Targets
- [ ] **Primary Goal:** Achieve F1 score of 0.7+ with Top-3 classification
- [ ] **Efficiency Goal:** Reduce inference time while maintaining accuracy
- [ ] **Scalability:** Support larger datasets and more categories

## Technical Requirements

```bash
pip install torch pandas scikit-learn sentence-transformers transformers
```

## Usage

```python
from utils import evaluate_qwen3_embedding_topk
from models import ParaphraserModel, ParaphraserConfig

# Evaluate with Top-5 classification
score = evaluate_qwen3_embedding_topk(
    df=your_dataframe,
    column_name="cleaned_item_name", 
    model_class=ParaphraserModel,
    config_class=ParaphraserConfig,
    k=5
)
```

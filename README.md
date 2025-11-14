# C5.0 Decision Tree Optimization with Pruning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)

## 📋 Overview

Implementation and comparison of different **pruning techniques** for C5.0-style decision trees to reduce complexity while maintaining accuracy on financial investment data.

## 🚀 Quick Start

```bash
pip install pandas numpy scikit-learn matplotlib seaborn

# Run pruning comparison
python c50_pruning_optimization.py

# Run full analysis with visualizations
python c50_complete_analysis.py
```

## 📊 Results

### Pruning Comparison (40 records, 24 features)

| Model | Depth | Leaves | Accuracy | Reduction |
|-------|-------|--------|----------|-----------|
| Base (No Pruning) | 2 | 3 | 0.8333 | - |
| **Pre-Pruning** | 1 | 2 | **0.9167** | **33.3%** |
| Post-Pruning (CCP) | 2 | 3 | 0.8333 | 0% |
| **Combined** | 1 | 2 | **0.9167** | **33.3%** |

### Boosting Performance

| Metric | Base DT | Boosted | Improvement |
|--------|---------|---------|-------------|
| Accuracy | 0.9167 | 0.9167 | - |
| Precision | 0.8400 | 0.9500 | +13.1% |
| F1-Score | 0.8800 | 0.9300 | +5.7% |

**Top Features:** Debentures (40%), Reason_FD (25%), Government_Bonds (20%)

## ⚡ Features

- **Pre-Pruning:** Limits tree depth during construction (`max_depth=5`)
- **Post-Pruning:** Cost Complexity Pruning with automatic alpha optimization
- **Combined Pruning:** Best balance of complexity and accuracy
- **Boosting:** AdaBoost with 50 pruned estimators
- **Visualizations:** 4-panel chart (performance, features, confusion matrix, metrics)

## 🎯 Key Findings

✅ 33.3% complexity reduction  
✅ +8.34% accuracy improvement  
✅ Better interpretability with fewer nodes  
✅ Boosting improves precision by 13%  

## 📁 Files

- `c50_pruning_optimization.py` - Pruning comparison (saves `.txt` output)
- `c50_complete_analysis.py` - Full analysis (generates `.png` charts)
- `Finance_data.csv` - Dataset (40 samples, 22 features after preprocessing)

## 📄 License

MIT License

---

**⭐ Star if useful! Made with Python & scikit-learn**

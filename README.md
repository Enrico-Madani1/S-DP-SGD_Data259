# Comparing Standard vs Selective DP-SGD: Privacy-Utility-Fairness Tradeoffs

This project studies how different Standard Differential Privacy Stochastic Gradient Descent (DP-SGD) with Selective DP-SGD on the Adult Income Census dataset. This project analyzes the tradeoffs between privacy, utility, and fairness of both models.

For more information about the result and methodology, feel free to check out the `Paper_DATA259.pdf` and `Poster_DATA259.pdf`
### Setup

1. **Clone or download this repository**

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Dependencies

The project requires the following packages (see `requirements.txt` for versions):

- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Baseline model and metrics
- `torch` - PyTorch for neural network models
- `opacus` - Differential privacy for PyTorch
- `ucimlrepo` - UCI ML repository data access
- `matplotlib` - Visualization
- `seaborn` - Statistical visualization (optional, for parameter sweep)
- `tqdm` - Progress bar

## Reproducing Results

### Quick Start (Main Results)

Run the main notebook to reproduce the core comparison:
```bash
jupyter notebook Project_DATA259.ipynb
```

Execute **all cells in order** to get:
1. Baseline model (no privacy)
2. Standard DP-SGD (ε≈0.64)
3. Selective DP-SGD (ε≈0.92)

**Expected runtime**: ~10-15 minutes (after going through all the files)

Note: the Analysis 2 plotting parts might take a while to finish

**Results will show**:
- Overall accuracy and AUC for each model
- Fairness metrics by sex and race
- Privacy budgets (epsilon values)

**Key Finding**: It depends. S-DP-SGD works better ONLY under carefully tuned, moderate privacy settings (C ≤ 2.0, σ ≤ 1.5).

**Implications**:
- No universal "best" approach → depends on context
- Privacy-fairness-FAIRNESS triple trade-off (sex vs. race)
- Need careful validation before deployment




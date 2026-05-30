# Crop Yield Forecasting with Multi-Layer Perceptron

A PyTorch-based neural network for predicting country-wise crop yields using climate, soil, and land cover data.

## Overview

This project implements a multi-layer perceptron (MLP) for time-series forecasting of crop yields across 102 crop types using environmental and agricultural data. The model processes 13 years of climate, soil moisture, temperature, and land cover data to predict yield outcomes with masked loss handling for sparse agricultural datasets.

## Key Results

- **SMAPE:** 35.14% on test set
- **Mean R²:** 0.669 (after outlier removal)  
- **Best crops:** Kapok fruit (R² = 0.980), Sugar crops (R² = 0.972), Jute (R² = 0.956)
- **Training:** Converged at 184 epochs with early stopping

## Dataset

**Environmental Variables (2010-2022):**
- Rainfall and snowfall precipitation
- Soil moisture (4 depth levels: 0-10cm, 10-40cm, 40-100cm, 100-200cm)
- Soil temperature (4 depth levels)
- Plant transpiration and canopy water
- Terrestrial water storage

**Engineered Features:**
- Rainfall to soil evaporation ratios
- Water storage to evaporation ratios  
- Soil moisture to temperature ratios
- 21 principal components (95% variance retained)

**Scale:** 194,298 coordinate points × 156 months × multiple environmental variables

## Model Architecture

```
Input Layer:     40 features (environmental + PCA components)
Hidden Layer 1:  256 neurons + ReLU + Dropout(0.4) + BatchNorm
Hidden Layer 2:  128 neurons + ReLU + Dropout(0.4) + BatchNorm  
Hidden Layer 3:  102 neurons + ReLU + Dropout(0.4) + BatchNorm
Output Layer:    102 neurons (crop yields)
```

**Key Features:**
- Custom masked loss function for handling sparse yield data
- Early stopping (patience=20 epochs)
- Learning rate scheduling (ReduceLROnPlateau)
- Adam optimizer with L2 regularization

## Installation

```bash
pip install torch torchvision sklearn matplotlib pandas numpy tqdm
```

## Usage

### Data Preprocessing
```python
# Run the notebook or extract the preprocessing functions
# Handles 14 environmental data files + land cover + yield data
python -c "
# File cleaning and feature engineering
# PCA dimensionality reduction  
# Train/validation/test split (70/15/15)
"
```

### Training
```python
# The model trains automatically with:
# - Batch size: 64
# - Learning rate: 0.001
# - Max epochs: 1000 (early stopping enabled)
# - GPU acceleration if available
```

### Results
The notebook generates:
- Loss curves (training vs validation)
- Per-crop R² scores and rankings
- Model performance metrics (MAE, RMSE, SMAPE)
- Comprehensive visualizations

## Files Structure

```
crop-yield-forecasting/
├── ML_ASI_NOTEBOOK-FORCAST_CROPS.ipynb  # Main analysis notebook
├── ml_report_finished.docx              # Detailed methodology report
├── requirements.txt                     # Dependencies
├── data/                               # Environmental and yield datasets
├── outputs/                           # Generated plots and metrics
└── README.md                         # This file
```

## Methodology

### Data Processing
1. **Cleaning:** Remove repeated values, outliers (±3σ), interpolation for missing data
2. **Feature Engineering:** Create ratios between environmental variables
3. **Dimensionality Reduction:** PCA to 21 components (95% variance)
4. **Masking:** Handle sparse crop yield data with binary masks

### Model Training
- **Architecture:** Feed-forward MLP with 3 hidden layers
- **Loss:** Masked MSE (ignores NaN/zero yield values)
- **Regularization:** Dropout, batch normalization, L2 weight decay
- **Optimization:** Adam with learning rate scheduling

### Evaluation
- **Metrics:** MAE, RMSE, R², SMAPE
- **Validation:** 15% holdout with early stopping
- **Testing:** 15% final evaluation set

## Performance Analysis

**Strong Performance Crops:**
- Kapok fruit, Sugar crops, Jute, Cotton lint
- Generally crops with consistent growing patterns

**Challenging Crops:**  
- Quinoa, Kenaf (removed as outliers)
- Crops with sparse training data or extreme yield variations

**Overall:** Model achieves reasonable accuracy (35% SMAPE) for agricultural forecasting, where high variability is expected.

## Technical Details

- **Training Environment:** GPU-accelerated PyTorch
- **Memory Efficiency:** Batch processing with masked operations
- **Convergence:** Early stopping prevents overfitting
- **Scalability:** Handles large-scale environmental datasets

## Future Improvements

- Extended temporal range for sparse crops
- Advanced sequence modeling (LSTM/Transformer)
- Weather forecast integration
- Regional climate zone stratification
- Ensemble methods for uncertainty quantification

## Citation

If you use this work, please cite:

```
Finnigan, D.A. (2025). Crop Yield Forecasting with Multi-Layer Perceptron: 
Time-series Prediction Using Climate and Soil Data. GitHub Repository.
```

## Contact

**David A. Finnigan**  
MSc AI & Adaptive Systems (Distinction)  
[GitHub](https://github.com/dafinnigan91) | [LinkedIn](https://linkedin.com/in/david-finnigan-ai)

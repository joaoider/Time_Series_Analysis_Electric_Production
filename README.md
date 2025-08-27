# Advanced Time Series Analysis - Electric Production

This project implements a comprehensive time series analysis for electric production data using a modular and organized architecture with advanced statistical testing and modeling capabilities.

## 📋 Project Structure

```
Time_Series_Analysis_Electric_Production/
├── config.py                      # Configurations and constants
├── data_loader.py                 # Data loading and preparation
├── data_analysis.py               # Exploratory data analysis
├── stationarity.py                # Basic stationarity tests
├── transformation.py              # Data transformations
├── models.py                      # Basic time series models
├── evaluation.py                  # Model evaluation and comparison
├── visualization.py               # Visualization functions
├── advanced_statistical_tests.py  # Advanced statistical testing
├── advanced_models.py             # Advanced time series models
├── main.py                        # Basic analysis script
├── advanced_analysis.py           # Advanced analysis script
├── requirements.txt               # Dependencies
├── README.md                      # Documentation
└── Electric_Production.ipynb      # Original notebook
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Time_Series_Analysis_Electric_Production
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Features

### 1. Data Loading (`data_loader.py`)
- Automatic Kaggle dataset download
- Data loading and preparation
- Train/test split functionality
- Lagged variable preparation for ML models

### 2. Exploratory Analysis (`data_analysis.py`)
- Basic statistical summaries
- Rolling statistics calculation
- Stationarity verification by data splitting

### 3. Basic Stationarity Tests (`stationarity.py`)
- Augmented Dickey-Fuller (ADF) test
- Automatic result interpretation
- Multiple window testing
- Complete analysis summary

### 4. Data Transformations (`transformation.py`)
- Logarithmic transformation (Box-Cox)
- Trend removal with moving averages
- Exponential decay transformation
- Complete transformation pipeline

### 5. Basic Time Series Models (`models.py`)
- **Persistence Model**: Simple baseline
- **Autoregression (AR)**: AR(p,d,q) model
- **Moving Average (MA)**: MA(p,d,q) model
- **ARIMA**: Integrated model
- **SARIMAX**: Seasonal model
- **AutoARIMA**: Automatic parameter selection
- **XGBoost**: Machine learning model

### 6. Model Evaluation (`evaluation.py`)
- Metric calculation (MSE, RMSE, MAE, MAPE)
- Pairwise model comparison
- Automatic model ranking
- Detailed evaluation reports

### 7. Visualizations (`visualization.py`)
- Time series plots
- Rolling statistics
- Seasonal decomposition
- ACF and PACF plots
- Model comparison charts
- Residual analysis

## 🔬 ADVANCED FEATURES

### 8. Advanced Statistical Testing (`advanced_statistical_tests.py`)
- **Comprehensive Stationarity Tests**:
  - ADF with multiple specifications (6 combinations)
  - KPSS test (complementary to ADF)
  - Zivot-Andrews test (structural breaks)
  - Phillips-Perron test (heteroscedasticity robust)
  - Structural break detection (t-tests, Levene)

- **Normality Tests**:
  - Shapiro-Wilk (most powerful for small samples)
  - Anderson-Darling (sensitive to tails)
  - Kolmogorov-Smirnov (distribution comparison)
  - Jarque-Bera (skewness and kurtosis)

- **Heteroscedasticity Tests**:
  - Breusch-Pagan test
  - White test (simplified)

- **Autocorrelation Tests**:
  - Ljung-Box test (multiple lags)
  - Durbin-Watson statistic

- **ARCH Effects Testing**:
  - ARCH(1) and ARCH(2) models
  - Parameter significance testing

### 9. Advanced Time Series Models (`advanced_models.py`)
- **Regime Switching Models**:
  - Markov Switching Regression
  - Markov Switching Autoregression

- **Multivariate Models**:
  - VAR (Vector Autoregression) with automatic lag selection
  - VECM (Vector Error Correction) for cointegration

- **Exponential Smoothing**:
  - Holt-Winters Additive/Multiplicative
  - Multiplicative trend models

- **Volatility Models**:
  - GARCH(1,1), GARCH(2,1)
  - EGARCH (Exponential GARCH)
  - GJR-GARCH (asymmetric effects)

- **Advanced Machine Learning**:
  - Random Forest with feature engineering
  - Gradient Boosting
  - Support Vector Regression
  - Neural Networks (MLP)

- **Ensemble Methods**:
  - Simple averaging
  - Weighted averaging (MSE-based)
  - Median ensemble

## 🎯 How to Use

### Basic Analysis
```bash
python main.py
```

### Advanced Analysis
```bash
python advanced_analysis.py
```

### Specific Statistical Tests
```bash
python advanced_analysis.py  # Choose option 2
```

### Modular Usage
```python
from advanced_statistical_tests import AdvancedStatisticalTests
from advanced_models import AdvancedTimeSeriesModels

# Advanced statistical testing
advanced_tester = AdvancedStatisticalTests(data)
comprehensive_report = advanced_tester.generate_comprehensive_report()

# Advanced modeling
advanced_models = AdvancedTimeSeriesModels(data)
all_results = advanced_models.run_all_advanced_models()
```

## 📈 Analysis Flow

### Basic Flow
1. **Loading**: Dataset download and preparation
2. **Exploration**: Initial analysis and statistics
3. **Stationarity**: ADF test and verification
4. **Transformation**: Apply transformations for stationarity
5. **Modeling**: Train multiple models
6. **Evaluation**: Compare and rank models
7. **Visualization**: Generate informative plots

### Advanced Flow
1. **Comprehensive Testing**: Multiple stationarity tests, normality, heteroscedasticity
2. **Advanced Transformations**: Sophisticated data preprocessing
3. **Advanced Modeling**: Regime switching, VAR, GARCH, ML ensembles
4. **Robust Evaluation**: Multiple metrics and statistical validation
5. **Professional Reporting**: Executive summaries and recommendations

## 🔧 Configuration

Settings can be adjusted in `config.py`:

- **Model Parameters**: ARIMA orders, SARIMAX parameters
- **Analysis Settings**: Rolling windows, test sizes
- **Visualization**: Colors, figure sizes
- **Test Settings**: Significance levels, confidence intervals

## 📊 Expected Results

The system will:
- Perform comprehensive statistical testing
- Identify stationarity and normality characteristics
- Apply appropriate transformations
- Train multiple advanced models
- Compare performance using multiple metrics
- Generate professional visualizations
- Provide model rankings and recommendations
- Generate comprehensive statistical reports

## 🛠️ Dependencies

### Core Dependencies
- **pandas**: Data manipulation
- **numpy**: Numerical computation
- **matplotlib/seaborn**: Visualizations
- **scikit-learn**: Machine learning and metrics
- **statsmodels**: Time series models
- **pmdarima**: AutoARIMA
- **xgboost**: Gradient boosting

### Advanced Dependencies
- **arch**: GARCH and volatility modeling
- **scipy**: Statistical transformations
- **kaggle**: API for dataset access

## 📝 Example Output

```
================================================================================
ADVANCED TIME SERIES ANALYSIS - ELECTRIC PRODUCTION
================================================================================

1. LOADING DATA...
Data loaded: (397, 1)

2. BASIC EXPLORATORY ANALYSIS...
Basic statistics:
  count: 397
  mean: 88.877
  std: 10.234
  ...

3. ADVANCED STATISTICAL TESTING...
3.1 Advanced stationarity tests...
3.2 Normality tests...
3.3 Heteroscedasticity tests...
3.4 Autocorrelation tests...
3.5 ARCH effects tests...

Executive Summary of Statistical Tests:
  overall_stationarity: Likely Non-Stationary
  normality_assessment: Likely Non-Normal
  heteroscedasticity: Present
  recommendations:
    - Apply differencing or transformations
    - Consider non-parametric methods
    - Use robust standard errors or GARCH models

4. APPLYING ADVANCED TRANSFORMATIONS...
Transformations applied successfully!

5. TRAINING BASIC MODELS...
6. TRAINING ADVANCED MODELS...
7. TRAINING XGBOOST...
8. COMPREHENSIVE MODEL EVALUATION...

Model Comparison:
  persistence: MSE = 0.0084
  arima: MSE = 0.0019
  sarimax: MSE = 0.0097
  ml_random_forest: MSE = 0.0012
  garch_garch_11: MSE = 0.0008
  ensemble_weighted: MSE = 0.0006

Best Model: ensemble_weighted (MSE: 0.0006)
```

## 🔬 Statistical Rigor

This project implements **academic-grade statistical testing**:

- **Multiple Test Specifications**: Ensures robustness across different assumptions
- **Comprehensive Diagnostics**: Tests for all major statistical violations
- **Automatic Recommendations**: AI-driven suggestions for data treatment
- **Professional Reporting**: Executive summaries suitable for business/ academic use

## 🚀 Performance Features

- **Parallel Processing**: Efficient model training
- **Automatic Parameter Selection**: AI-driven optimization
- **Ensemble Methods**: Combines multiple models for better performance
- **Feature Engineering**: Advanced lag and rolling statistics
- **Cross-Validation**: Robust model evaluation

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the project
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is under MIT license. See LICENSE file for details.

## 📞 Support

For questions or issues:
- Open a GitHub issue
- Contact maintainers

---

**Developed with ❤️ for advanced time series analysis**

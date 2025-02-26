<<<<<<< HEAD
# International Airlines Group(IAG) Customer Recommendation Prediction System

## Project Overview

This machine learning system predicts customer recommendation likelihood for IAG insurance products based on customer attributes and satisfaction metrics. The project demonstrates advanced ML techniques including:

- Custom feature engineering with domain-specific transformations
- Sophisticated preprocessing pipelines for categorical and numerical data
- Multiple modeling approaches (scikit-learn and statsmodels)
- Class imbalance handling with SMOTE
- Model calibration for reliable probability estimates
- Deployment as a REST API with FastAPI

## Key Features

### Advanced Data Preprocessing

- **Numeric Feature Transformation**: Custom binning of trust and price satisfaction scores with domain-specific thresholds
- **Feature Engineering**: Creation of interaction terms, satisfaction gaps, and customer segments
- **Categorical Encoding**: Optimized one-hot encoding with reference categories

### Modeling Approaches

1. **Scikit-learn Pipeline**:
   - Custom transformers for domain-specific feature engineering
   - Logistic regression with multinomial classification
   - Feature selection to improve model interpretability

2. **Statsmodels Implementation**:
   - Detailed statistical analysis with p-values and odds ratios
   - McFadden's Pseudo R-squared evaluation
   - Comprehensive feature importance analysis

### Performance Optimization

- **Class Imbalance Handling**: Strategic oversampling with SMOTE
- **Calibrated Probabilities**: Sigmoid calibration for reliable probability estimates
- **Hyperparameter Tuning**: Optimized C values and solver selection

### API Deployment

- **FastAPI Implementation**: Modern, high-performance REST API
- **Pydantic Validation**: Type-safe request/response handling
- **Containerization-Ready**: Structured for easy Docker deployment

## Technical Stack

- **Data Processing**: Pandas, NumPy
- **Machine Learning**: scikit-learn, statsmodels, imbalanced-learn
- **API Framework**: FastAPI
- **Serialization**: Pickle

## Model Performance

The final model achieves strong predictive performance across all recommendation categories:

- High accuracy in identifying promoters (customers likely to recommend)
- Balanced precision and recall for detractors
- Robust handling of edge cases (super detractors)

## Project Structure

```
├── notebooks/
│   ├── model_pipeline.ipynb     # Model development and evaluation
│   └── data_pipeline.ipynb      # Data preprocessing and feature engineering
├── src/
│   └── backend/
│       └── main.py              # FastAPI implementation
├── model_sklearn/               # Scikit-learn model artifacts
├── model_statsmodels/           # Statsmodels artifacts
└── data/                        # Data directory (not included in repo)
```

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running the API

```bash
uvicorn src.backend.main:app --reload
```

### Making Predictions

```python
import requests

customer_data = {
    "iag_business_unit_ug": "NRMA",
    "iag_age_band_auto": "51-60",
    "iag_tenure_band_enum": "5-10 years",
    "iag_site_ug": "Branch",
    "iag_product_type_auto": "Comprehensive",
    "iag_region_ug": "Metro",
    "iag_trust_confidence_scale11": 8.0,
    "iag_value_price_of_policy_reflects_scale11": 7.0
}

response = requests.post("http://localhost:8000/predict", json=customer_data)
prediction = response.json()
```

## Future Enhancements

- Integration with cloud-based model monitoring
- A/B testing framework for model deployment
- Explainable AI components for business stakeholders
- Automated retraining pipeline with data drift detection

## Conclusion

This project demonstrates advanced machine learning techniques applied to a real-world customer recommendation prediction problem. The implementation balances model performance with interpretability, making it valuable for both technical and business stakeholders.
=======
# International-Airlines-Group-Customer-Prediction-Analytics
>>>>>>> acfe21640bc29086ed18b7bd128ff0f4f7b53d20

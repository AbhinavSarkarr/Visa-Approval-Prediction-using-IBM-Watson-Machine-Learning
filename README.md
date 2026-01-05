# H-1B Visa Approval Prediction

A machine learning-powered web application that predicts the outcome of H-1B visa applications using historical data analysis and classification algorithms. The system provides real-time predictions to help prospective applicants understand their approval chances.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Machine Learning Approach](#machine-learning-approach)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Technologies](#technologies)
- [Project Structure](#project-structure)
- [Author](#author)
- [License](#license)

## Overview

The H-1B is an employment-based visa in the United States that allows U.S. employers to temporarily employ foreign workers in specialty occupations. This project analyzes 2011-2016 H-1B petition disclosure data to predict application outcomes using machine learning.

### Key Questions Addressed

- What are the top companies applying for H-1B visas?
- What is the trend in total H-1B applications over time?
- What are the most popular job titles and work sites for H-1B holders?
- What factors most influence visa approval?

## Features

- **Real-Time Prediction**: Web interface for instant visa status predictions
- **Multiple ML Models**: Comparison of Decision Tree, Random Forest, and Logistic Regression
- **Exploratory Data Analysis**: Comprehensive visualizations of H-1B trends
- **Feature Importance Analysis**: Understanding which factors impact approval most
- **User-Friendly Interface**: Flask-based web application with intuitive design

## Dataset

- **Source**: H-1B Petition Disclosure Data (2011-2016)
- **Size**: Over 3 million records
- **Features Used**:
  - SOC_NAME (Occupation Category)
  - Prevailing Wage
  - Year of Application
  - Job Duration
  - Employer Information
  - Work Location

## Machine Learning Approach

### Algorithms Implemented

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| **Logistic Regression** | Multinomial classification using MLE | Primary prediction model (best accuracy) |
| **Random Forest** | Ensemble of decision trees | Feature importance analysis |
| **Decision Tree** | Sequential decision modeling | Baseline comparison |

### Feature Engineering

1. **Outlier Removal**: Statistical cleaning of wage and duration data
2. **Label Encoding**: Categorical variable transformation
3. **Feature Selection**: Based on correlation and importance analysis

### Model Pipeline

```
Raw Data → Preprocessing → Feature Engineering → Model Training → Evaluation → Deployment
    │            │               │                    │              │            │
    └── Clean ───┴── Encode ─────┴── Split ───────────┴── Tune ──────┴── Flask ───┘
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/AbhinavSarkarr/visa-prediction.git
cd visa-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
flask
pandas
numpy
scikit-learn
matplotlib
seaborn
```

## Usage

### Running the Web Application

```bash
python app.py
```

Access the application at `http://localhost:5000`

### Making Predictions

1. Navigate to the prediction page
2. Enter applicant details:
   - Occupation category
   - Prevailing wage
   - Application year
   - Job duration
3. Click "Predict" to get the visa status prediction

## Model Performance

| Metric | Logistic Regression | Random Forest | Decision Tree |
|--------|---------------------|---------------|---------------|
| Accuracy | **89.2%** | 87.5% | 82.1% |
| Precision | 0.88 | 0.86 | 0.80 |
| Recall | 0.87 | 0.85 | 0.79 |

### Feature Importance

The top factors influencing visa approval:
1. **Prevailing Wage** - Higher wages correlate with approval
2. **Occupation Category** - Specialty occupations have higher rates
3. **Employer Track Record** - Companies with history of approvals
4. **Application Year** - Policy changes affect outcomes

## Technologies

| Technology | Purpose |
|------------|---------|
| **Flask** | Web application framework |
| **scikit-learn** | Machine learning models |
| **pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computations |
| **Matplotlib/Seaborn** | Data visualization |
| **HTML/CSS** | Frontend templates |

## Project Structure

```
visa-prediction/
├── app.py                      # Flask application
├── model.py                    # ML model training
├── templates/
│   ├── index.html             # Home page
│   ├── info.html              # Information page
│   └── predict.html           # Prediction interface
├── static/
│   └── css/                   # Stylesheets
├── data/
│   └── h1b_data.csv          # Training dataset
├── models/
│   └── visa_model.pkl        # Trained model
├── notebooks/
│   └── EDA.ipynb             # Exploratory analysis
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Screenshots

### Home Page
![Home Page](https://user-images.githubusercontent.com/54931557/128338883-4dcb8e7e-9b2f-416a-b8df-074093637e45.png)

### Information Page
![Info Page](https://user-images.githubusercontent.com/54931557/128338898-340e6bdd-5333-4a26-a399-4cc75ca9e929.png)

### Prediction Page
![Prediction Page](https://user-images.githubusercontent.com/54931557/128338910-fd4f5ab7-c41e-46f4-a0a9-666ee6ee4458.png)

## Future Enhancements

- Integration of more recent H-1B data (2017-2024)
- Implementation of additional algorithms (SVM, Neural Networks)
- Enhanced UI with real-time validation
- API endpoint for programmatic access
- Location-based analysis and visualization

## References

- [Kaggle H-1B Visa Dataset](https://www.kaggle.com/nsharan/h-1b-visa)
- [H-1B Visa Benefits Overview](https://www.immi-usa.com/h1b-visa/h1b-visa-benefits/)
- [Logistic Regression Theory](https://www.javatpoint.com/logistic-regression-in-machine-learning)
- [Random Forest Optimization](https://towardsdatascience.com/optimizing-hyperparameters-in-random-forest-classification-ec7741f9d3f6)

## Author

**Abhinav Sarkar**
- GitHub: [@AbhinavSarkarr](https://github.com/AbhinavSarkarr)
- LinkedIn: [abhinavsarkarrr](https://www.linkedin.com/in/abhinavsarkarrr)
- Portfolio: [abhinav-ai-portfolio.lovable.app](https://abhinav-ai-portfolio.lovable.app/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- IBM Watson Machine Learning for initial guidance
- SmartInternz for project mentorship
- The open-source community for excellent ML libraries

---

<p align="center">
  <strong>Predict your H-1B visa approval chances with machine learning</strong>
</p>

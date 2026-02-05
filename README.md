# Boston Housing Price Prediction

A simple **machine learning project** that predicts house prices in Boston using **Linear Regression**.  
The project includes exploratory data analysis (EDA) in a Jupyter notebook and a basic **Flask web application** for making real-time predictions.

## Features

- Exploratory Data Analysis of the classic Boston Housing dataset
- Linear Regression model trained on 13 housing features
- Feature scaling using StandardScaler
- Flask-based web app with two prediction interfaces:
  - Web form (browser)
  - REST API endpoint (`/predict_api`)

## Dataset

- **Source**: Modified version of the classic Boston Housing dataset (506 samples)
- **Target**: MEDV → renamed to **Price** (median value of owner-occupied homes in $1000s)
- **Features** (13):
  - CRIM: per capita crime rate
  - ZN: proportion of residential land zoned for lots >25,000 sq.ft.
  - INDUS: proportion of non-retail business acres per town
  - CHAS: Charles River dummy variable (1 = tract bounds river; 0 otherwise)
  - NOX: nitric oxides concentration (parts per 10 million)
  - RM: average number of rooms per dwelling
  - AGE: proportion of owner-occupied units built prior to 1940
  - DIS: weighted distances to five Boston employment centres
  - RAD: index of accessibility to radial highways
  - TAX: full-value property-tax rate per $10,000
  - PTRATIO: pupil-teacher ratio by town
  - B: 1000(Bk - 0.63)² where Bk is the proportion of blacks by town
  - LSTAT: % lower status of the population

Strongest correlations with price: RM (+), LSTAT (−), PTRATIO (−)

## Project Structure
Project_Boston_Housing_Pricing/
├── HousingData.csv                  # Dataset
├── Linear Regression ML Implementation[1].ipynb   # EDA + partial model code
├── app.py                           # Flask web application
├── regmodel.pkl                     # Trained Linear Regression model
├── scaling.pkl                      # Fitted StandardScaler
├── requirements.txt                 # Dependencies
├── templates/                       # Flask HTML templates
│   └── home.html                    # (should contain prediction form – add if missing)
├── .gitignore
└── LICENSE                          # Apache 2.0

**Note**: The notebook currently performs EDA and correlation analysis but does **not** include model training, evaluation, or pickle saving. The pickled files (`regmodel.pkl`, `scaling.pkl`) were likely created in an uncommitted or separate script.

## Installation & Setup

1. Clone the repository
   ```bash
   git clone https://github.com/Sham-S08/Project_Boston_Housing_Pricing.git
   cd Project_Boston_Housing_Pricing
2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS / Linux
   source venv/bin/activate

## Technologies Used
- Python 3
- pandas, numpy
- scikit-learn (LinearRegression + scaling)
- Flask (web framework)
- matplotlib (EDA – optional)
- pickle (model persistence)

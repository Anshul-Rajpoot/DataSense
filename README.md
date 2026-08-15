# 📊 DataSense

### Intelligent Data Cleaning, Exploratory Data Analysis & Machine Learning Platform

DataSense is an interactive **Streamlit-based data science platform** that helps users transform raw tabular datasets into analysis-ready and machine-learning-ready data.

Instead of manually inspecting datasets for duplicates, missing values, incorrect data types, outliers, and statistical patterns, DataSense provides a guided workflow for **data quality analysis, preprocessing, exploratory data analysis, model comparison, and report generation**.

---

## 🚀 Features

### 📂 1. Dataset Upload

Upload tabular datasets directly through the Streamlit interface.

**Supported formats:**

* CSV
* XLS
* XLSX

After uploading, DataSense immediately provides a high-level overview of the dataset.

---

### 🔍 2. Data Quality Analysis

Automatically analyzes the uploaded dataset for common data-quality issues.

It identifies:

* Number of rows and columns
* Missing cells
* Duplicate rows
* Numerical columns
* Categorical columns
* Unique values
* Current data types
* Detected data types
* Potential data-type conversion issues

The application maintains both the **original dataset** and a separate **working dataset**, allowing cleaning operations without destroying the original upload.

---

### 🧹 3. Interactive Data Cleaning

DataSense does not blindly modify the dataset.

Instead, it allows the user to inspect problems and choose how they should be handled.

#### Duplicate Handling

* Detect duplicate rows
* View duplicate records
* Remove duplicates with user approval
* Keep duplicates if they are intentional

#### Missing Value Handling

Supported strategies include:

* Delete rows
* Mean imputation
* Median imputation
* Mode imputation
* Forward fill
* Backward fill
* Constant-value imputation
* Leave unchanged

The system also provides an **automatic recommendation** based on the characteristics of the column.

For numerical columns, recommendations consider:

* Distribution skewness
* Potential outliers

For categorical columns, the system can recommend mode-based imputation.

---

### 📈 4. Outlier Detection

Numerical columns can be analyzed using:

* **IQR (Interquartile Range)**
* **Z-score**

Users can choose to:

* Keep outliers
* Remove outliers
* Cap outliers

The application also displays the number and percentage of detected outliers for numerical columns.

---

### 🔢 5. Data-Type Detection

DataSense attempts to identify columns whose actual contents do not match their stored pandas datatype.

For example:

```text
Current type: object
Detected type: numeric

Recommendation:
Convert to numeric
```

It can identify potential:

* Numeric columns stored as text
* Datetime columns stored as text

It also displays examples of values that could not be converted successfully.

---

### ↩️ 6. Undo & Reset

Cleaning operations are tracked using Streamlit session state.

Users can:

* Undo the last cleaning operation
* Reset the working dataset to the original uploaded dataset

This prevents accidental permanent modification of the original data during an interactive cleaning session.

---

### 📊 7. Exploratory Data Analysis

DataSense provides automated EDA for numerical and categorical variables.

#### Numerical Analysis

Includes:

* Descriptive statistics
* Mean
* Median
* Standard deviation
* Minimum
* Maximum
* Histogram
* Box plot

#### Categorical Analysis

Includes:

* Unique category count
* Most frequent value
* Frequency distribution
* Percentage distribution
* Bar chart

---

### 🔗 8. Correlation Analysis

The platform automatically calculates correlations between numerical variables.

It also identifies strong relationships above a configurable correlation threshold.

Example:

```text
Experience ↔ Salary
Correlation = 0.82
```

This helps users quickly identify potentially important relationships between numerical features.

---

### 💡 9. Automated Data Insights

DataSense generates rule-based insights from the dataset.

Examples include:

* Columns with high missing-value percentages
* Strong correlations
* Highly skewed numerical variables
* High-cardinality categorical variables
* Potential outlier-heavy columns

This provides a quick interpretation of the dataset without requiring the user to manually inspect every statistic.

---

### 🤖 10. Machine Learning Pipeline

DataSense includes an end-to-end machine learning workflow.

The user selects a target column, after which the application attempts to determine whether the problem is:

* Classification
* Regression

The ML pipeline automatically handles preprocessing for numerical and categorical features.

### Numerical Pipeline

```text
Missing Value Imputation
        ↓
Standard Scaling
```

### Categorical Pipeline

```text
Missing Value Imputation
        ↓
One-Hot Encoding
```

These transformations are combined using **Scikit-learn's `ColumnTransformer` and `Pipeline`**.

This prevents preprocessing logic from being manually duplicated for different column types.

---

## 🧠 Machine Learning Models

### Classification

Depending on the selected configuration, DataSense can compare:

* Logistic Regression
* Decision Tree
* Random Forest
* K-Nearest Neighbors
* Support Vector Machine
* XGBoost *(when available)*

### Regression

Available models include:

* Linear Regression
* Ridge Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor

Users can also enable slower/heavier models when required.

---

## 📏 Model Evaluation

### Classification Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

### Regression Metrics

* MAE
* MSE
* RMSE
* R² Score

The application compares the selected models and identifies the best-performing model based on the appropriate evaluation score.

For tree-based models, DataSense can also display the most important features.

---

## 📑 Reports & Downloads

Users can export the processed results in multiple formats.

### Available exports

* Cleaned CSV
* Cleaned Excel
* EDA report in JSON
* Model results in JSON
* Complete ZIP bundle

The complete export bundle can contain:

```text
datasense_export_bundle.zip
│
├── datasense_cleaned.csv
├── datasense_cleaned.xlsx
├── datasense_eda_report.json
└── datasense_model_results.json
```

---

# 🏗️ Application Architecture

```text
                         DataSense
                            │
                            ▼
                    ┌───────────────┐
                    │ Dataset Upload│
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Data Quality  │
                    └───────┬───────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        Duplicates      Missing Values   Datatypes
              │             │             │
              └─────────────┼─────────────┘
                            ▼
                    ┌───────────────┐
                    │    Cleaning   │
                    └───────┬───────┘
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
             Outlier Handling    Imputation
                  │                   │
                  └─────────┬─────────┘
                            ▼
                    ┌───────────────┐
                    │      EDA      │
                    └───────┬───────┘
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
        Statistics     Correlation     Insights
             │              │              │
             └──────────────┼──────────────┘
                            ▼
                    ┌───────────────┐
                    │ ML Modeling   │
                    └───────┬───────┘
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
          Classification          Regression
                 │                     │
                 └──────────┬──────────┘
                            ▼
                    Model Comparison
                            │
                            ▼
                    ┌───────────────┐
                    │    Reports    │
                    └───────────────┘
```

---

# 📁 Project Structure

```text
DataSense/
│
├── Home.py
├── requirements.txt
├── README.md
│
├── modules/
│   ├── __init__.py
│   ├── datatype.py
│   ├── duplicates.py
│   ├── eda_engine.py
│   ├── insight_engine.py
│   ├── loader.py
│   ├── missing_values.py
│   ├── ml_engine.py
│   ├── outliers.py
│   ├── preprocessing.py
│   └── profiler.py
│
├── pages/
│   ├── __init__.py
│   ├── 1_Upload.py
│   ├── 2_Data_Quality.py
│   ├── 3_Cleaning.py
│   ├── 4_EDA.py
│   ├── 5_ML_Modeling.py
│   └── 6_Reports.py
│
└── utils/
    ├── __init__.py
    ├── constants.py
    └── helpers.py
```

---

# 🛠️ Tech Stack

| Technology       | Purpose                                |
| ---------------- | -------------------------------------- |
| **Python**       | Core programming language              |
| **Streamlit**    | Interactive web application            |
| **Pandas**       | Data manipulation and preprocessing    |
| **NumPy**        | Numerical operations                   |
| **Scikit-learn** | ML preprocessing, pipelines and models |
| **Matplotlib**   | Data visualization                     |
| **Seaborn**      | Statistical visualization support      |
| **Plotly**       | Interactive visualization support      |
| **OpenPyXL**     | Excel file processing                  |

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/Anshul-Rajpoot/DataSense.git
cd DataSense
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run the application

```bash
streamlit run Home.py
```

The application will open in your browser.

---

# 🔄 Typical Workflow

```text
1. Upload Dataset
        ↓
2. Inspect Data Quality
        ↓
3. Review Duplicates
        ↓
4. Analyze Missing Values
        ↓
5. Choose Cleaning Strategies
        ↓
6. Detect & Handle Outliers
        ↓
7. Validate Data Types
        ↓
8. Perform EDA
        ↓
9. Generate Automated Insights
        ↓
10. Select ML Target
        ↓
11. Train & Compare Models
        ↓
12. Export Results
```

---

# 🧪 Example Use Case

Suppose a user uploads:

```text
customer_data.xlsx
```

DataSense might identify:

```text
Rows:              10,542
Columns:               14
Missing cells:        437
Duplicate rows:       128
Numerical columns:      8
Categorical columns:    6
```

It can then identify:

```text
Salary
├── 86 missing values
├── High skewness
└── Recommended: Median imputation

City
├── 42 missing values
└── Recommended: Mode imputation

Age
├── Stored as object
└── Recommended: Convert to numeric

Experience
└── Potential outliers detected
```

After cleaning, the user can continue directly to EDA and machine learning.

---

# 🧠 Key Design Decisions

### Original vs Working Dataset

DataSense maintains two versions of the dataframe:

```text
Original Dataset
       │
       └──────────────► Never modified
       
Working Dataset
       │
       ├── Duplicate removal
       ├── Missing-value handling
       ├── Outlier handling
       └── Datatype conversion
```

This allows users to experiment with preprocessing while retaining the original dataset.

### Modular Architecture

Data processing logic is separated from the Streamlit interface.

For example:

```text
modules/
    missing_values.py
    outliers.py
    datatype.py
    ml_engine.py
```

This makes the project easier to maintain, test, and extend.

### ML Pipeline

The modeling engine uses:

```text
ColumnTransformer
        +
Pipeline
        +
Imputation
        +
Encoding
        +
Scaling
        +
Model
```

This creates a reproducible preprocessing and modeling workflow.

---

# 🔮 Future Improvements

The current architecture can be extended with:

* [ ] YData Profiling / Pandas Profiling integration
* [ ] Interactive Plotly dashboards
* [ ] Automated feature selection
* [ ] Cross-validation
* [ ] Hyperparameter tuning
* [ ] Confusion matrix visualization
* [ ] ROC and Precision-Recall curves
* [ ] SHAP-based model explainability
* [ ] Feature engineering suggestions
* [ ] Dataset versioning
* [ ] Detailed cleaning history
* [ ] Automated HTML/PDF reports
* [ ] AI-powered natural-language data insights
* [ ] Model prediction interface
* [ ] Cloud deployment
* [ ] User authentication

---

# 📌 Current Limitations

* Model selection is currently based on a fixed set of supported algorithms.
* Problem-type detection uses target cardinality/type heuristics and may require user validation for unusual datasets.
* Outlier capping currently uses IQR bounds.
* The EDA insight engine is rule-based.
* The current reporting system exports JSON rather than a fully formatted HTML/PDF report.
* YData/Pandas Profiling is not yet integrated.

---

# 🎯 Why DataSense?

Traditional data science workflows often require manually repeating the same operations:

```text
Load Data
   ↓
Check Missing Values
   ↓
Check Duplicates
   ↓
Inspect Data Types
   ↓
Handle Outliers
   ↓
EDA
   ↓
Preprocessing
   ↓
Model Training
```

DataSense brings these steps together into one interactive platform while keeping the user **in control of preprocessing decisions**.

The goal is not to hide the data science workflow, but to make it **faster, more transparent, and easier to understand**.

---

# 👨‍💻 Author

**Anshul Rajpoot**

B.Tech — Electronics & Communication Engineering
MANIT Bhopal

GitHub: [Anshul-Rajpoot](https://github.com/Anshul-Rajpoot)

---

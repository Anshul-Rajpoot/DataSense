# DataSense

DataSense is a Streamlit multipage app for automated data cleaning, exploratory data analysis, and machine learning on tabular datasets.

## Workflow

- Upload Excel, XLS, or CSV files
- Profile data quality, duplicate rows, and missing values
- Apply guided cleaning actions with original and working dataframe tracking
- Explore numeric, categorical, and correlation patterns with rule-based insights
- Compare multiple ML models for classification or regression
- Export cleaned data and report artifacts

## Pages

- Home
- Upload
- Data Quality
- Cleaning
- EDA
- ML Modeling
- Reports

## Run

```bash
streamlit run Home.py
```

## Notes

- The ML page uses `ColumnTransformer` for numeric and categorical preprocessing.

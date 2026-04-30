# AI + Data Science + Analysis Project

This project is a complete starter template that combines:

- **Data Science**: synthetic business dataset generation + feature engineering
- **AI / Machine Learning**: regression, classification, and clustering models
- **Analysis**: automated KPI summary, visualizations, and Markdown reporting

## Project Structure

`ai_data_analysis_project/`

- `src/main.py` - runs the full pipeline
- `src/data_pipeline.py` - data generation and preprocessing
- `src/train_ai_model.py` - ML training and evaluation
- `src/analysis_report.py` - KPI/plot/report generation
- `data/` - generated CSV dataset
- `reports/` - generated analysis outputs
- `notebooks/` - optional notebook workspace

## Quick Start

1. Create and activate a virtual environment (recommended)
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the full project:

```bash
python src/main.py
```

## Outputs

After running, you will get:

- `data/business_data.csv`
- `reports/model_metrics.json`
- `reports/analysis_report.md`
- `reports/figures/*.png`

## Example Use Cases

- Evaluate revenue drivers
- Predict customer churn risk
- Segment customers into behavior-based clusters
- Build an auto-generated management report

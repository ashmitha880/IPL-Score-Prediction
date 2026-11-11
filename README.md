
# IPL First Innings Score Prediction

A Machine Learning project that predicts the final score of an IPL first innings using match information and ball-by-ball data.

## Features

- Predicts the final first-innings score.
- Uses current runs, wickets, overs and recent batting performance.
- Incorporates venue information.
- Uses different trained models for different innings stages.
- Interactive Streamlit web application.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Jupyter Notebook

## Project Structure

```text
IPL_Score_Prediction/
├── app.py
├── models/
│   └── ipl_score_prediction.pkl
├── notebooks/
│   └── IPL_Score_Prediction.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## Run Locally

1. Clone this repository.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```

## Dataset

Ball-by-ball IPL data sourced from Cricsheet.

https://cricsheet.org/

The raw dataset is not included in this repository.

## Model

The project uses stage-specific machine learning models to estimate the final first-innings score from the current match situation.

## Author

Ashmitha Reddy


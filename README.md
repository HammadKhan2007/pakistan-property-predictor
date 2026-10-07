# Pakistan Property Price Predictor 🏡

Hey everyone! This is a data science project I've been working on to solve a very specific, real-world Pakistani problem: **predicting property prices accurately when the data is incredibly messy.**

If you've ever looked at Zameen.com or Graana, you know the drill. One person writes "5 Marla", the other writes "5Marla", someone else lists a 5 Marla house for 50 Crores (obviously a typo), and locations are a nightmare ("DHA Ph 5" vs "D.H.A Phase V"). 

I built this end-to-end ML pipeline to clean all that up, engineer some solid features (like calculating real-world distances to the city center using the Haversine formula), and predict prices using XGBoost and Quantile Regression.

## 🧠 The "Aha!" Moments & Challenges

While building this, I ran into a few classic data science traps that taught me a lot:

1. **The Data Leakage Trap:** I initially calculated the `society_avg_price` before splitting my train and test data. My R² shot up to 95%! I felt like a genius until I realized I was leaking the target variable into the test set. I had to rewrite the pipeline to compute the average *only* on the training fold. Real-world R² settled at ~79%, which is much more honest.
2. **Log Transform saved my model:** Property prices in Pakistan are ridiculously right-skewed. Predicting on raw PKR meant the model was just obsessing over 50-Crore mansions and ignoring the 1-Crore flats. Predicting `log1p(price)` fixed the residuals beautifully.
3. **Point Estimates are Useless:** Telling someone a house is exactly "1.24 Crores" is misleading. Real estate is a negotiation. I used `GradientBoostingRegressor` with a quantile loss function (alpha 0.1 and 0.9) to generate a realistic **price range** instead (e.g., 1.10 Cr - 1.40 Cr). 

## 📊 Quick Results

Trained on ~1.2 Lakh cleaned listings (2018-2019 data).

- **Best Model:** XGBoost
- **Median Absolute Error:** ~ PKR 13.5 Lakhs (Around 21% MAPE)
- **80% Prediction Interval:** Caught the actual price within the bounds ~79% of the time. 

## 🛠 What's in the repo?

- `data_pipeline.py`: My custom cleaning logic (regex for unit conversions, `difflib` for fuzzy matching locations).
- `train_model.py`: The training pipeline. I compare Baseline -> Ridge -> Random Forest -> XGBoost.
- `eda.py`: Generates all the charts (Saved in `reports/`).
- `app.py`: A Streamlit dashboard to play with the model interactively.

## 🚀 How to run it locally

```bash
pip install -r requirements.txt

# Want to see the charts?
python eda.py

# Want to train the model yourself? (Make sure the Kaggle CSV is in data/)
python train_model.py

# Run the UI
python -m streamlit run app.py
```

## 📈 Underpriced Properties Finder

As a side-quest, I added a script that compares the model's predicted "intrinsic value" of a house vs its asking price. If the model thinks a house is worth 30% more than what the seller is asking, it dumps it into `data/underpriced_opportunities.csv`. Real estate investors, you're welcome! 😉

---
*Feel free to fork this, open issues, or reach out if you want to chat about ML!*

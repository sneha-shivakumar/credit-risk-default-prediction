# Credit Card Default Risk Prediction

Predicting which credit card customers are likely to default next month, and grouping them into risk bands a credit risk team could actually use.

## Overview

A credit risk team needs to know ahead of time which customers are likely to miss payments, so they can adjust credit limits, flag accounts for review, or prioritise collections. This project builds two models for that and turns the output into a simple risk segmentation.

## Data

[UCI "Default of Credit Card Clients" dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) — 30,000 credit card customers in Taiwan (2005), with credit limit, demographics, 6 months of repayment history, and whether they defaulted the following month.

22.1% of customers defaulted (6,636 of 30,000).

`EDUCATION` and `MARRIAGE` both had category codes not covered in the dataset's documentation (0, 5, 6 for education; 0 for marriage). These were grouped into "other" instead of dropped, to avoid losing around 1,300 records.

## Models

### Logistic Regression
Simple, explainable baseline — every coefficient has a clear direction and size, which matters if this had to be justified to a regulator or auditor.

67.95% accuracy, 0.708 ROC-AUC.

![Logistic Regression Confusion Matrix](images/logreg_confusion_matrix.png)

### Random Forest
77.83% accuracy, 0.775 ROC-AUC. A clear improvement over logistic regression, and the model used for the risk bands below.

![Random Forest Confusion Matrix](images/rf_confusion_matrix.png)
![ROC Comparison](images/roc_comparison.png)

The strongest predictor isn't credit limit or demographics — it's recent repayment history. `PAY_0` (most recent month's repayment status) alone accounts for about 29% of the model's decisions. How someone paid last month predicts next month better than anything else in the data.

![Feature Importance](images/feature_importance.png)

## Risk bands

A model score on its own isn't something a risk manager can act on directly, so every customer was scored and split into three bands:

| Risk band | Customers | Actual default rate |
|---|---|---|
| Low | 1,109 | 2.0% |
| Medium | 20,923 | 11.4% |
| High | 7,968 | 53.1% |

![Risk Segments](images/risk_segments.png)

The "High" band is 27% of customers but over half of them default — more than 25x the rate of the "Low" band. A risk team could use this to focus review on roughly 8,000 accounts instead of treating all 30,000 the same way.

## Takeaway

The 77.8% accuracy figure undersells the model. The more useful output is the separation between risk bands: knowing that one group defaults 53% of the time and another almost never does is something a risk team could actually build a process around.

## Next steps

- Try a gradient-boosted model (XGBoost/LightGBM) to see if AUC improves further
- Build an interactive dashboard (Power BI or Tableau) so risk bands can be filtered without touching code
- Check the model's fairness across demographic groups before considering real deployment

## Repo structure

```
credit-risk-default-prediction/
├── credit_risk_model.py          # cleaning, both models, risk segmentation, charts
├── data/
│   ├── UCI_Credit_Card.csv
│   └── customer_risk_scores.csv   # every customer scored + risk band
├── images/
└── README.md
```

## Running it

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python credit_risk_model.py
```


# House Prices Prediction - Kaggle Competition

This project is a submission for the Kaggle competition **Housing Prices Competition for Kaggle Learn Users**, where participants are tasked with predicting the sale price of homes based on a wide variety of features. This competition uses the Ames Housing dataset, which contains 79 features describing different aspects of residential homes in Ames, Iowa.

## Project Overview

The goal of this competition is to predict the final price of each house in the test set using advanced regression techniques. Submissions are evaluated based on the Root Mean Squared Error (RMSE) between the logarithm of the predicted and actual sale prices.

### Notable Achievement

In this competition, I ranked in the **top 1%**, securing a position within the **top 1500** out of:

- **185,867 Entrants**
- **109,368 Participants**
- **109,328 Teams**
- **322,476 Submissions**

This highlights the performance and accuracy of the model I built using advanced techniques like XGBoost and feature engineering.

### Dataset

The dataset consists of two main files:

1. **train.csv**: Contains 1460 rows with 80 features (79 explanatory variables and the target variable `SalePrice`).
2. **test.csv**: Contains 1459 rows with the same 79 explanatory features, excluding the target variable `SalePrice`.

The data includes various aspects of the homes, such as the number of rooms, square footage, basement conditions, and other features that influence housing prices.

### Evaluation Metric

The competition uses **Root Mean Squared Error (RMSE)** on the logarithm of the SalePrice as the evaluation metric. Specifically:

\[
RMSE = \sqrt{rac{1}{n} \sum_{i=1}^{n} (\log(	ext{SalePrice}_{	ext{predicted}}) - \log(	ext{SalePrice}_{	ext{actual}}))^2}
\]

This means that errors in predicting both expensive and inexpensive homes are treated equally.

### Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/kaggle-housing-prices.git
   cd kaggle-housing-prices
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from Kaggle:
   - Go to the [competition page](https://www.kaggle.com/competitions/home-data-for-ml-course/data) and download the `train.csv` and `test.csv` files.
   - Place the files in a folder named `input`.

4. Run the model:
   ```bash
   python kaggle_competition_script.py
   ```

   This script will generate a `submission.csv` file containing the predictions.

### Feature Engineering

In this project, we implemented basic feature engineering, including:

- **Handling missing values**: We removed rows with missing target values and dealt with missing data for categorical and numeric columns.
- **Categorical encoding**: We used one-hot encoding for categorical variables with low cardinality (i.e., categories with fewer than 10 unique values).
- **Numeric features**: All numeric columns were selected as features for the model.

### Model Description

We used **XGBoost**, a popular gradient boosting algorithm, to predict house prices. The model was configured with the following parameters:

- `n_estimators`: 1000 (The maximum number of boosting rounds)
- `learning_rate`: 0.05 (Learning rate for each boosting round)
- `max_depth`: 6 (Maximum depth of a tree)
- `min_child_weight`: 1 (Minimum sum of instance weight needed in a child)
- `subsample`: 0.8 (The fraction of samples used for fitting individual base learners)
- `colsample_bytree`: 0.8 (The fraction of features used for each tree)
- `early_stopping_rounds`: 10 (Stops training when validation score stops improving)

The model was trained on the full training data, with one-hot encoding applied to both training and test datasets.

### Output

The script generates a file named `submission.csv`, which contains the predicted sale prices for the test dataset. The format of the submission file is:

```
Id,SalePrice
1461,169000.1
1462,187724.1233
1463,175221.0
...
```

### Results

- **Top 1% of all participants**
- Ranked **within the top 1500** out of **185,867 entrants** and **109,368 participants**.
- The model was evaluated on Kaggle’s public leaderboard using the RMSE metric, and we achieved notable performance with fine-tuned hyperparameters.

### Acknowledgments

- **Dataset**: The Ames Housing dataset was compiled by Dean De Cock for use in data science education.
- **Competition**: This project is part of the Kaggle competition [Housing Prices Competition for Kaggle Learn Users](https://www.kaggle.com/competitions/home-data-for-ml-course).

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
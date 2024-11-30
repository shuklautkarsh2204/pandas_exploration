

---

# Behavioral Data Analysis using Machine Learning Models

This project applies multiple machine learning techniques to analyze user behavioral data. The dataset contains features like app usage, screen-on time, battery drain, and demographic information to classify user behavior and predict app usage patterns.

## Features

The project demonstrates:
1. **Logistic Regression** for classification tasks.
2. **K-Nearest Neighbors (KNN)** for classification with parameter tuning (number of neighbors).
3. **Linear Regression** for predicting app usage based on battery drain.

## Prerequisites

- Python 3.8+
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`

You can install the required libraries using:
```bash
pip install pandas numpy matplotlib scikit-learn
```

## Dataset

The dataset should be named **`dataset.csv`** and located in the project root. It must contain the following columns:
- `UserID`
- `DeviceModel`
- `OS`
- `AppUsage` (minutes/day)
- `ScreenOnTime` (minutes/day)
- `BatteryDrain` (%/day)
- `NumApps` (count)
- `DataUsage` (MB/day)
- `Age`
- `Gender`
- `BehaviorClass` (categorical)

## Implementation Details

### Logistic Regression
- Predicts user behavior class (`BehaviorClass`) based on app usage patterns.
- Outputs accuracy and confusion matrix.
- Plots:
  - Confusion matrix for Logistic Regression.

### K-Nearest Neighbors (KNN)
- Classifies user behavior and evaluates accuracy for different `K` values.
- Plots:
  - Accuracy vs. Number of Neighbors (K).

### Linear Regression
- Predicts app usage (`AppUsage`) using battery drain percentage (`BatteryDrain`).
- Evaluates the model using residual analysis.
- Plots:
  - Actual vs. Predicted values.
  - Residuals histogram.

## How to Run

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Ensure the dataset is in the same directory and named `dataset.csv`.
3. Run the script:
   ```bash
   python <script_name>.py
   ```

## Output

- **Logistic Regression**:
  - Accuracy score.
  - Confusion matrix visualization.
- **KNN**:
  - Accuracy score for `K=5`.
  - Plot of accuracy vs. neighbors.
- **Linear Regression**:
  - Scatter plot of actual vs. predicted values.
  - Histogram of residuals.

## Results

This analysis helps in:
1. **Classifying User Behavior**:
   - Understanding app usage patterns.
   - Optimizing app performance based on usage.
2. **Predicting App Usage**:
   - Estimating app engagement based on device battery usage.

## License

This project is licensed under the MIT License. Feel free to use and modify it.

## Contributing

Feel free to fork the repository, create a feature branch, and submit a pull request. Suggestions and contributions are welcome!

---


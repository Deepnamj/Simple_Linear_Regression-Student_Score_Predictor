# PROJECT 1 — STUDENT EXAM SCORE PREDICTION

## 1. Problem Definition

### Business / ML Problem
Predict a student's exam score based on the number of hours they studied.

### Problem Type
**Supervised Learning → Regression**

Why regression?

Because the target (`Exam_Score`) is a **continuous numerical value**.

Example:

`75, 82.5, 91, 63.7`

We are predicting a quantity, not a class/category.

### Feature
`Hours_Studied`

### Target
`Exam_Score`

### Model
**Simple Linear Regression**

---

# 2. What is Supervised Learning?

In supervised learning, we give the model:

- Input data → X
- Correct output → y

The model learns the relationship between them.

For this project:

`Hours Studied → Exam Score`

Example:

`5 hours → 75 marks`

`8 hours → 88 marks`

The model learns from these known examples and can then predict the score for a new student.

---

# 3. What is Regression?

Regression is a supervised learning technique used to predict a **continuous numerical value**.

Examples:

- House price → ₹50 lakh
- Salary → ₹60,000
- Temperature → 31.5°C
- Exam score → 82.4
- Sales → ₹2.5 lakh

### Regression vs Classification

**Regression:**

`Input → numerical value`

Example:

`Hours Studied → 82.5 marks`

**Classification:**

`Input → category/class`

Example:

`Hours Studied → Pass / Fail`

---

# 4. Dataset

Dataset: **Student Performance Factors**

Relevant columns:

`Hours_Studied`

`Exam_Score`

For this project, we intentionally use only these two columns.

Why?

Because the goal is to learn **Simple Linear Regression** first.

---

# 5. X and y — One of the Most Important Concepts

In ML:

**X = features/input variables**

**y = target/output variable**

For this project:

`X = Hours_Studied`

`y = Exam_Score`

So:

`Hours Studied → X → Model → y → Exam Score`

### In Python

```python
X = df[["Hours_Studied"]]
y = df["Exam_Score"]
```

### Why double brackets for X?

```python
df[["Hours_Studied"]]
```

returns a DataFrame, which is 2-dimensional.

Shape:

`(number_of_rows, number_of_features)`

For example:

`(5000, 1)`

But:

```python
df["Exam_Score"]
```

returns a Series, which is normally 1-dimensional.

Shape:

`(5000,)`

Scikit-learn expects X in a 2D feature matrix.

### Remember

```text
X → input/features → usually 2D

y → target/output → usually 1D
```

---

# 6. Where X and y Come From

### Scikit-learn dataset

Some scikit-learn datasets already provide X and y separately.

Example:

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = data.data
y = data.target
```

Here, the dataset has already separated the features and target.

### Kaggle / CSV dataset

Usually you start with one DataFrame:

```python
df = pd.read_csv("StudentPerformanceFactors.csv")
```

Then you define X and y yourself:

```python
X = df[["Hours_Studied"]]
y = df["Exam_Score"]
```

Or, when appropriate:

```python
X = df.drop("Exam_Score", axis=1)
y = df["Exam_Score"]
```

### Important

The source of the dataset does **not** determine how X and y work.

The universal idea is:

`X = inputs used for prediction`

`y = output we want to predict`

---

# 7. Why We Don't Put the Target in X

Suppose:

```text
Hours_Studied
Attendance
Exam_Score
```

If `Exam_Score` is what we want to predict, it must not be included in X.

Wrong:

```python
X = df[["Hours_Studied", "Exam_Score"]]
y = df["Exam_Score"]
```

The model would already be given the answer.

This is a form of **target leakage / data leakage**.

Correct:

```python
X = df[["Hours_Studied"]]
y = df["Exam_Score"]
```

---

# 8. Data Understanding

Before training a model, understand the dataset.

Important commands:

```python
df.head()
df.shape
df.info()
df.describe()
df.dtypes
df.isnull().sum()
df.duplicated().sum()
```

### What they tell us

`head()` → first rows

`shape` → number of rows and columns

`info()` → data types and non-null values

`describe()` → statistical summary

`dtypes` → column data types

`isnull().sum()` → missing values

`duplicated().sum()` → duplicate rows

---

# 9. Data Cleaning

Before training:

### Missing values

Check:

```python
df.isnull().sum()
```

For this project, because we only need two columns:

```python
df = df[["Hours_Studied", "Exam_Score"]].dropna()
```

This removes rows where either required value is missing.

### Duplicates

Check:

```python
df.duplicated().sum()
```

Do not automatically delete duplicates without understanding the data.

Two students can legitimately have the same study hours and score.

---

# 10. Exploratory Data Analysis — EDA

EDA means **Exploratory Data Analysis**.

The goal is to understand:

- distributions
- relationships
- missing values
- outliers
- patterns
- correlations

For this project, the most important relationship is:

**Hours Studied vs Exam Score**

---

# 11. Correlation

Correlation measures the strength and direction of a **linear relationship** between two numerical variables.

Example:

```python
correlation = df["Hours_Studied"].corr(
    df["Exam_Score"]
)

print(correlation)
```

Correlation ranges from:

`-1 to +1`

### Interpretation

`+1` → perfect positive linear relationship

`0` → no linear correlation

`-1` → perfect negative linear relationship

### Positive correlation

As X increases, y tends to increase.

Example:

`Hours Studied ↑ → Exam Score ↑`

### Negative correlation

As X increases, y tends to decrease.

### Important

**No linear correlation does NOT necessarily mean no relationship.**

There could be a non-linear relationship.

---

# 12. Scatter Plot

For this project:

```python
plt.scatter(
    df["Hours_Studied"],
    df["Exam_Score"]
)
```

This helps us visually see whether a linear relationship appears reasonable.

If the points roughly follow a straight-line pattern, Linear Regression may be appropriate.

---

# 13. Simple Linear Regression

Simple Linear Regression predicts y using **one independent variable X**.

The equation is:

`ŷ = b₀ + b₁x`

Where:

- `ŷ` = predicted value
- `b₀` = intercept
- `b₁` = coefficient/slope
- `x` = input feature

For this project:

`Exam Score = b₀ + b₁ × Hours Studied`

---

# 14. Intercept

The intercept is `b₀`.

It represents the predicted value of y when:

`x = 0`

For this project:

> The intercept is the model's predicted exam score when Hours_Studied is zero.

However, interpretation should consider whether `x = 0` is meaningful in the dataset.

---

# 15. Coefficient / Slope

The coefficient is `b₁`.

It tells us how much the predicted target changes for a **one-unit increase in X**.

Example:

If:

`b₁ = 2.5`

then:

> For every additional hour studied, the model predicts an average increase of approximately 2.5 exam-score points.

This is a model-based association, not proof of causation.

---

# 16. How Does Linear Regression Find the Best Line?

The model tries to find the line that produces the smallest overall prediction error.

For each observation:

`Residual = Actual value - Predicted value`

The method used by ordinary Linear Regression is based on **Least Squares**.

It minimizes the **sum of squared residuals**:

`Σ(yᵢ - ŷᵢ)²`

Why square the errors?

Because:

- positive and negative errors don't cancel
- large errors receive greater penalty

The resulting line is called the **best-fit line / least-squares regression line**.

---

# 17. Regression Line

The regression line represents the relationship learned by the model.

Visually:

```text
Exam Score
   ↑
   |                 •
   |             •
   |          •
   |       •
   |    •
   | •
   +----------------------→ Hours Studied
```

The actual observations are scattered around the line.

The line represents the model's predicted values.

---

# 18. Train/Test Split

We should not train and evaluate on exactly the same data.

Why?

Because we want to know whether the model can generalize to unseen data.

Typical split:

`80% → training`

`20% → testing`

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Training data

Used to learn the relationship.

### Test data

Used to evaluate the trained model on unseen examples.

### random_state

Makes the split reproducible.

If we use:

```python
random_state=42
```

we can obtain the same split again.

---

# 19. Training the Model

Import:

```python
from sklearn.linear_model import LinearRegression
```

Create model:

```python
model = LinearRegression()
```

Train:

```python
model.fit(X_train, y_train)
```

### What does fit() do?

`fit()` is where the model **learns the parameters** from the training data.

For Linear Regression, it learns:

- coefficient(s)
- intercept

It finds the parameters that minimize the least-squares error.

---

# 20. Prediction

After training:

```python
y_pred = model.predict(X_test)
```

The model takes X values and produces predicted y values.

```text
X_test
  ↓
trained model
  ↓
y_pred
```

---

# 21. Coefficient and Intercept in Python

```python
print(model.coef_)
print(model.intercept_)
```

For one feature:

```python
coefficient = model.coef_[0]
intercept = model.intercept_
```

Then:

```python
print(
    f"Exam Score = {intercept:.2f} + "
    f"({coefficient:.2f} × Hours Studied)"
)
```

---

# 22. Evaluation Metrics

We need to measure how well predictions match actual values.

Main regression metrics:

### MAE — Mean Absolute Error

`MAE = average(|actual - predicted|)`

It tells us the average absolute prediction error.

Example:

`MAE = 3.2`

means the predictions are off by about 3.2 score points on average.

### MSE — Mean Squared Error

`MSE = average((actual - predicted)²)`

Because errors are squared, large errors receive greater weight.

### RMSE — Root Mean Squared Error

`RMSE = √MSE`

RMSE is useful because it is in the **same unit as the target**.

If Exam Score is measured in marks:

`RMSE = 4.5`

means the typical error scale is around 4.5 marks.

### R² — Coefficient of Determination

R² measures how much of the variation in the target is explained by the model relative to a mean-prediction baseline.

A commonly used interpretation:

`R² = 0` → no improvement over predicting the mean, in the standard formulation

`R² = 1` → perfect predictions

R² can also be negative on unseen test data.

---

# 23. Which Metric Should I Focus On?

For this project, remember:

**MAE** → easy-to-understand average error

**MSE** → strongly penalizes large errors

**RMSE** → error in the same units as target

**R²** → goodness of fit / explained variation relative to baseline

Do not say:

> "R² tells us the accuracy."

R² is **not classification accuracy**.

---

# 24. The Three Important ML Graphs

## Graph 1 — Actual Data + Regression Line

Shows:

- actual observations
- relationship between X and y
- fitted regression line

### Importance

It helps us visually determine whether a linear model is reasonable.
The scatter points represent the actual observations from the test dataset. Since this is supervised learning, the actual exam scores are already known. The regression line represents the linear relationship learned by the model from the training data. We use this learned line to predict exam scores for the test inputs and compare those predictions with the actual test scores
---

## Graph 2 — Actual vs Predicted

X-axis:

`Actual values`

Y-axis:

`Predicted values`

Ideal predictions lie close to:

`y = x`

### Importance

It shows how close the predictions are to the actual values.

The closer the points are to the diagonal reference line, the better the predictions generally are.

---

## Graph 3 — Residual Plot

Residual:

`Residual = Actual - Predicted`

Plot:

`Predicted values vs Residuals`

### What we want

Residuals should generally:

- be centered around zero
- look randomly scattered
- show no obvious pattern

### Why?

A clear pattern can indicate that the model is missing some structure in the data.

Examples:

- curve → relationship may not be linear
- funnel shape → changing variance / heteroscedasticity
- clusters → missing groups/features

This graph is especially important for understanding **whether the Linear Regression assumptions are reasonable**.

---

# 25. Linear Regression Assumptions

You should know these even though not all are directly checked in the basic project.

### 1. Linearity

The relationship between predictors and target should be approximately linear.

### 2. Independence

Observations/errors should generally be independent.

### 3. Homoscedasticity

The variance of residuals should be reasonably constant across predicted values.

### 4. Normality of residuals

Residuals are ideally approximately normally distributed for certain statistical inference procedures.

Important:

> Normal residuals are not a universal requirement for making predictions with LinearRegression in scikit-learn.

### 5. Low multicollinearity

For multiple Linear Regression, predictors should not be excessively correlated with each other.

This becomes especially important in Project 2 and Project 5.

---

# 26. Outliers

An outlier is an observation that is unusually far from the general pattern.

Linear Regression can be sensitive to outliers, particularly because squared errors are used in least squares.

Do not automatically remove an outlier.

Ask:

- Is it a data-entry error?
- Is it a legitimate observation?
- Does it strongly influence the model?
- Does the business context justify keeping it?

---

# 27. Prediction for a New Student

After training:

```python
new_student = pd.DataFrame({
    "Hours_Studied": [8.5]
})

prediction = model.predict(new_student)
```

Then:

```python
print(prediction[0])
```

The workflow is:

```text
New input
   ↓
Hours Studied = 8.5
   ↓
Trained model
   ↓
Predicted Exam Score
```

---

# 28. Complete ML Workflow

This is the workflow you should remember:

```text
Problem Definition
       ↓
Data Collection
       ↓
Data Understanding
       ↓
EDA
       ↓
Data Cleaning
       ↓
Feature / Target Selection
       ↓
Train/Test Split
       ↓
Model Training
       ↓
Prediction
       ↓
Evaluation
       ↓
Error / Residual Analysis
       ↓
Interpretation
       ↓
New Predictions
```

---

# 29. Important Things to Remember

### Must know

- `X` = features/input
- `y` = target/output
- Simple Linear Regression → one feature
- Multiple Linear Regression → multiple features
- `fit()` → learns from training data
- `predict()` → produces predictions
- coefficient → change in predicted y per one-unit change in x
- intercept → predicted y when x = 0
- residual → actual − predicted
- MAE → average absolute error
- MSE → squared-error metric
- RMSE → square root of MSE
- R² → explains variation relative to a baseline
- train data → used for learning
- test data → unseen evaluation data

### Very important

**Correlation ≠ causation**

If Hours Studied and Exam Score are positively correlated, we cannot conclude:

> "Studying one extra hour definitely causes a certain increase in marks."

The model identifies a statistical relationship in the data.

---

# 30. Things You Should Be Able to Explain in an Interview

### Q1. Why is this a regression problem?

Because the target, Exam Score, is a continuous numerical value.

### Q2. What is X?

X contains the input feature(s) used by the model to make predictions.

### Q3. What is y?

y is the target value that the model learns to predict.

### Q4. Why do you use double brackets for X?

Because scikit-learn expects the feature matrix to be 2-dimensional.

### Q5. What does `fit()` do?

It learns the model parameters from the training data.

### Q6. What does the coefficient mean?

It represents the expected change in predicted exam score for a one-unit increase in Hours Studied, according to the fitted model.

### Q7. Why split the data?

To evaluate whether the model generalizes to unseen data.

### Q8. Why not train and test on the same data?

Because that does not provide a reliable estimate of performance on unseen data.

### Q9. What is a residual?

`Actual − Predicted`

### Q10. Why do we look at residuals?

To understand prediction errors and identify patterns that may indicate model problems.

### Q11. MAE vs RMSE?

MAE gives the average absolute error.

RMSE penalizes larger errors more strongly and is expressed in the target's units.

### Q12. What does R² mean?

It measures how much variation in the target is explained by the model relative to a baseline based on the target mean.

---

# 31. Project Findings

### Objective

Predict Exam Score from Hours Studied using Simple Linear Regression.

### Dataset

Student Performance Factors.

### Feature

`Hours_Studied`

### Target

`Exam_Score`

### Model

Simple Linear Regression.

### Main Finding

The model learns a linear relationship between study hours and exam score. The coefficient indicates the estimated change in predicted exam score associated with one additional hour of study.

### Evaluation

The model is evaluated using:

- MAE
- MSE
- RMSE
- R²

### Visual Findings

1. **Scatter plot + regression line** → shows the relationship and fitted linear trend.
2. **Actual vs predicted** → shows how close predictions are to actual values.
3. **Residual plot** → helps identify systematic prediction errors and possible violations of linear-model assumptions.

### Limitations

Exam performance can also depend on:

- attendance
- previous academic performance
- sleep
- tutoring
- parental involvement
- learning resources
- motivation
- other factors

Therefore, this model should not be interpreted as proving that study hours alone cause exam-score changes.

### Conclusion

This project demonstrates the complete workflow of Simple Linear Regression from data preparation and X/y separation through model training, prediction, evaluation, visualization and interpretation.

---

# 32. The One Mental Model to Keep

```text
DATA
 ↓
What am I predicting?
 ↓
TARGET = y
 ↓
What information can help predict it?
 ↓
FEATURES = X
 ↓
Split into training and testing
 ↓
Train Linear Regression
 ↓
Learn coefficient + intercept
 ↓
Make predictions
 ↓
Compare actual vs predicted
 ↓
Evaluate errors
 ↓
Check residuals
 ↓
Interpret the model
```

**This is the foundation for Projects 2–5.** Once you genuinely understand this project, Multiple Linear Regression, Polynomial Regression, Ridge, Lasso, Pipelines and the production project will all build on the same basic `X → model → y` idea.
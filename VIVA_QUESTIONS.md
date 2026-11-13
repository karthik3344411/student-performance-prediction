# Viva Questions & Answers – Student Performance Prediction

Short answers you can remember. Numbers come from this project's real results.

---

## A. Project overview

**1. Explain your project in one minute.**
I predict a student's final Maths grade (G3, 0–20) using 30 background features from the UCI Student Performance dataset (395 students), without the earlier grades G1 and G2. I cleaned the data, did EDA with statistical tests, compared 5 regression models with cross-validation, and tuned the best one (Random Forest) with GridSearchCV. The final model has test MAE 3.04, RMSE 3.76 and R² 0.31, beating the baseline (MAE 3.65). It is saved with joblib and used in a Streamlit app.

**2. Is this classification or regression? Why?**
Regression, because the target G3 is a number from 0 to 20. I also did an optional pass/fail classification (G3 ≥ 10) as an extension.

**3. Why did you not use G1 and G2?**
They are earlier grades of the same course and correlate 0.80 and 0.90 with G3. With them the model would just copy the last grade, which is easy but not useful early in the year. Without them we learn which background factors matter.

**4. Who would use this, and what is the main limitation?**
Teachers and counsellors, as an early warning signal. Limitation: the typical error is about 3 points, the data is small (395 students from 2 Portuguese schools), and it is only an academic demonstration.

## B. Python & Pandas

**5. How did you load the data? What was special?**
`pd.read_csv("data/raw/student-mat.csv", sep=";")`. The file uses a semicolon, not a comma, as the separator.

**6. How did you check missing values and duplicates?**
`df.isna().sum()` for missing values and `df.duplicated().sum()` for duplicates. Both were 0.

**7. What does `df.describe()` show?**
Count, mean, standard deviation, min, quartiles (25%, 50% = median, 75%) and max for every numeric column.

## C. Data preprocessing

**8. Why do we encode categorical variables? Which method did you use?**
Models only understand numbers. I used **One-Hot Encoding**: each category becomes its own 0/1 column (e.g. `sex_F`, `sex_M`). It does not invent a false order like label encoding would.

**9. What is scaling and why use StandardScaler?**
StandardScaler converts each value to (value − mean) / std, so features are on the same scale (absences 0–75 vs. 1–5 scales). It matters for Linear/Ridge regression; tree models don't need it but it does no harm.

**10. What is data leakage and how did you prevent it?**
Leakage is when information from the test data gets into training, which makes results look better than they really are. I put the scaler and encoder inside a scikit-learn **Pipeline**, so they are fitted only on training data (and on training folds during CV). The test set is never used for any decision.

**11. What is a ColumnTransformer?**
It applies different preprocessing to different columns: StandardScaler to the 13 numeric columns and OneHotEncoder to the 17 categorical columns, in one step.

**12. How did you handle outliers?**
I found them with the IQR rule (e.g. 15 students with more than 20 absences, maximum 75). I **kept** them because they are real, possible values, the dataset is small, and Random Forest is robust to outliers.

**13. Why an 80/20 train-test split with `random_state=42`?**
80% (316 students) to learn, and 20% (79) to test on unseen data. The fixed seed makes the split, and therefore the results, reproducible.

## D. EDA & statistics

**14. What was the most important EDA finding?**
Past failures: the mean grade falls from 11.3 (0 failures) to 5.7 (3 or more). Kruskal–Wallis p < 0.001, Spearman ρ = −0.36.

**15. What surprising pattern did you find?**
All 38 students with G3 = 0 have 0 absences. They probably dropped out or missed the exam. Without them, absences correlate negatively with grades (ρ = −0.24).

**16. Which statistical tests did you use?**
Welch t-test (gender, p = 0.04), one-way ANOVA and Kruskal–Wallis (study time, p = 0.16 / 0.056, not significant), Kruskal–Wallis (failures, p < 0.001), chi-square (higher education vs pass, p = 0.004), Spearman correlation, and a 95% confidence interval for the mean grade (9.96–10.87).

**17. What does a p-value mean?**
The probability of seeing a difference at least this big if there were really no difference. If p < 0.05, we call the result statistically significant.

**18. Does correlation mean causation?**
No. For example, students who get extra school support have lower grades, most likely because weak students are the ones who receive support, not because support lowers grades.

## E. Regression models

**19. Which models did you train and why?**
Dummy Regressor (a baseline that always predicts the mean), Linear Regression (simple), Ridge (linear with regularisation), Decision Tree (non-linear) and Random Forest (an ensemble of trees).

**20. Why do we need a baseline model?**
It tells us whether the model learned anything. If a model can't beat "always predict the average" (MAE 3.65), it is useless.

**21. What is Linear Regression?**
It fits y = b0 + b1·x1 + … + bn·xn and chooses the coefficients that minimise the squared errors.

**22. What is Ridge Regression?**
Linear regression plus an L2 penalty (alpha × sum of squared coefficients). It shrinks large coefficients and helps when features are correlated (e.g. Medu–Fedu 0.62).

**23. How does a Random Forest work?**
It builds many decision trees, each on a random bootstrap sample of the students and a random subset of the features, and averages their predictions. Averaging reduces overfitting.

## F. Evaluation metrics

**24. Explain MAE, MSE, RMSE and R².**
- MAE = mean of |actual − predicted|; our 3.04 means "off by about 3 points on average".
- MSE = mean of the squared errors (14.1); big errors count much more.
- RMSE = √MSE (3.76), in grade points.
- R² = share of the variance explained (0.31 = 31%); 0 is as good as predicting the mean, and negative is worse than that.

**25. Why didn't you choose the model by R² only?**
MAE and RMSE are in grade points, so teachers can understand them, and they show the size of the errors. I ranked models by CV RMSE and CV MAE together.

**26. What do the residual and actual-vs-predicted plots show?**
Residuals are centred near 0 (no big bias), but predictions are squeezed towards 8–14. The biggest errors are for students with G3 = 0 (MAE 8.06, against 2.70 for the others).

## G. Overfitting, cross-validation, tuning

**27. What is overfitting? Give an example from your project.**
The model memorises the training data and fails on new data. My unlimited Decision Tree had a training RMSE of 0.00 but a CV RMSE of 5.53, worse than the baseline.

**28. What is underfitting?**
The model is too simple to capture the pattern, so it is bad on both training and test data. The Dummy Regressor is an extreme example.

**29. What is k-fold cross-validation and why did you use it?**
The training set is split into 5 parts; the model trains on 4 parts and validates on the 5th, repeated 5 times, and the scores are averaged. It gives a more reliable estimate than one split. In my project Linear Regression looked better than the baseline on the test set but worse in CV, so a single split could have misled me.

**30. What is GridSearchCV? What did you tune?**
It tries every combination of the given hyperparameters with cross-validation and keeps the best. For Random Forest I tuned n_estimators (200, 500), max_depth (None, 6, 10), min_samples_leaf (1, 3, 5) and max_features (1.0, 0.5, sqrt): 54 combinations × 5 folds. Best: 200 trees, max_features 0.5. CV RMSE improved only slightly (3.93 → 3.90).

**31. What is the difference between a parameter and a hyperparameter?**
Parameters are learned from the data (e.g. regression coefficients, tree split points). Hyperparameters are set before training (e.g. number of trees, max depth).

## H. Project-specific / deployment

**32. Which features were most important?**
By permutation importance: failures, then absences, then sex, goout, schoolsup, Medu and age.

**33. How is the model saved and used in the app?**
`joblib.dump(pipeline, "models/best_model.joblib")`. The pipeline includes preprocessing, so the app only builds a one-row DataFrame with the same 30 column names and calls `model.predict()`.

**34. How does the app handle errors?**
It shows a clear error if the model file is missing, validates every input range and category, and limits predictions to 0–20. If a value ever had to be limited, it shows the raw value too, so nothing is hidden.

**35. How would you improve the project?**
Handle G3 = 0 (dropouts) with a separate classifier, add the Portuguese dataset for more data, try gradient boosting with SHAP explanations, and check fairness across gender and address groups.

**36. In the pass/fail extension, why is accuracy misleading?**
67% of students pass, so "always predict pass" gets 67% accuracy but finds zero failing students. Recall and F1 for the "fail" class are more meaningful (Logistic Regression: recall 0.46, F1 0.43).

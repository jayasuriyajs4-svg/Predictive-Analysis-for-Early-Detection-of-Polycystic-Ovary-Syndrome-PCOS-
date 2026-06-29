
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb


import pandas as pd
# ------------------------------
df = pd.read_csv("/PCOS_extended_dataset.csv")
df.head()

print("Columns:", df.columns)

# Drop duplicates
df = df.drop_duplicates()

# Encode categorical variables
label_enc = LabelEncoder()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str).str.strip()
        df[col] = label_enc.fit_transform(df[col])

# Fill missing values with median
df = df.fillna(df.median(numeric_only=True))

# ------------------------------
# 3. Feature Engineering
# ------------------------------
# Remove low variance columns (if any)
df = df.loc[:, df.var(numeric_only=True) > 0.01]



# Correlation check
corr = df.corr(numeric_only=True)
plt.figure(figsize=(10,6))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.show()


# Features & target
X = df.drop(columns=["PCOS (Y/N)"], errors="ignore")
y = df["PCOS (Y/N)"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)



# 4. Logistic Regression (Tuned)
# ------------------------------
log_reg = LogisticRegression(max_iter=500)
param_grid_lr = {'C':[0.01,0.1,1,10]}
grid_lr = GridSearchCV(log_reg, param_grid_lr, cv=5)
grid_lr.fit(X_train, y_train)
y_pred_lr = grid_lr.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr) * 100
print("Best Logistic Regression Params:", grid_lr.best_params_)
print("Accuracy: {:.2f}%".format(acc_lr))

plt.figure(figsize=(5,4))
plt.bar(["Logistic Regression"], [acc_lr], color="blue")
plt.ylabel("Accuracy (%)")
plt.title("Logistic Regression Accuracy")
plt.show()


# 5. Random Forest (Tuned)
# ------------------------------
rf = RandomForestClassifier(random_state=42)
param_grid_rf = {'n_estimators':[100,200],
                 'max_depth':[5,10,None]}
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5)
grid_rf.fit(X_train, y_train)
y_pred_rf = grid_rf.predict(X_test)

print("Best RF Params:", grid_rf.best_params_)
print(classification_report(y_test, y_pred_rf))

feat_imp = pd.Series(grid_rf.best_estimator_.feature_importances_, index=X.columns)
plt.figure(figsize=(8,6))
feat_imp.sort_values(ascending=False).head(15).plot(kind="barh")
plt.title("Random Forest Feature Importance (Top 15)")
plt.show()



#6. XGBoost (Tuned)
# ------------------------------
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
param_grid_xgb = {
    'max_depth':[3,5,7],
    'learning_rate':[0.01,0.1,0.2],
    'n_estimators':[100,200]
}
grid_xgb = GridSearchCV(xgb_clf, param_grid_xgb, cv=3)
grid_xgb.fit(X_train, y_train)
y_pred_xgb = grid_xgb.predict(X_test)
y_pred_prob_xgb = grid_xgb.predict_proba(X_test)[:,1]

print("Best XGB Params:", grid_xgb.best_params_)
print(classification_report(y_test, y_pred_xgb))
plt.figure(figsize=(7,5))
sns.histplot(y_pred_prob_xgb, bins=20, kde=True, color="orange")
plt.title("XGBoost Prediction Confidence Levels")
plt.xlabel("Confidence (Probability of PCOS=1)")
plt.ylabel("Count")
plt.show()


# ------------------------------
# 7. kNN (Tuned)
# ------------------------------
knn = KNeighborsClassifier()
param_grid_knn = {'n_neighbors':[3,5,7,9,11]}
grid_knn = GridSearchCV(knn, param_grid_knn, cv=5)
grid_knn.fit(X_train, y_train)
y_pred_knn = grid_knn.predict(X_test)

print("Best kNN Params:", grid_knn.best_params_)
print(classification_report(y_test, y_pred_knn))

plt.figure(figsize=(6,4))
plt.plot([3,5,7,9,11], grid_knn.cv_results_['mean_test_score'], marker="o")
plt.title("kNN: Accuracy vs Neighborhood Size")
plt.xlabel("k (Number of Neighbors)")
plt.ylabel("CV Accuracy")
plt.show()

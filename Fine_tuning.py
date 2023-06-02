import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Load data
df = pd.read_excel("W4.1_CreditRisk.xls")

# Clean data
df = df.dropna(subset=['Gender'])
df['Credit Standing'] = df['Credit Standing'].map({'Good': 1, 'Bad': 0})

# Feature Engineering
df = pd.get_dummies(df, columns=['Checking Acct', 'Credit Hist', 'Purpose', 'Savings Acct', 'Employment', 'Gender', 'Personal Status', 'Housing', 'Job', 'Telephone', 'Foreign'])
df['Months_Age_Ratio'] = df['Months Acct (Added 1 to original Months Acct Variable)'] / df['Age subtracted 1 from original Age variable']
df['Months_Age_Difference'] = df['Months Acct (Added 1 to original Months Acct Variable)'] - df['Age subtracted 1 from original Age variable']

# Split features and target variable
X = df.drop('Credit Standing', axis=1)
y = df['Credit Standing']

# Standardize numerical columns
scaler = StandardScaler()
cols_to_scale = ['Months Acct (Added 1 to original Months Acct Variable)', 'Residence Time', 'Age subtracted 1 from original Age variable', 'Months_Age_Ratio', 'Months_Age_Difference']
X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define models and their hyperparameters
models = {
    'RandomForest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [200, 300],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [4, 5, 6, 7],
            'criterion': ['gini', 'entropy']
        }
    },
    'SVM': {
        'model': SVC(probability=True),
        'params': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001],
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
        }
    }
}

# Initialize figure for plotting ROC curves
plt.figure()

# Evaluate each model
for name, model in models.items():
    print(f"\n{name}")

    # Grid search to find best hyperparameters
    grid_search = GridSearchCV(estimator=model['model'], param_grid=model['params'], cv=10)
    grid_search.fit(X_train, y_train)

    print("Best Parameters: ", grid_search.best_params_)

    # Best model
    best_model = grid_search.best_estimator_

    # Cross validation scores
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=10)
    print('Cross-Validation Mean: ', cv_scores.mean())
    print('Cross-Validation Standard Deviation: ', cv_scores.std(), "\n")

    # Predict using test set
    predictions = best_model.predict(X_test)

    # Print metrics
    print('Optimized Model Performance:')
    print('-----------------------------------')
    print('Accuracy:', accuracy_score(y_test, predictions))
    print('Precision:', precision_score(y_test, predictions))
    print('Recall:', recall_score(y_test, predictions))
    print('F1 Score:', f1_score(y_test, predictions))

    # Compute ROC curve and ROC area for each model
    probs = best_model.predict_proba(X_test)
    preds = probs[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.plot(fpr, tpr, label=f'{name} (area = {roc_auc:.2f})')

# Plot layout
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Show plot
plt.show()



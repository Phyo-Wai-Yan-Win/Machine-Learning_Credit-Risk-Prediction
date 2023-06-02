import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# Load data
df = pd.read_excel("W4.1_CreditRisk.xls")

# Check for missing values
missing_values = df.isnull().sum()
print(f"Missing values:\n{missing_values}\n")

# Handle missing values
df = df.dropna(subset=['Gender'])

# One-hot encoding for categorical variables
categorical_columns = ['Checking Acct', 'Credit Hist', 'Purpose', 'Savings Acct', 'Employment', 'Gender',
                       'Personal Status', 'Housing', 'Job', 'Telephone', 'Foreign']
df = pd.get_dummies(df, columns=categorical_columns)

# Check and plot the distribution of the target variable
class_counts = df['Credit Standing'].value_counts()
plt.figure(figsize=(6, 4))
plt.bar(class_counts.index, class_counts.values)
plt.xlabel('Credit Standing')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.xticks(class_counts.index, ['Bad', 'Good'])
plt.show()

# Standardize numerical columns
num_cols = ['Months Acct (Added 1 to original Months Acct Variable)', 'Residence Time',
            'Age subtracted 1 from original Age variable']
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Encode target variable
df['Credit Standing'] = df['Credit Standing'].map({'Good': 1, 'Bad': 0})

# Split the data into training and testing sets
X = df.drop('Credit Standing', axis=1)
y = df['Credit Standing']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize the models
models = {
    'Naive Bayes': GaussianNB(),
    'SVC': SVC(probability=True),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression()
}

# Define a function to plot ROC curves
def plot_roc_curves(models):
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve of {name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Apply k-Fold Cross Validation
kfold = KFold(n_splits=10, random_state=1, shuffle=True)

# Train, evaluate models and plot ROC curves
for name, model in models.items():
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    print(f"Model: {name}")
    print(f"Cross Validation Mean Accuracy: {cv_results.mean()}")
    print(f"Standard Deviation: {cv_results.std()}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Test Set Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print(f"F1 Score: {f1_score(y_test, y_pred)}")
    print("-----------------------------------")

plot_roc_curves(models)

# Display feature importances for Random Forest
feature_importances = pd.DataFrame(models['Random Forest'].feature_importances_,
                                   index=X_train.columns, columns=['importance']).sort_values('importance', ascending=False)
print(f"Feature Importances:\n{feature_importances}\n")

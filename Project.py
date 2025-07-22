import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier , plot_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report)
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import shap
import joblib
scaler=StandardScaler()

def dataattrib(df):
    print(df.head())
    print(df.info())
    # print(df.describe())
    print(df.isnull().sum())
    all_labels = df.columns.tolist()
    for label in all_labels:
        print(df[label].value_counts(),"\n")

def perform_eda(df_original):
    df = df_original.copy()
    #Categorial value
    categorical_cols = ['workclass', 'marital-status', 'occupation', 
                    'relationship', 'race', 'gender', 'native-country', 'income']
    for col in categorical_cols:
        plt.figure(figsize=(10, 4))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index)
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    #Numerical value
    numerical_cols = ['age', 'educational-num', 'capital-gain',
                  'capital-loss', 'hours-per-week']
    df[numerical_cols].hist(bins=30, figsize=(15, 10), edgecolor='black')
    plt.suptitle('Histogram of Numerical Variables', fontsize=16)
    plt.tight_layout()
    plt.show()

    for col in categorical_cols[:-1]:  # exclude 'income' itself
        plt.figure(figsize=(10, 4))
        sns.countplot(data=df, x=col, hue="income", order=df[col].value_counts().index)
        plt.title(f'{col} vs Income')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    for col in numerical_cols:
        plt.figure(figsize=(6,4))
        sns.boxplot(data=df, y='income', x=col)
        plt.title(f'{col} distribution by Income')
        plt.tight_layout()
        plt.show()

    plt.figure(figsize=(10, 8))
    corr_matrix = df[numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="RdBu", center=0)
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.show()

def preprocess(df):
    df['occupation'] = df['occupation'].replace({'?':'No Information'})
    df['workclass'] = df['workclass'].replace({'?':'other'})
    df['native-country'] = df['native-country'].replace({'?': 'NotListed'})

    columns_to_drop=['fnlwgt','education']
    df=df.drop(columns=columns_to_drop)

    df=df[(df['age']<=74)& (df['age']>=17)]
    df=df[(df['workclass'] !='Without-pay') & (df['workclass'] !='Never-worked')]
    df=df[(df['educational-num'] != 1) & (df['educational-num'] != 2) ]
    df=df[(df['marital-status']!='Married-AF-spouse')]
    df=df[(df['occupation'] != 'Armed-Forces')& (df['occupation'] != 'Priv-house-serv')]
    df=df[(df['race'] != 'Other')]
    df=df[df['native-country']!='Holand-Netherlands']

    df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})

    df['capital-gain'] = np.log1p(df['capital-gain'])
    df['capital-loss'] = np.log1p(df['capital-loss'])

    numerical_cols = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    categorical_cols = df.select_dtypes(include='object').columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    x=df.drop(columns=['income'])
    y=df['income']
    x_train ,x_test, y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

    return x_train, x_test, y_train, y_test, scaler, x_train.columns

def train_xgboost(x_train, y_train):
   
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        n_estimators=100,         
        max_depth=5,              
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1                 # Use all cores available
    )
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    acc=accuracy_score(y_test, y_pred)
    prec=precision_score(y_test, y_pred)
    rec=recall_score(y_test, y_pred)
    f1=f1_score(y_test, y_pred)
    roc_auc=roc_auc_score(y_test, y_proba)
    print("Accuracy:",acc)
    print("Precision:",prec)
    print("Recall:",rec)
    print("F1 score:",f1)
    print("ROC-AUC:",roc_auc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return  acc, prec, rec, f1, roc_auc

def plot_roc_curves(models, x_test, y_test, model_names):
    plt.figure(figsize=(8, 6))
    for model, name in zip(models, model_names):
        # Calculate probabilities and ROC curve
        y_prob = model.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance (AUC = 0.5)")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

def train_classic_models(x_train, y_train):
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(x_train, y_train)

    # Logistic Regression
    logreg_model = LogisticRegression(max_iter=1000)
    logreg_model.fit(x_train, y_train)

    # SVM (with probability=True to support ROC/AUC)
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(x_train, y_train)

    return rf_model, logreg_model, svm_model

def plots(xgb_model):
    booster = xgb_model.get_booster()
    importances = booster.get_score(importance_type='gain')
    items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]
    features, scores = zip(*items)
    colors = plt.cm.tab20.colors 
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(features)), scores, color=colors[:len(features)])
    plt.xlabel('Feature', fontsize=12, fontweight='bold')
    plt.ylabel('Gain (Importance)', fontsize=12, fontweight='bold')
    plt.title('Top 15 Feature Importances (XGBoost)', fontsize=16, fontweight='bold')
    plt.xticks(range(len(features)), features, rotation=45, ha='right', fontsize=11)
    plt.ylim(0, max(scores) * 1.10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
        

def main():
    df = pd.read_csv(r'C:\Users\mouni\OneDrive\Desktop\Resources\Pyton-AI\Project\adult3.csv')
    # dataattrib(df)
    # perform_eda(df)
    X_train,X_test,y_train,y_test,scaler,X_train.columns=preprocess(df)
    
    joblib.dump(X_train.columns.tolist(), "columns.pkl")
    joblib.dump(scaler, "scaler.pkl")


    # xgb_model = train_xgboost(X_train, y_train)
    # plots(xgb_model)
 
    # rf_model, logreg_model, svm_model = train_classic_models(X_train, y_train)


    # models = [rf_model, xgb_model, logreg_model, svm_model]
    # model_names = ['Random Forest', 'XGBoost', 'Logistic Regression', 'SVM']

    # plt.figure(figsize=(12, 9))
    # for i, (model, name) in enumerate(zip(models, model_names)):
    #     y_pred = model.predict(X_test)
    #     plt.subplot(2, 2, i + 1)
    #     ax = plt.gca()
    #     ConfusionMatrixDisplay.from_predictions(
    #         y_test, y_pred, display_labels=['≤50K', '>50K'],
    #         cmap='Blues', ax=ax, colorbar=False
    #     )
    #     plt.title(name)
    # plt.tight_layout()
    # plt.show()

    # explainer = shap.TreeExplainer(xgb_model)
    # shap_values = explainer.shap_values(X_test)
    # shap.summary_plot(shap_values, X_test, plot_type='dot', show=True)

    # metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    # scores = {}
    # result={}
    # for model, name in zip(models, model_names):
    #     acc, prec, rec, f1, roc_auc = evaluate_model(model, X_test, y_test)  
    #     scores[name] = [acc, prec, rec, f1, roc_auc]
    #     result[name]=acc
    
    # best_model_name = max(result, key=result.get)
    # best_model_index = model_names.index(best_model_name)
    # best_model = models[best_model_index]
    # print(f"Best model: {best_model_name} with accuracy {result[best_model_name]:.4f}")
    # joblib.dump(best_model,"best_model.pkl")
    # print("Best Model is saved")

    # bar_width = 0.18
    # indices = np.arange(len(metrics))

    # plt.figure(figsize=(12, 6))
    # colors = ['#348ABD', '#A60628', '#7A68A6', '#467821']  # example color palette

    # for i, name in enumerate(model_names):
    #     plt.bar(indices + i * bar_width, scores[name], width=bar_width, label=name, color=colors[i])

    # plt.xlabel('Evaluation Metric', fontsize=12, weight='bold')
    # plt.ylabel('Score', fontsize=12, weight='bold')
    # plt.title('ML Model Metric Comparison', fontsize=16, weight='bold')

    # plt.xticks(indices + bar_width * 1.5, metrics, fontsize=10, weight='bold')
    # plt.ylim(0, 1.05)
    # plt.legend(title='Models', loc='upper left', fontsize=10)
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.tight_layout()
    # plt.show()

    # plot_roc_curves(
    #     models,
    #     x_test=X_test,
    #     y_test=y_test,
    #     model_names=['Random Forest', 'XGBoost', 'Logistic Regression', 'SVM']
    # )

    
if __name__=="__main__":
    main()
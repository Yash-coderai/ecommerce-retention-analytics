"""
E-Commerce Customer Churn Prediction
Model training, evaluation and visualization using
Random Forest and LightGBM.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report,precision_score, recall_score, f1_score, accuracy_score
import lightgbm as lgb
import warnings
import pickle
warnings.filterwarnings('ignore')

# Load Dataset
with open("cleaned_data.pkl", "rb") as f:
    data = pickle.load(f)

X_train_smote = data["X_train"]
X_test_encoded = data["X_test"]
y_train_smote = data["y_train"]
y_test = data["y_test"]
feature_names = data["feature_names"]


# Model Training - Random Forest

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

rf_model.fit(X_train_smote, y_train_smote)
rf_pred = rf_model.predict(X_test_encoded)
rf_pred_proba = rf_model.predict_proba(X_test_encoded)[:, 1]

print(f"Random Forest training complete")
print(f"\nClassification Report (Random Forest):\n{classification_report(y_test, rf_pred)}")

# Feature importance for RF
feature_importance_rf = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False).head(15)

# Model Training - LightGBM
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=31,
    random_state=42,
    is_unbalanced=True,
    verbose=-1
)

lgb_model.fit(X_train_smote, y_train_smote)
lgb_pred = lgb_model.predict(X_test_encoded)
lgb_pred_proba = lgb_model.predict_proba(X_test_encoded)[:, 1]

print(f"LightGBM training complete")
print(f"\nClassification Report (LightGBM):\n{classification_report(y_test, lgb_pred)}")

# Feature importance for LightGBM
feature_importance_lgb = pd.DataFrame({
    'feature': feature_names,
    'importance': lgb_model.feature_importances_
}).sort_values('importance', ascending=False).head(15)

# Visualize feature importance
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
feature_importance_rf.plot(x='feature', y='importance', kind='barh', ax=axes[0], color='#3498db')
axes[0].set_title('Feature Importance - Random Forest', fontweight='bold', fontsize=12)
axes[0].set_xlabel('Importance')
axes[0].invert_yaxis()
feature_importance_lgb.plot(x='feature', y='importance', kind='barh', ax=axes[1], color='#e74c3c')
axes[1].set_title('Feature Importance - LightGBM', fontweight='bold', fontsize=12)
axes[1].set_xlabel('Importance')
axes[1].invert_yaxis()
plt.tight_layout()
plt.savefig('03_feature_importance.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: 03_feature_importance.png")

# Compute ROC AUC of trained models
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_pred_proba)
roc_auc_rf = auc(fpr_rf, tpr_rf)
fpr_lgb, tpr_lgb, _ = roc_curve(y_test, lgb_pred_proba)
roc_auc_lgb = auc(fpr_lgb, tpr_lgb)

print(f"\nRandom Forest ROC-AUC: {roc_auc_rf:.3f}")
print(f"LightGBM ROC-AUC: {roc_auc_lgb:.3f}")

# Plot ROC AUC curves of trained models 
plt.figure(figsize=(10, 8))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (ROC AUC = {roc_auc_rf:.3f})', linewidth=2.5, color='#3498db')
plt.plot(fpr_lgb, tpr_lgb, label=f'LightGBM (ROC AUC = {roc_auc_lgb:.3f})', linewidth=2.5, color='#e74c3c')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('ROC Curve Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('04_roc_auc_curve.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: 04_roc_auc_curve.png")

# Compute precision-recall of trained models
precision_rf, recall_rf, _ = precision_recall_curve(y_test, rf_pred_proba)
precision_lgb, recall_lgb, _ = precision_recall_curve(y_test, lgb_pred_proba)

# Plot precision-recall curves of trained models 
plt.figure(figsize=(10, 8))
plt.plot(recall_rf, precision_rf, label=f'Random Forest', linewidth=2.5, color='#3498db')
plt.plot(recall_lgb, precision_lgb, label=f'LightGBM', linewidth=2.5, color='#e74c3c')
plt.xlabel('Recall (True Positive Rate)', fontsize=12, fontweight='bold')
plt.ylabel('Precision', fontsize=12, fontweight='bold')
plt.title('Precision-Recall Curve\n(Higher is Better)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('05_precision_recall_curve.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: 05_precision_recall_curve.png")

# Compute metrics of trained models
metrics_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Random Forest': [
        accuracy_score(y_test, rf_pred),
        precision_score(y_test, rf_pred),
        recall_score(y_test, rf_pred),
        f1_score(y_test, rf_pred),
        roc_auc_rf
    ],
    'LightGBM': [
        accuracy_score(y_test, lgb_pred),
        precision_score(y_test, lgb_pred),
        recall_score(y_test, lgb_pred),
        f1_score(y_test, lgb_pred),
        roc_auc_lgb
    ]
}

metrics_df = pd.DataFrame(metrics_data)

# Plot metrics comparison of trained models 
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(metrics_df))
width = 0.35

bars1 = ax.bar(x - width/2, metrics_df['Random Forest'], width, label='Random Forest', color='#3498db')
bars2 = ax.bar(x + width/2, metrics_df['LightGBM'], width, label='LightGBM', color='#e74c3c')

ax.set_ylabel('Score', fontweight='bold', fontsize=12)
ax.set_title('Model Metrics Comparison', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(metrics_df['Metric'])
ax.legend(fontsize=11)
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('06_metrics_comparison.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: 06_metrics_comparison.png")

# Compute confusion matrices of trained models 
cm_rf = confusion_matrix(y_test, rf_pred)
cm_lgb = confusion_matrix(y_test, lgb_pred)

# Plot confusion matrices of trained models 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
            xticklabels=['Active', 'Churned'], yticklabels=['Active', 'Churned'])
axes[0].set_title('Confusion Matrix - Random Forest', fontweight='bold')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')
sns.heatmap(cm_lgb, annot=True, fmt='d', cmap='Reds', ax=axes[1],
            xticklabels=['Active', 'Churned'], yticklabels=['Active', 'Churned'])
axes[1].set_title('Confusion Matrix - LightGBM', fontweight='bold')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('07_confusion_matrices.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: 07_confusion_matrices.png")

print(f"\nFinal Results")
print(metrics_df)
print(f"\nAll visualizations saved successfully.")

"""
Data Cleaning and Preprocessing Pipeline

Prepares the e-commerce customer churn dataset for machine learning.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import warnings
import pickle
warnings.filterwarnings('ignore')

# Load Dataset from Sheet 2 (E Comm sheet with actual data)
# Sheet 1 is "Data Dict" (metadata), Sheet 2 is "E Comm" (actual data)
try:
    df = pd.read_excel('ecommerce_customer_churn_dataset.xlsx', sheet_name='E Comm')
    print(f"Loaded from 'E Comm' sheet")
except:
    try:
        df = pd.read_excel('ecommerce_customer_churn_dataset.xlsx', sheet_name=1)
        print(f"Loaded from sheet index 1 (E Comm)")
    except:
        df = pd.read_excel('ecommerce_customer_churn_dataset.xlsx')
        print(f"Loaded from default sheet")

print(f"\nDataset loaded: {df.shape}")
df_clean = df.copy()

# Drop ID column
if 'CustomerID' in df_clean.columns:
    df_clean.drop('CustomerID', axis=1, inplace=True)


# Fix Redundant Categorical Labels
categorical_cols = df_clean.select_dtypes(include='object').columns

for col in categorical_cols:
    unique_count = df_clean[col].nunique()

redundancy_fixes = {
    'PreferredLoginDevice': {
        'Phone': 'Mobile Phone',
        'Mobile Phone': 'Mobile Phone',
        'Mobile': 'Mobile Phone'
    },
    'PreferredPaymentMode': {
        'COD': 'COD',
        'Cash on Delivery': 'COD',
        'Debit Card': 'Debit Card',
        'Credit Card': 'Credit Card',
        'E wallet': 'E wallet',
        'UPI': 'UPI'
    },
    'Gender': {
        'Male': 'Male',
        'Female': 'Female',
        'M': 'Male',
        'F': 'Female'
    }
}

for col, mapping in redundancy_fixes.items():
    if col in df_clean.columns:
        for old_val, new_val in mapping.items():
            df_clean[col] = df_clean[col].replace(old_val, new_val)

for col in categorical_cols:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].str.strip()

X = df_clean.drop('Churn', axis=1)
y = df_clean['Churn']

if y.dtype == 'object':
    y = (y == y.unique()[1]).astype(int)


# Train-test split before preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f" Training set: {X_train.shape}")
print(f" Test set: {X_test.shape}")
print(f" Training churn rate: {y_train.mean():.2%}")
print(f" Test churn rate: {y_test.mean():.2%}")

# Handling Missing Values
numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols_X = X_train.select_dtypes(include='object').columns.tolist()
missing_numeric = X_train[numerical_cols].isnull().sum()
missing_numeric = missing_numeric[missing_numeric > 0]
missing_cat = X_train[categorical_cols_X].isnull().sum()
missing_cat = missing_cat[missing_cat > 0]
skewed_features = ['WarehouseToHome', 'OrderAmountHikeFromLastYear', 
                   'HourSpendOnApp', 'Tenure', 'OrderCount', 'CashbackAmount']
imputer = SimpleImputer(strategy='median')

X_train_imputed = X_train.copy()
X_test_imputed = X_test.copy()

for col in numerical_cols:
    if col in X_train_imputed.columns:
        if X_train_imputed[col].isnull().sum() > 0:
            imputer_col = SimpleImputer(strategy='median')
            X_train_imputed[col] = imputer_col.fit_transform(X_train_imputed[[col]])
            X_test_imputed[col] = imputer_col.transform(X_test_imputed[[col]])

for col in categorical_cols_X:
    if col in X_train_imputed.columns:
        if X_train_imputed[col].isnull().sum() > 0:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            X_train_imputed[col] = imputer_cat.fit_transform(X_train_imputed[[col]])
            X_test_imputed[col] = imputer_cat.transform(X_test_imputed[[col]])


# Outlier Detection
outlier_details = {}

for col in ['WarehouseToHome', 'OrderCount', 'CashbackAmount', 'Tenure']:
    if col in X_train_imputed.columns:
        Q1 = X_train_imputed[col].quantile(0.25)
        Q3 = X_train_imputed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_mask = (X_train_imputed[col] < lower_bound) | (X_train_imputed[col] > upper_bound)
        outlier_count = outlier_mask.sum()
        outlier_details[col] = {
            'Q1': Q1, 'Q3': Q3, 'IQR': IQR,
            'lower': lower_bound, 'upper': upper_bound,
            'outlier_count': outlier_count, 'outlier_pct': outlier_count/len(X_train_imputed)*100
        }

# Remove physically impossible values (e.g., negative distances)
physical_impossibility_count = 0

if 'WarehouseToHome' in X_train_imputed.columns:
    mask = X_train_imputed['WarehouseToHome'] < 0
    if mask.sum() > 0:
        X_train_imputed = X_train_imputed[~mask]
        y_train = y_train[~mask]
        physical_impossibility_count += mask.sum()

if 'OrderCount' in X_train_imputed.columns:
    mask = X_train_imputed['OrderCount'] < 0
    if mask.sum() > 0:
        X_train_imputed = X_train_imputed[~mask]
        y_train = y_train[~mask]
        physical_impossibility_count += mask.sum()

# Log-Transformation of Skewed Features
features_to_log = ['CashbackAmount', 'OrderCount']

for col in features_to_log:
    if col in X_train_imputed.columns:
        # Add 1 to avoid log(0)
        X_train_imputed[f'{col}_log'] = np.log1p(X_train_imputed[col])
        X_test_imputed[f'{col}_log'] = np.log1p(X_test_imputed[col])

# Visualize transformations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

if 'CashbackAmount' in X_train_imputed.columns:
    axes[0, 0].hist(X_train_imputed['CashbackAmount'], bins=50, color='#3498db', edgecolor='black')
    axes[0, 0].set_title('CashbackAmount - Original (Right-Skewed)', fontweight='bold')
    axes[0, 0].set_xlabel('Amount')
    axes[0, 1].hist(X_train_imputed['CashbackAmount_log'], bins=50, color='#e74c3c', edgecolor='black')
    axes[0, 1].set_title('CashbackAmount - Log-Transformed (Reduced Skewness)', fontweight='bold')
    axes[0, 1].set_xlabel('Log(Amount + 1)')

if 'OrderCount' in X_train_imputed.columns:
    axes[1, 0].hist(X_train_imputed['OrderCount'], bins=50, color='#3498db', edgecolor='black')
    axes[1, 0].set_title('OrderCount - Original (Right-Skewed)', fontweight='bold')
    axes[1, 0].set_xlabel('Amount')
    axes[1, 1].hist(X_train_imputed['OrderCount_log'], bins=50, color='#e74c3c', edgecolor='black')
    axes[1, 1].set_title('OrderCount - Log-Transformed (Reduced Skewness)', fontweight='bold')
    axes[1, 1].set_xlabel('Log(Amount + 1)')

plt.tight_layout()
plt.savefig('01_log_transformations.png', dpi=300, bbox_inches='tight')
print(f"\nSaved visualization: 01_log_transformations.png")

# Encoding
X_train_encoded = pd.get_dummies(X_train_imputed, columns=categorical_cols_X, drop_first=True)
X_test_encoded = pd.get_dummies(X_test_imputed, columns=categorical_cols_X, drop_first=True)

train_cols = set(X_train_encoded.columns)
test_cols = set(X_test_encoded.columns)

missing_in_test = train_cols - test_cols
extra_in_test = test_cols - train_cols

if missing_in_test:
    for col in missing_in_test:
        X_test_encoded[col] = 0

if extra_in_test:
    X_test_encoded = X_test_encoded.drop(extra_in_test, axis=1)

X_test_encoded = X_test_encoded[X_train_encoded.columns]

# Visualize before SMOTE
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

y_train.value_counts().plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
axes[0].set_title('Training Set - Before SMOTE', fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(['Active', 'Churned'], rotation=0)
# SMOTE on training only
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_encoded, y_train)

y_train_balanced.value_counts().plot(kind='bar', ax=axes[1], color=['#2ecc71', '#e74c3c'])
axes[1].set_title('Training Set - After SMOTE', fontweight='bold')
axes[1].set_ylabel('Count')
axes[1].set_xticklabels(['Active', 'Churned'], rotation=0)

plt.tight_layout()
plt.savefig('02_smote_effect.png', dpi=300, bbox_inches='tight')
print(f"\nSaved visualization: 02_smote_effect.png")

# Save outputs
cleaned_data = {
    'X_train': X_train_balanced,
    'X_test': X_test_encoded,
    'y_train': y_train_balanced,
    'y_test': y_test,
    'feature_names': X_train_balanced.columns.tolist()
}

with open('cleaned_data.pkl', 'wb') as f:
    pickle.dump(cleaned_data, f)

X_train_balanced.to_csv('X_train_cleaned.csv', index=False)
X_test_encoded.to_csv('X_test_cleaned.csv', index=False)
y_train_balanced.to_csv('y_train_cleaned.csv', index=False)
y_test.to_csv('y_test_cleaned.csv', index=False)
print(f"\nSaved as CSV files:")
print(f"  • X_train_cleaned.csv")
print(f"  • X_test_cleaned.csv")
print(f"  • y_train_cleaned.csv")
print(f"  • y_test_cleaned.csv")

print(f"\nData preparation completed.")
print(f"Training shape: {X_train_balanced.shape}")
print(f"Test shape: {X_test_encoded.shape}")

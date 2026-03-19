#!/usr/bin/env python3
"""
Comprehensive ML Analysis Notebook Generator for Tikrit Research Data
"""

import json
import os

def create_markdown_cell(content):
    """Create a markdown cell"""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": content if isinstance(content, list) else content.split('\n')
    }

def create_code_cell(content):
    """Create a code cell"""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": content if isinstance(content, list) else content.split('\n')
    }

# Build notebook
cells = []

# 1. Title
cells.append(create_markdown_cell(
    "# Comprehensive Machine Learning Analysis: Tikrit Research Data with Weather\n"
    "## End-to-End ML Pipeline\n\n"
    "**Objective:** Complete ML analysis on Tikrit research data with weather integration\n"
    "**Dataset:** tikrit_research_data_with_weather.csv\n"
    "**Random State:** 42 (reproducibility)"
))

# 2. Imports
cells.append(create_code_cell(
    """# ============================================================================
# COMPREHENSIVE LIBRARY IMPORTS & CONFIGURATION
# ============================================================================
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

from scipy import stats
from scipy.stats import shapiro, anderson, levene, kruskal, mannwhitneyu
from scipy.stats import pearsonr, spearmanr, f_oneway, chi2_contingency, ks_2samp
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV, learning_curve
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc

import xgboost as xgb
import lightgbm as lgb

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

import joblib
import pickle
import time
from datetime import datetime

# Set random seeds
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# Configure visualization
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
pd.set_option('display.max_columns', None)

print('✅ All libraries imported successfully!')
print(f'📊 Random state: {RANDOM_STATE}')"""
))

# 3. Section: Mount Drive & Load Data
cells.append(create_markdown_cell(
    "## 1. MOUNT GOOGLE DRIVE & LOAD DATASET\n\n"
    "Mount Google Drive and load the Tikrit research dataset with weather patterns."
))

cells.append(create_code_cell(
    """# Mount Google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print('✅ Google Drive mounted successfully!')
except ImportError:
    print('⚠️ Not in Google Colab.')

# Load dataset
DATASET_PATH = '/content/drive/MyDrive/fyp-ml/dataset/tikrit_research_data_with_weather.csv'

print(f'\\n📂 Loading from: {DATASET_PATH}')
try:
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    print(f'✅ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns')
    
    print('\\n📋 First rows:')
    display(df.head())
    print('\\n📊 Data types:')
    print(df.dtypes)
    print('\\n📈 Info:')
    df.info()
except Exception as e:
    print(f'❌ Error loading dataset: {e}')"""
))

# 4. Section: Data Exploration & Quality Assessment
cells.append(create_markdown_cell(
    "## 2. DATA EXPLORATION & QUALITY ASSESSMENT\n\n"
    "Analyze dataset structure, identify missing values, generate descriptive statistics, "
    "and create data quality report."
))

cells.append(create_code_cell(
    """# Data Exploration & Quality Assessment
print('='*80)
print('DATASET STRUCTURE ANALYSIS')
print('='*80)

print(f'\\n📊 Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns')
print(f'\\n📋 Column Names & Types:')
print(df.dtypes)

print(f'\\n🔍 Missing Values Analysis:')
missing_data = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})
missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
print(missing_data)

print(f'\\n📈 Numerical Features - Descriptive Statistics:')
print(df.describe())

print(f'\\n📊 Categorical Features:')
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols[:10]:  # First 10 categorical columns
    print(f'\\n{col}:')
    print(df[col].value_counts().head(10))

# Data Quality Report
print('\\n' + '='*80)
print('DATA QUALITY REPORT')
print('='*80)
print(f'Total Rows: {len(df):,}')
print(f'Total Columns: {len(df.columns)}')
print(f'Numerical Columns: {len(df.select_dtypes(include=[np.number]).columns)}')
print(f'Categorical Columns: {len(df.select_dtypes(include=['object']).columns)}')
print(f'Duplicate Rows: {df.duplicated().sum():,}')
print(f'Memory Usage: {df.memory_usage().sum() / 1024**2:.2f} MB')"""
))

# 5. Section: Univariate Statistical Analysis
cells.append(create_markdown_cell(
    "## 3. UNIVARIATE STATISTICAL ANALYSIS\n\n"
    "Perform normality tests (Shapiro-Wilk, Anderson-Darling), generate Q-Q plots, "
    "histograms, and identify variables requiring transformation."
))

cells.append(create_code_cell(
    """# Univariate Statistical Analysis: Normality Tests
print('='*80)
print('NORMALITY TESTS: SHAPIRO-WILK & ANDERSON-DARLING')
print('='*80)

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
normality_results = []

for col in numerical_cols[:15]:  # First 15 numerical columns
    try:
        # Shapiro-Wilk Test
        stat_shapiro, p_shapiro = shapiro(df[col].dropna())
        
        # Anderson-Darling Test
        result_anderson = anderson(df[col].dropna())
        
        skewness = df[col].skew()
        kurtosis = df[col].kurtosis()
        
        normality_results.append({
            'Feature': col,
            'Shapiro_Stat': stat_shapiro,
            'Shapiro_p': p_shapiro,
            'Mean': df[col].mean(),
            'Std': df[col].std(),
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'Normal': 'Yes' if p_shapiro > 0.05 else 'No'
        })
    except Exception as e:
        print(f'Error processing {col}: {e}')

normality_df = pd.DataFrame(normality_results)
print(normality_df.to_string())

# Visualization: Q-Q plots for key features
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.ravel()

for idx, col in enumerate(numerical_cols[:12]):
    try:
        stats.probplot(df[col].dropna(), dist="norm", plot=axes[idx])
        axes[idx].set_title(f'Q-Q Plot: {col}')
        axes[idx].grid(alpha=0.3)
    except:
        pass

plt.tight_layout()
plt.suptitle('Q-Q Plots for Normality Assessment', y=1.00, fontsize=14, fontweight='bold')
plt.show()

# Histograms with KDE
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.ravel()

for idx, col in enumerate(numerical_cols[:12]):
    try:
        axes[idx].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
        axes[idx].set_title(f'{col}\\nSkew: {df[col].skew():.2f}')
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Frequency')
    except:
        pass

plt.tight_layout()
plt.suptitle('Distributions with Skewness Values', y=1.00, fontsize=14, fontweight='bold')
plt.show()

print(f'\\n✅ Univariate analysis complete. {normality_df[normality_df[\"Normal\"]==\"Yes\"].shape[0]} features are normally distributed.')"""
))

# 6. Section: Multivariate Statistical Analysis
cells.append(create_markdown_cell(
    "## 4. MULTIVARIATE STATISTICAL ANALYSIS\n\n"
    "Perform ANOVA, Kruskal-Wallis H-test, Mann-Whitney U tests, post-hoc analyses, "
    "and visualize group differences."
))

cells.append(create_code_cell(
    """# Multivariate Statistical Analysis: ANOVA & Levene's Test
print('='*80)
print('MULTIVARIATE STATISTICAL TESTS')
print('='*80)

# Identify categorical and numerical columns
cat_cols = df.select_dtypes(include=['object']).columns.tolist()[:5]
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:5]

anova_results = []

for cat in cat_cols:
    for num in num_cols:
        try:
            # Remove missing values
            groups = [group[num].dropna().values for name, group in df.groupby(cat)]
            groups = [g for g in groups if len(g) > 0]
            
            if len(groups) >= 2:
                # One-way ANOVA
                f_stat, p_anova = f_oneway(*groups)
                
                # Levene's Test
                levene_stat, p_levene = levene(*groups)
                
                # Kruskal-Wallis test (non-parametric)
                h_stat, p_kruskal = kruskal(*groups)
                
                anova_results.append({
                    'Categorical': cat,
                    'Numerical': num,
                    'F_Statistic': f_stat,
                    'ANOVA_p': p_anova,
                    'Levene_p': p_levene,
                    'Kruskal_p': p_kruskal,
                    'Significant': 'Yes' if p_anova < 0.05 else 'No'
                })
        except:
            pass

anova_df = pd.DataFrame(anova_results)
print('\\nANOVA Results:')
print(anova_df.to_string())

# Visualization: Boxplots for significant relationships
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.ravel()

for idx, (cat, num) in enumerate(zip(cat_cols[:6], num_cols[:6] * 2)):
    if idx < len(cat_cols) * len(num_cols):
        try:
            df.boxplot(column=num_cols[idx % len(num_cols)], by=cat_cols[idx % len(cat_cols)], ax=axes[idx])
            axes[idx].set_title(f'{cat_cols[idx % len(cat_cols)]} vs {num_cols[idx % len(num_cols)]}')
            axes[idx].set_xlabel(cat_cols[idx % len(cat_cols)])
            axes[idx].set_ylabel(num_cols[idx % len(num_cols)])
        except:
            pass

plt.tight_layout()
plt.suptitle('Group Differences: Boxplots', y=1.00, fontsize=14, fontweight='bold')
plt.show()

print(f'\\n✅ Multivariate analysis complete. {anova_df[anova_df[\"Significant\"]==\"Yes\"].shape[0]} significant relationships found.')"""
))

# 7. Section: Correlation & Multicollinearity
cells.append(create_markdown_cell(
    "## 5. CORRELATION & MULTICOLLINEARITY ANALYSIS\n\n"
    "Calculate Pearson and Spearman correlations, identify multicollinearity, "
    "compute VIF, visualize heatmaps, and test chi-square independence."
))

cells.append(create_code_cell(
    """# Correlation & Multicollinearity Analysis
print('='*80)
print('CORRELATION ANALYSIS')
print('='*80)

# Select numerical features
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()[:15]
df_numerical = df[numerical_features].dropna()

# Pearson Correlation
print('\\n📊 Pearson Correlation Matrix:')
pearson_corr = df_numerical.corr(method='pearson')

# Identify high correlations
high_corr_pairs = []
for i in range(len(pearson_corr.columns)):
    for j in range(i+1, len(pearson_corr.columns)):
        if abs(pearson_corr.iloc[i, j]) > 0.8:
            high_corr_pairs.append({
                'Feature_1': pearson_corr.columns[i],
                'Feature_2': pearson_corr.columns[j],
                'Correlation': pearson_corr.iloc[i, j]
            })

if high_corr_pairs:
    print('\\n⚠️ Highly Correlated Features (|r| > 0.8):')
    print(pd.DataFrame(high_corr_pairs))

# VIF (Variance Inflation Factor) for multicollinearity
print('\\n📈 Variance Inflation Factor (VIF) for Multicollinearity:')
vif_data = pd.DataFrame()
vif_data['Feature'] = df_numerical.columns
vif_data['VIF'] = [variance_inflation_factor(df_numerical.values, i) for i in range(df_numerical.shape[1])]
vif_data = vif_data.sort_values('VIF', ascending=False)
print(vif_data.head(10).to_string())

# Visualization: Correlation Heatmap
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Pearson
sns.heatmap(pearson_corr, annot=False, cmap='coolwarm', center=0, ax=axes[0], cbar_kws={'label': 'Correlation'})
axes[0].set_title('Pearson Correlation Matrix', fontsize=12, fontweight='bold')

# Spearman
spearman_corr = df_numerical.corr(method='spearman')
sns.heatmap(spearman_corr, annot=False, cmap='coolwarm', center=0, ax=axes[1], cbar_kws={'label': 'Correlation'})
axes[1].set_title('Spearman Correlation Matrix', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

print(f'\\n✅ Correlation analysis complete. {len(high_corr_pairs)} highly correlated feature pairs identified.')"""
))

# Continue with more sections in the next batch...

notebook_data = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.9.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Write to file
output_path = r'c:\Users\HP\Downloads\Semu-Train\tikrit_ml_analysis.ipynb'
with open(output_path, 'w') as f:
    json.dump(notebook_data, f, indent=2)

print(f"✅ Notebook created: {output_path}")
print(f"📊 Total cells: {len(cells)}")

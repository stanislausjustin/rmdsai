import pandas as pd

df = pd.read_csv('MCPHASES.csv')

df['lh'] = df['lh'].str.replace(',', '.', regex=False).astype(float)
df['estrogen'] = df['estrogen'].str.replace(',', '.', regex=False).astype(float)

print("First 5 rows of the dataset:")
print(df.head().to_markdown(index=False))

import matplotlib.pyplot as plt
import seaborn as sns

print("Descriptive Statistics for 'lh' and 'estrogen':")
print(df[['lh', 'estrogen']].describe().to_markdown(numalign="left", stralign="left"))

#histo
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df['lh'], kde=True, bins=30)
plt.title('Distribution of LH')
plt.xlabel('LH Value')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(df['estrogen'], kde=True, bins=30)
plt.title('Distribution of Estrogen')
plt.xlabel('Estrogen Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

import numpy as np

lh_q1 = df['lh'].quantile(0.25)
lh_q3 = df['lh'].quantile(0.75)

estrogen_q1 = df['estrogen'].quantile(0.25)
estrogen_q3 = df['estrogen'].quantile(0.75)

def categorize_lh(lh_value):
    if lh_value < lh_q1:
        return 'Low LH'
    elif lh_q1 <= lh_value <= lh_q3:
        return 'Normal LH'
    else:
        return 'High LH'

df['lh_category'] = df['lh'].apply(categorize_lh)

def categorize_estrogen(estrogen_value):
    if estrogen_value < estrogen_q1:
        return 'Low Estrogen'
    elif estrogen_q1 <= estrogen_value <= estrogen_q3:
        return 'Normal Estrogen'
    else:
        return 'High Estrogen'

df['estrogen_category'] = df['estrogen'].apply(categorize_estrogen)

def get_hormone_imbalance_status(row):
    lh_cat = row['lh_category']
    estrogen_cat = row['estrogen_category']

    if lh_cat == 'Normal LH' and estrogen_cat == 'Normal Estrogen':
        return 'Normal'
    elif lh_cat != 'Normal LH' and estrogen_cat != 'Normal Estrogen':
        return f'{lh_cat} and {estrogen_cat}'
    elif lh_cat != 'Normal LH':
        return lh_cat
    else:
        return estrogen_cat

df['hormone_imbalance_status'] = df.apply(get_hormone_imbalance_status, axis=1)

print("\nValue counts for 'hormone_imbalance_status':")
print(df['hormone_imbalance_status'].value_counts().to_markdown())

print("\nFirst 5 rows with new hormone categories:")
print(df[['lh', 'lh_category', 'estrogen', 'estrogen_category', 'hormone_imbalance_status']].head().to_markdown(index=False))

import matplotlib.pyplot as plt
import seaborn as sns

selected_cols = ['lh', 'estrogen', 'headaches', 'cramps', 'bloating', 'moodswing', 'stress', 'fatigue', 'appetite', 'sleepissue']

existing_selected_cols = [col for col in selected_cols if col in df.columns]

correlation_matrix_subset = df[existing_selected_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_subset, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={"size": 8})
plt.xticks(rotation=90, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

symptoms_to_analyze = [
    'headaches', 'cramps', 'bloating', 'moodswing', 'stress', 'fatigue',
    'appetite', 'sleepissue'
]

phases_order = df['phase'].unique()

num_symptoms = len(symptoms_to_analyze)
num_cols = 2
num_rows = (num_symptoms + num_cols - 1) // num_cols

plt.figure(figsize=(num_cols * 7, num_rows * 5))

for i, symptom in enumerate(symptoms_to_analyze):
    plt.subplot(num_rows, num_cols, i + 1)
    sns.boxplot(x='phase', y=symptom, data=df, order=phases_order)
    plt.title(f'Distribution of {symptom.replace("_", " ").title()} by Menstrual Phase')
    plt.xlabel('Menstrual Cycle Phase')
    plt.ylabel(symptom.replace("_", " ").title())
    plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

hormonal_data = df[['lh', 'estrogen']]

scaler = StandardScaler()
hormonal_data_scaled = scaler.fit_transform(hormonal_data)

hormonal_data_scaled_df = pd.DataFrame(hormonal_data_scaled, columns=['lh_scaled', 'estrogen_scaled'])

print("First 5 rows of scaled hormonal data:")
print(hormonal_data_scaled_df.head().to_markdown(index=False))

from sklearn.cluster import KMeans

k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['hormone_cluster'] = kmeans.fit_predict(hormonal_data_scaled)

print("\nMean 'lh' and 'estrogen' per hormone_cluster:")
print(df.groupby('hormone_cluster')[['lh', 'estrogen']].mean().to_markdown())

symptoms_to_analyze = [
    'headaches', 'cramps', 'bloating', 'fatigue', 'stress', 'appetite', 'sleepissue'
]

print(
    f"\nMean of selected symptoms per hormone_cluster:"
)
print(df.groupby('hormone_cluster')[symptoms_to_analyze].mean().to_markdown())

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='hormone_cluster', hue='phase', palette='viridis')
plt.xlabel('Hormone Cluster')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Menstrual Phase')
plt.tight_layout()
plt.show()

from scipy.stats import f_oneway

unique_phases = df['phase'].unique()

print("ANOVA Test for LH levels across Menstrual Phases:")
lh_groups = [df['lh'][df['phase'] == phase] for phase in unique_phases]
f_stat_lh, p_value_lh = f_oneway(*lh_groups)
print(f"  F-statistic: {f_stat_lh:.2f}")
print(f"  P-value: {p_value_lh:.3e}")

if p_value_lh < 0.05:
    print("  Conclusion: There are statistically significant differences in mean LH levels across menstrual phases (p < 0.05).")
else:
    print("  Conclusion: There are no statistically significant differences in mean LH levels across menstrual phases (p >= 0.05).")

print("\nANOVA Test for Estrogen levels across Menstrual Phases:")
estrogen_groups = [df['estrogen'][df['phase'] == phase] for phase in unique_phases]
f_stat_estrogen, p_value_estrogen = f_oneway(*estrogen_groups)
print(f"  F-statistic: {f_stat_estrogen:.2f}")
print(f"  P-value: {p_value_estrogen:.3e}")

if p_value_estrogen < 0.05:
    print("  Conclusion: There are statistically significant differences in mean Estrogen levels across menstrual phases (p < 0.05).")
else:
    print("  Conclusion: There are no statistically significant differences in mean Estrogen levels across menstrual phases (p >= 0.05).")

from scipy.stats import chi2_contingency

hormone_category_cols = ['lh_category', 'estrogen_category', 'hormone_imbalance_status']

for col in hormone_category_cols:
    print(f"\nChi-square Test for {col} and Menstrual Phase:")
    contingency_table = pd.crosstab(df[col], df['phase'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    print(f"  Chi-square statistic: {chi2:.2f}")
    print(f"  P-value: {p_value:.3e}")

    if p_value < 0.05:
        print(f"  Conclusion: There is a statistically significant association between {col} and menstrual phase (p < 0.05).")
    else:
        print(f"  Conclusion: There is no statistically significant association between {col} and menstrual phase (p >= 0.05).")

import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.violinplot(x='phase', y='lh', data=df, palette='viridis', order=unique_phases)
#plt.title('LH Levels Across Menstrual Phases (Violin Plot)')
plt.xlabel('Menstrual Phase')
plt.ylabel('LH Level')
plt.xticks(rotation=45, ha='right')

plt.subplot(1, 2, 2)
sns.violinplot(x='phase', y='estrogen', data=df, palette='viridis', order=unique_phases)
#plt.title('Estrogen Levels Across Menstrual Phases (Violin Plot)')
plt.xlabel('Menstrual Phase')
plt.ylabel('Estrogen Level')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

hormone_category_cols = ['lh_category', 'estrogen_category', 'hormone_imbalance_status']
unique_phases = df['phase'].unique()

for col in hormone_category_cols:
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='phase', hue=col, palette='viridis', order=unique_phases)
    plt.xlabel('Menstrual Phase')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Distribution of {col.replace("_", " ").title()} by Menstrual Phase') # Added title
    if col == 'hormone_imbalance_status':
        plt.legend(title=col.replace("_", " ").title(), bbox_to_anchor=(0.98, 0.98), loc='upper right', fontsize='xx-small')
    else:
        plt.legend(title=col.replace("_", " ").title(), bbox_to_anchor=(0.98, 0.98), loc='upper right')
    plt.tight_layout()
    plt.show()

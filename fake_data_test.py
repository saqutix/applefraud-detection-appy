import pandas as pd
import numpy as np

# Load the holdout test set
df_holdout = pd.read_csv('holdout_test_set.csv')

# Separate fraud and legitimate
fraud_samples = df_holdout[df_holdout['Class'] == 1]
legit_samples = df_holdout[df_holdout['Class'] == 0]

# Take 25 fraud and 25 legitimate (balanced)
n_fraud = 25
n_legit = 25

# Sample
fraud_sample = fraud_samples.sample(n=n_fraud, random_state=42)
legit_sample = legit_samples.sample(n=n_legit, random_state=123)

# Combine and shuffle
balanced_sample = pd.concat([fraud_sample, legit_sample])
balanced_sample = balanced_sample.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
balanced_sample.to_csv('fake_test_data_balanced.csv', index=False)

print(f" Created BALANCED fake test data:")
print(f"   Total: {len(balanced_sample)} transactions")
print(f"   Fraud: {balanced_sample['Class'].sum()}")
print(f"   Legit: {len(balanced_sample) - balanced_sample['Class'].sum()}")
print(f"   Fraud %: {balanced_sample['Class'].mean()*100:.1f}%")
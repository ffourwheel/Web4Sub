import pandas as pd
import numpy as np

df = pd.read_csv('../data/cleansing_water_data.csv')
df2 = df[df['use_cleansing_water'] == '\u0e43\u0e0a\u0e49'].dropna(subset=['skin_type','concerns'])
print(f"Total rows: {len(df)}")
print(f"Filtered rows (used for model): {len(df2)}")
print(f"Unique skin types: {df2['skin_type'].nunique()}")
print(f"Skin types: {df2['skin_type'].unique().tolist()}")

# Count features after encoding
skin_dummies = pd.get_dummies(df2['skin_type'], prefix='skin')
def get_dummies_multiselect(series, prefix):
    dummies = series.str.split(',').explode().str.strip().str.get_dummies()
    return dummies.add_prefix(f"{prefix}_").groupby(level=0).sum()
concern_dummies = get_dummies_multiselect(df2['concerns'], 'concern')
X = pd.concat([skin_dummies, concern_dummies], axis=1).fillna(0)
print(f"\nNumber of features after encoding: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")
print(f"Feature-to-sample ratio: {X.shape[1]/X.shape[0]:.2f}")
print(f"\nFeature names: {list(X.columns)}")

import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# =============================================================================
# 1. Configuration & Setup
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / 'models' / 'data' / 'cleansing_water_data.csv'
OUT_JSON = BASE_DIR / 'models' / 'unsupervise' / 'cluster_profiles.json'
IMG_DIR = BASE_DIR / 'models' / 'images' / 'unsup'

IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2. Data Preprocessing
# =============================================================================
df = pd.read_csv(DATA_PATH)

df_clean = df.dropna(subset=['skin_type', 'concerns']).copy()

skin_type_encoded = pd.get_dummies(df_clean['skin_type'], prefix='skin')

def get_dummies_multiselect(series, prefix):
    dummies = series.str.split(',').explode().str.strip().str.get_dummies()
    return dummies.add_prefix(f"{prefix}_").groupby(level=0).sum()

concerns_encoded = get_dummies_multiselect(df_clean['concerns'], 'concern')

X = pd.concat([skin_type_encoded, concerns_encoded], axis=1).fillna(0)

# =============================================================================
# 3. K-Means Clustering Modeling
# =============================================================================
N_CLUSTERS = 3
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)

df_clean['cluster'] = kmeans.fit_predict(X)


# =============================================================================
# 4. Profile Generation (Analysis)
# =============================================================================
cluster_profiles = []

for c in range(N_CLUSTERS):
    cluster_data = df_clean[df_clean['cluster'] == c]
    size = len(cluster_data)
    
    features_mean = X.loc[cluster_data.index].mean().round(3).to_dict()
    
    skin_means = {k.replace('skin_', ''): v for k, v in features_mean.items() if k.startswith('skin_')}
    concern_means = {k.replace('concern_', ''): v for k, v in features_mean.items() if k.startswith('concern_')}

    top_skin = sorted(skin_means.items(), key=lambda x: x[1], reverse=True)[:2]
    top_concerns = sorted(concern_means.items(), key=lambda x: x[1], reverse=True)[:3]
    
    cluster_profiles.append({
        'cluster_id': c,
        'size': size,
        'percentage': round((size / len(df_clean)) * 100, 1),
        'top_skin_types': [{"name": k, "pct": v} for k, v in top_skin if v > 0],
        'top_concerns': [{"name": k, "pct": v} for k, v in top_concerns if v > 0],
        'all_features': features_mean
    })

with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump({"clusters": cluster_profiles}, f, ensure_ascii=False, indent=4)


# =============================================================================
# 5. Visualizations
# =============================================================================
plt.rcParams['font.family'] = 'Tahoma'
plt.figure(figsize=(8, 5))
sizes = [p['percentage'] for p in cluster_profiles]
labels = [f"Cluster {c}" for c in range(N_CLUSTERS)]
colors = ['#f5576c', '#4facfe', '#43e97b']

bars = plt.bar(labels, sizes, color=colors, alpha=0.8)
plt.title("Cluster Distribution (%)", fontsize=14, pad=15)
plt.ylabel("Percentage (%)")
plt.ylim(0, max(sizes) + 10)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval}%", ha='center', va='bottom', color='black', fontweight='bold')

plt.tight_layout()
plt.savefig(IMG_DIR / 'cluster_sizes.png', dpi=150)
plt.close()

cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)
clean_centers = cluster_centers.loc[:, (cluster_centers > 0.15).any(axis=0)]
clean_centers.index = [f"Cluster {i}" for i in range(N_CLUSTERS)]
clean_centers.columns = [c.replace('concern_', '').replace('skin_', '') for c in clean_centers.columns]

plt.figure(figsize=(10, 6))
sns.heatmap(clean_centers.T, annot=True, cmap="Purples", fmt=".2f", cbar=True)
plt.title("Feature Dominance per Cluster", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig(IMG_DIR / 'cluster_heatmap.png', dpi=150)
plt.close()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
df_clean['pca_1'] = X_pca[:, 0]
df_clean['pca_2'] = X_pca[:, 1]

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df_clean, x='pca_1', y='pca_2', hue='cluster', 
    palette=colors, s=100, alpha=0.9, edgecolor='white'
)
plt.title("Customer Segments (PCA Projection)", fontsize=14, pad=15)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(IMG_DIR / 'cluster_pca.png', dpi=150)
plt.close()
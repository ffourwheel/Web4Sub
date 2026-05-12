import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
from pathlib import Path
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = BASE_DIR / 'backend' / 'survey.db'
IMG_DIR = BASE_DIR / 'models' / 'images' / 'unsup'

IMG_DIR.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(str(DB_PATH))
df = pd.read_sql_query('SELECT * FROM survey_responses', conn)
conn.close()

df_clean = df[df['use_cleansing_water'] == 'ใช้'].dropna(subset=['skin_type', 'concerns']).copy()

skin_type_encoded = pd.get_dummies(df_clean['skin_type'], prefix='skin')

def get_dummies_multiselect(series, prefix):
    dummies = series.str.split(',').explode().str.strip().str.get_dummies()
    return dummies.add_prefix(f"{prefix}_").groupby(level=0).sum()

concerns_encoded = get_dummies_multiselect(df_clean['concerns'], 'concern')

X = pd.concat([skin_type_encoded, concerns_encoded], axis=1).fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_clustering = PCA(n_components=2, random_state=42)
X_pca = pca_clustering.fit_transform(X_scaled)

N_CLUSTERS = 3

models = {
    'K-Means': KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10),
    'Agglomerative': AgglomerativeClustering(n_clusters=N_CLUSTERS),
    'Gaussian Mixture': GaussianMixture(n_components=N_CLUSTERS, random_state=42)
}

model_scores = {}
model_labels = {}

for name, model in models.items():
    labels = model.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    
    model_scores[name] = score
    model_labels[name] = labels
    print(f" - {name}: {score:.4f}")

best_model_name = max(model_scores, key=model_scores.get)

df_clean['cluster'] = model_labels[best_model_name]

iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
df_clean['anomaly_label'] = iso_forest.fit_predict(X_scaled)
df_clean['anomaly_status'] = df_clean['anomaly_label'].map({1: 'Normal', -1: 'Anomaly'})

plt.rcParams['font.family'] = 'Tahoma'
colors = ['#f5576c', '#4facfe', '#43e97b', '#f8b500']

pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)
df_clean['pca_1'] = X_pca_2d[:, 0]
df_clean['pca_2'] = X_pca_2d[:, 1]

plt.figure(figsize=(8, 5))
bars = plt.bar(model_scores.keys(), model_scores.values(), color=['#667eea', '#764ba2', '#ed6ea0'])
plt.title("Model Comparison (Silhouette Score - Higher is better)", fontsize=14, pad=15)
plt.ylabel("Silhouette Score")
plt.ylim(0, max(model_scores.values()) + 0.1)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.3f}", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(IMG_DIR / '1_model_scores.png', dpi=150)
plt.close()
#plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
fig.suptitle(f"Customer Segmentation Comparison (Best: {best_model_name})", fontsize=16)

for ax, (name, labels) in zip(axes, model_labels.items()):
    sns.scatterplot(
        x=df_clean['pca_1'], y=df_clean['pca_2'], 
        hue=labels, palette=colors[:N_CLUSTERS], 
        s=80, alpha=0.8, edgecolor='white', ax=ax, legend=False
    )
    ax.set_title(f"{name} (Score: {model_scores[name]:.3f})")
    ax.set_xlabel("PCA 1")
    if ax == axes[0]: ax.set_ylabel("PCA 2")

plt.tight_layout()
plt.savefig(IMG_DIR / '2_models_pca_comparison.png', dpi=150)
plt.close()
#plt.show()

X_temp = X.copy()
X_temp['cluster'] = df_clean['cluster']
cluster_centers = X_temp.groupby('cluster')[X.columns].mean()

clean_centers = cluster_centers.loc[:, (cluster_centers > 0.15).any(axis=0)]
clean_centers.index = [f"Cluster {i}" for i in range(N_CLUSTERS)]
clean_centers.columns = [c.replace('concern_', '').replace('skin_', '') for c in clean_centers.columns]

plt.figure(figsize=(10, 6))
sns.heatmap(clean_centers.T, annot=True, cmap="Purples", fmt=".2f", cbar=True)
plt.title(f"Feature Dominance per Cluster ({best_model_name})", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig(IMG_DIR / '3_best_cluster_heatmap.png', dpi=150)
plt.close()
#plt.show()

plt.figure(figsize=(9, 6))
anomaly_colors = {'Normal': '#4facfe', 'Anomaly': '#ff6b6b'}
sns.scatterplot(
    data=df_clean, x='pca_1', y='pca_2', hue='anomaly_status',
    palette=anomaly_colors, s=100, alpha=0.85, edgecolor='white',
    style='anomaly_status', markers={'Normal': 'o', 'Anomaly': 'X'}
)
plt.title("Anomaly Detection (Isolation Forest)", fontsize=14, pad=15)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title="Status", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(IMG_DIR / '4_anomaly_detection.png', dpi=150)
plt.close()
#plt.show()

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df_clean, x='pca_1', y='pca_2', hue='cluster', 
    palette=colors[:N_CLUSTERS], s=100, alpha=0.9, edgecolor='white'
)
plt.title(f"Customer Segments (Best Model: {best_model_name})", fontsize=14, pad=15)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(IMG_DIR / '5_best_model_segmentation.png', dpi=150)
plt.close()
#plt.show()

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
    })

n_anomalies = int((df_clean['anomaly_label'] == -1).sum())
anomaly_pct = round((n_anomalies / len(df_clean)) * 100, 1)

conn = sqlite3.connect(str(DB_PATH))

cluster_df = pd.DataFrame({
    'cluster_id': [cp['cluster_id'] for cp in cluster_profiles],
    'size': [cp['size'] for cp in cluster_profiles],
    'percentage': [cp['percentage'] for cp in cluster_profiles],
    'top_skin_types': [json.dumps(cp['top_skin_types'], ensure_ascii=False) for cp in cluster_profiles],
    'top_concerns': [json.dumps(cp['top_concerns'], ensure_ascii=False) for cp in cluster_profiles],
})
cluster_df.to_sql('cluster_profiles', conn, if_exists='replace', index=False)

scores_df = pd.DataFrame([
    {'model_name': k, 'silhouette_score': v} for k, v in model_scores.items()
])
scores_df['is_best'] = (scores_df['model_name'] == best_model_name).astype(int)
scores_df.to_sql('unsup_model_scores', conn, if_exists='replace', index=False)

anomaly_df = pd.DataFrame([{
    'method': 'Isolation Forest',
    'n_anomalies': n_anomalies,
    'anomaly_percentage': anomaly_pct,
}])
anomaly_df.to_sql('anomaly_detection', conn, if_exists='replace', index=False)

conn.close()
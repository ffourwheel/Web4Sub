import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

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
# 3. K-Means Clustering Modeling (วิธีที่ 1: Clustering)
# =============================================================================
N_CLUSTERS = 3
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)

df_clean['cluster'] = kmeans.fit_predict(X)


# =============================================================================
# 4. Anomaly Detection - Isolation Forest (วิธีที่ 2: Anomaly Detection)
# =============================================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.1,  # คาดว่า ~10% เป็น anomaly
    random_state=42
)

df_clean['anomaly_label'] = iso_forest.fit_predict(X_scaled)  # 1=normal, -1=anomaly
df_clean['anomaly_score'] = iso_forest.decision_function(X_scaled)

# Map labels: -1 → "Anomaly", 1 → "Normal"
df_clean['anomaly_status'] = df_clean['anomaly_label'].map({1: 'Normal', -1: 'Anomaly'})

# Anomaly statistics
n_anomalies = int((df_clean['anomaly_label'] == -1).sum())
n_normal = int((df_clean['anomaly_label'] == 1).sum())
anomaly_pct = round((n_anomalies / len(df_clean)) * 100, 1)

# Anomaly distribution per cluster
anomaly_per_cluster = []
for c in range(N_CLUSTERS):
    cluster_data = df_clean[df_clean['cluster'] == c]
    anom_count = int((cluster_data['anomaly_label'] == -1).sum())
    total = len(cluster_data)
    anomaly_per_cluster.append({
        'cluster_id': c,
        'total': total,
        'anomalies': anom_count,
        'normal': total - anom_count,
        'anomaly_rate': round((anom_count / total) * 100, 1) if total > 0 else 0
    })

# Anomaly feature analysis — what features are most different in anomalies
anomaly_data = df_clean[df_clean['anomaly_label'] == -1]
normal_data = df_clean[df_clean['anomaly_label'] == 1]

anomaly_feature_diff = []
for col in X.columns:
    anom_mean = X.loc[anomaly_data.index, col].mean()
    norm_mean = X.loc[normal_data.index, col].mean()
    diff = abs(anom_mean - norm_mean)
    if diff > 0.1:  # Only include significant differences
        clean_name = col.replace('concern_', '').replace('skin_', '')
        anomaly_feature_diff.append({
            'feature': clean_name,
            'anomaly_mean': round(float(anom_mean), 3),
            'normal_mean': round(float(norm_mean), 3),
            'difference': round(float(diff), 3)
        })

anomaly_feature_diff.sort(key=lambda x: x['difference'], reverse=True)


# =============================================================================
# 5. PCA Analysis (Dimensionality Reduction — เสริมวิเคราะห์)
# =============================================================================
pca_full = PCA(n_components=min(X.shape[1], X.shape[0]))
pca_full.fit(X_scaled)

# Explained variance
explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Number of components to explain 90% variance
n_components_90 = int(np.argmax(cumulative_variance >= 0.90) + 1)
n_components_95 = int(np.argmax(cumulative_variance >= 0.95) + 1)

# PCA 2D for visualization
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)
df_clean['pca_1'] = X_pca_2d[:, 0]
df_clean['pca_2'] = X_pca_2d[:, 1]

pca_summary = {
    'total_features': int(X.shape[1]),
    'total_samples': int(X.shape[0]),
    'components_for_90_pct': n_components_90,
    'components_for_95_pct': n_components_95,
    'variance_explained_2d': round(float(cumulative_variance[1]) * 100, 1),
    'top_components': [
        {
            'component': i + 1,
            'variance_explained': round(float(v) * 100, 1),
            'cumulative': round(float(cumulative_variance[i]) * 100, 1)
        }
        for i, v in enumerate(explained_variance[:5])
    ]
}


# =============================================================================
# 6. Profile Generation (Cluster Analysis)
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


# =============================================================================
# 7. Save JSON Output
# =============================================================================
output_data = {
    "clusters": cluster_profiles,
    "anomaly_detection": {
        "method": "Isolation Forest",
        "total_samples": int(len(df_clean)),
        "n_anomalies": n_anomalies,
        "n_normal": n_normal,
        "anomaly_percentage": anomaly_pct,
        "contamination_param": 0.1,
        "per_cluster": anomaly_per_cluster,
        "top_anomaly_features": anomaly_feature_diff[:8]
    },
    "pca_analysis": pca_summary
}

with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)


# =============================================================================
# 8. Visualizations
# =============================================================================
plt.rcParams['font.family'] = 'Tahoma'
colors = ['#f5576c', '#4facfe', '#43e97b']

# ── 8.1 Cluster Distribution Bar Chart ──
plt.figure(figsize=(8, 5))
sizes = [p['percentage'] for p in cluster_profiles]
labels = [f"Cluster {c}" for c in range(N_CLUSTERS)]

bars = plt.bar(labels, sizes, color=colors, alpha=0.8)
plt.title("Cluster Distribution (%)", fontsize=14, pad=15)
plt.ylabel("Percentage (%)")
plt.ylim(0, max(sizes) + 10)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval}%", ha='center', va='bottom', color='black', fontweight='bold')

plt.tight_layout()
plt.savefig(IMG_DIR / 'cluster_sizes.png', dpi=150)
#plt.show()

# ── 8.2 Cluster Feature Heatmap ──
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)
clean_centers = cluster_centers.loc[:, (cluster_centers > 0.15).any(axis=0)]
clean_centers.index = [f"Cluster {i}" for i in range(N_CLUSTERS)]
clean_centers.columns = [c.replace('concern_', '').replace('skin_', '') for c in clean_centers.columns]

plt.figure(figsize=(10, 6))
sns.heatmap(clean_centers.T, annot=True, cmap="Purples", fmt=".2f", cbar=True)
plt.title("Feature Dominance per Cluster", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig(IMG_DIR / 'cluster_heatmap.png', dpi=150)
#plt.show()

# ── 8.3 PCA Cluster Scatter Plot ──
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
#plt.show()

# ── 8.4 Anomaly Detection Scatter Plot (PCA 2D) ──
plt.figure(figsize=(9, 6))
anomaly_colors = {'Normal': '#4facfe', 'Anomaly': '#ff6b6b'}
sns.scatterplot(
    data=df_clean, x='pca_1', y='pca_2', hue='anomaly_status',
    palette=anomaly_colors, s=100, alpha=0.85, edgecolor='white',
    style='anomaly_status', markers={'Normal': 'o', 'Anomaly': 'X'}
)
plt.title("Anomaly Detection (Isolation Forest + PCA)", fontsize=14, pad=15)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Status", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(IMG_DIR / 'anomaly_pca.png', dpi=150)
#plt.show()

# ── 8.5 Anomaly Score Distribution ──
plt.figure(figsize=(8, 5))
plt.hist(df_clean['anomaly_score'], bins=30, color='#667eea', alpha=0.7, edgecolor='white')
threshold = iso_forest.offset_
plt.axvline(x=threshold, color='#ff6b6b', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.3f})')
plt.title("Anomaly Score Distribution", fontsize=14, pad=15)
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig(IMG_DIR / 'anomaly_score_dist.png', dpi=150)
#plt.show()

# ── 8.6 PCA Variance Explained ──
plt.figure(figsize=(8, 5))
n_show = min(10, len(explained_variance))
x_range = range(1, n_show + 1)
plt.bar(x_range, explained_variance[:n_show] * 100, color='#667eea', alpha=0.7, label='Individual')
plt.plot(x_range, cumulative_variance[:n_show] * 100, 'o-', color='#f5576c', linewidth=2, label='Cumulative')
plt.axhline(y=90, color='#43e97b', linestyle='--', alpha=0.7, label='90% threshold')
plt.title("PCA Explained Variance", fontsize=14, pad=15)
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained (%)")
plt.legend()
plt.tight_layout()
plt.savefig(IMG_DIR / 'pca_variance.png', dpi=150)
#plt.show()
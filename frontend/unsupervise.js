/**
 * Unsupervised Learning Dashboard
 * Fetches and displays customer segments from FastAPI.
 * Aligned with unsupervise.py output: 3 models + Isolation Forest
 */

const isLocal = window.location.hostname === '127.0.0.1' || window.location.hostname === 'localhost' || window.location.protocol === 'file:';
const API_BASE = isLocal ? 'http://127.0.0.1:8000' : '';

// Colors matching the Python script (K=3: Pink, Blue, Green)
const CLUSTER_COLORS = {
    0: "#f5576c",
    1: "#4facfe",
    2: "#43e97b"
};

// Model colors matching unsupervise.py bar chart
const MODEL_COLORS = {
    'K-Means': '#667eea',
    'Agglomerative': '#764ba2',
    'Gaussian Mixture': '#ed6ea0'
};

// ── Initialize everything after DOM is ready ──
document.addEventListener('DOMContentLoaded', () => {

    // Observer for scroll animation
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

    // Observe all sections
    document.querySelectorAll('.unsup-section').forEach(el => {
        observer.observe(el);
    });

    // Load data from API
    loadClusterProfiles(observer);
});

async function loadClusterProfiles(observer) {
    const distContainer = document.getElementById('distribution-bars');
    const personaContainer = document.getElementById('persona-cards');

    try {
        const res = await fetch(`${API_BASE}/api/unsupervise`);
        if (!res.ok) throw new Error(`HTTP Error: ${res.status}`);

        const data = await res.json();

        if (data.error) {
            throw new Error(data.error);
        }

        const bestModel = data.best_model_used || 'Unknown';
        const modelScores = data.model_scores || {};
        const totalSamples = data.clusters
            ? data.clusters.reduce((sum, c) => sum + c.size, 0)
            : 0;

        // Generate dynamic cluster names from the data
        const clusterNames = generateClusterNames(data.clusters);

        // Render model scores
        renderModelScores(modelScores, bestModel);

        // Update dynamic descriptions
        updateDynamicText(bestModel, totalSamples);

        // Render distribution bars
        renderDistributionBars(data.clusters, distContainer, clusterNames);

        // Render persona cards
        renderPersonaCards(data.clusters, personaContainer, observer, clusterNames);

        // Render anomaly data
        if (data.anomaly_detection) {
            renderAnomalyData(data.anomaly_detection, totalSamples);
        }

        // Animate cards that are already visible
        animateVisibleCards();

    } catch (err) {
        console.error("Failed to load clusters:", err);
        const errorHTML = `
            <div class="predict-result-error" style="grid-column: 1 / -1;">
                <i class="ph-fill ph-warning-circle"></i>
                <p>เกิดข้อผิดพลาดในการโหลดข้อมูล: ${err.message}</p>
            </div>
        `;
        if (distContainer) distContainer.innerHTML = errorHTML;
        if (personaContainer) personaContainer.innerHTML = errorHTML;
    }
}

// ── Generate cluster names dynamically from top features ──
function generateClusterNames(clusters) {
    if (!clusters) return {};
    const names = {};
    clusters.forEach(cluster => {
        const topSkin = cluster.top_skin_types && cluster.top_skin_types.length > 0
            ? cluster.top_skin_types[0].name
            : '';
        const topConcern = cluster.top_concerns && cluster.top_concerns.length > 0
            ? cluster.top_concerns[0].name
            : '';
        names[cluster.cluster_id] = `${topSkin} / ${topConcern}`;
    });
    return names;
}

// ── Update dynamic text based on API data ──
function updateDynamicText(bestModel, totalSamples) {
    const descEl = document.getElementById('distribution-desc');
    if (descEl) {
        descEl.textContent = `จากกลุ่มตัวอย่าง ${totalSamples} คน แบ่งด้วย ${bestModel} (k=3)`;
    }

    const pcaDescEl = document.getElementById('pca-card-desc');
    if (pcaDescEl) {
        pcaDescEl.textContent = `การกระจายตัวของกลุ่มลูกค้าจาก ${bestModel}`;
    }

    const methodEl = document.getElementById('method-info');
    if (methodEl) {
        methodEl.innerHTML = `<strong>Best Model:</strong> ${bestModel} — แบ่งกลุ่มตามความสัมพันธ์ของฟีเจอร์ (skin_type + concerns) ผ่าน PCA ลดมิติ แล้วเลือกโมเดลที่ได้ Silhouette Score สูงสุด`;
    }
}

// ── Model Scores ──
function renderModelScores(scores, bestModel) {
    const container = document.getElementById('model-scores-list');
    const badgeEl = document.getElementById('best-model-badge');
    if (!container) return;

    const sortedModels = Object.entries(scores).sort((a, b) => b[1] - a[1]);
    const maxScore = sortedModels.length > 0 ? sortedModels[0][1] : 1;

    container.innerHTML = sortedModels.map(([name, score]) => {
        const color = MODEL_COLORS[name] || '#ffffff';
        const isBest = name === bestModel;
        const widthPct = (score / (maxScore + 0.05)) * 100;

        return `
            <div class="unsup-model-row ${isBest ? 'unsup-model-best' : ''}">
                <div class="unsup-model-row-header">
                    <div class="unsup-model-row-name">
                        <div class="unsup-bar-dot" style="background-color: ${color};"></div>
                        ${name}
                        ${isBest ? '<span class="unsup-best-tag">BEST</span>' : ''}
                    </div>
                    <div class="unsup-model-row-score" style="color: ${color};">${score.toFixed(4)}</div>
                </div>
                <div class="unsup-bar-track">
                    <div class="unsup-bar-fill" style="background: ${color}; width: ${widthPct}%;"></div>
                </div>
            </div>
        `;
    }).join('');

    if (badgeEl) {
        badgeEl.innerHTML = `
            <i class="ph-fill ph-crown"></i>
            <span>Best Model: <strong>${bestModel}</strong> (Score: ${scores[bestModel]?.toFixed(4) || 'N/A'})</span>
        `;
    }
}

// ── Anomaly Detection ──
function renderAnomalyData(anomaly, totalSamples) {
    const pctEl = document.getElementById('anomaly-pct');
    const countEl = document.getElementById('anomaly-counts');

    if (pctEl) pctEl.innerText = `${anomaly.anomaly_percentage}%`;
    if (countEl) countEl.innerText = `พบ ${anomaly.n_anomalies} คน จาก ${totalSamples} คน`;

    // Animate cards visibility
    const anomalyCards = document.querySelectorAll('#anomaly-section .sup-model-card');
    anomalyCards.forEach((card, index) => {
        setTimeout(() => {
            card.classList.add('visible');
        }, 100 * index);
    });
}

// ── Distribution Progress Bars ──
function renderDistributionBars(clusters, container, clusterNames) {
    if (!container) return;
    container.innerHTML = '';

    clusters.forEach(cluster => {
        const color = CLUSTER_COLORS[cluster.cluster_id] || "#ffffff";
        const name = clusterNames[cluster.cluster_id] || `Cluster ${cluster.cluster_id}`;

        const row = document.createElement('div');
        row.className = 'unsup-bar-row';
        row.innerHTML = `
            <div class="unsup-bar-header">
                <div class="unsup-bar-name">
                    <div class="unsup-bar-dot" style="background-color: ${color};"></div>
                    Cluster ${cluster.cluster_id}: ${name}
                </div>
                <div class="unsup-bar-meta">
                    <span class="count">${cluster.size} คน</span>
                    <strong class="pct" style="color: ${color};">${cluster.percentage.toFixed(1)}%</strong>
                </div>
            </div>
            <div class="unsup-bar-track">
                <div class="unsup-bar-fill" style="background: ${color}; width: ${cluster.percentage}%;"></div>
            </div>
        `;
        container.appendChild(row);
    });
}

// ── Persona Cards ──
function renderPersonaCards(clusters, container, observer, clusterNames) {
    if (!container) return;
    container.innerHTML = '';

    clusters.forEach((cluster, index) => {
        const color = CLUSTER_COLORS[cluster.cluster_id] || "#ffffff";
        const name = clusterNames[cluster.cluster_id] || `Cluster ${cluster.cluster_id}`;

        // Merge skin types and concerns, sort by percentage
        let topFeatures = [];
        if (cluster.top_skin_types) {
            cluster.top_skin_types.forEach(s => topFeatures.push({ name: s.name, pct: s.pct, type: 'skin' }));
        }
        if (cluster.top_concerns) {
            cluster.top_concerns.forEach(c => topFeatures.push({ name: c.name, pct: c.pct, type: 'concern' }));
        }
        topFeatures.sort((a, b) => b.pct - a.pct);

        // Build feature bars (top 5)
        const featuresHTML = topFeatures.slice(0, 5).map(f => {
            const percent = (f.pct * 100).toFixed(0);
            const icon = f.type === 'skin'
                ? '<i class="ph-fill ph-drop" style="font-size: 12px; opacity: 0.6;"></i>'
                : '<i class="ph-fill ph-bandaids" style="font-size: 12px; opacity: 0.6;"></i>';
            return `
                <div class="unsup-feat-row">
                    <div class="unsup-feat-header">
                        <span class="unsup-feat-name">${icon} ${f.name}</span>
                        <span style="color: ${color};">${percent}%</span>
                    </div>
                    <div class="unsup-feat-track">
                        <div class="unsup-feat-fill" style="background: ${color}; width: ${percent}%;"></div>
                    </div>
                </div>
            `;
        }).join('');

        // Generate description from data
        const skinDesc = cluster.top_skin_types && cluster.top_skin_types.length > 0
            ? `สภาพผิวหลัก: ${cluster.top_skin_types.map(s => `${s.name} (${(s.pct * 100).toFixed(0)}%)`).join(', ')}`
            : '';
        const concernDesc = cluster.top_concerns && cluster.top_concerns.length > 0
            ? `ปัญหาเด่น: ${cluster.top_concerns.map(c => `${c.name} (${(c.pct * 100).toFixed(0)}%)`).join(', ')}`
            : '';
        const desc = `${skinDesc}${skinDesc && concernDesc ? ' — ' : ''}${concernDesc}`;

        const card = document.createElement('div');
        card.className = 'sup-model-card unsup-persona';

        card.innerHTML = `
            <div class="unsup-persona-header">
                <div class="unsup-persona-icon" style="color: ${color};">
                    <i class="ph-fill ph-user-focus"></i>
                </div>
                <div>
                    <div class="unsup-persona-label" style="color: ${color};">
                        CLUSTER ${cluster.cluster_id}
                    </div>
                    <div class="unsup-persona-name">${name}</div>
                    <div class="unsup-persona-pct">
                        ${cluster.percentage.toFixed(1)}% (${cluster.size} คน)
                    </div>
                </div>
            </div>

            <p class="unsup-persona-desc">${desc}</p>

            <div class="unsup-features-box">
                <div class="unsup-features-title">Feature Importance</div>
                <div class="unsup-features-list">
                    ${featuresHTML}
                </div>
            </div>
        `;

        container.appendChild(card);

        // Staggered fade-in animation
        setTimeout(() => {
            card.classList.add('visible');
        }, 100 * (index + 1));
    });
}

// ── Animate cards that are already in viewport ──
function animateVisibleCards() {
    document.querySelectorAll('.sup-model-card').forEach((card, index) => {
        setTimeout(() => {
            card.classList.add('visible');
        }, 80 * index);
    });
}

// ── Lightbox logic ──
function openImageOnly(src) {
    const lb = document.getElementById('lightbox');
    const lbImg = document.getElementById('lightbox-img');
    const lbCap = document.getElementById('lightbox-caption');

    lbImg.src = src;
    if (lbCap) lbCap.style.display = 'none';

    lb.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function openLightbox(card) {
    const lb = document.getElementById('lightbox');
    const lbImg = document.getElementById('lightbox-img');
    const lbCap = document.getElementById('lightbox-caption');

    const img = card.querySelector('img');
    const cap = card.querySelector('.sup-graph-caption');

    lbImg.src = img.src;
    if (cap) {
        lbCap.style.display = 'block';
        lbCap.innerHTML = cap.innerHTML;
    } else {
        lbCap.style.display = 'none';
    }

    lb.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeLightbox(e) {
    if (e.target.id === 'lightbox' || e.target.id === 'lightbox-close') {
        document.getElementById('lightbox').classList.remove('active');
        document.body.style.overflow = '';
    }
}

// Close lightbox on Escape key
document.addEventListener('keydown', (e) => {
    const lb = document.getElementById('lightbox');
    if (e.key === 'Escape' && lb && lb.classList.contains('active')) {
        lb.classList.remove('active');
        document.body.style.overflow = '';
    }
});

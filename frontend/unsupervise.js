/**
 * Unsupervised Learning Dashboard
 * Fetches and displays customer segments (K-Means) from FastAPI.
 */

const isLocal = window.location.hostname === '127.0.0.1' || window.location.hostname === 'localhost' || window.location.protocol === 'file:';
const API_BASE = isLocal ? 'http://127.0.0.1:8000' : '';

// Cluster Names mapped by ID (Based on K=3)
const CLUSTER_NAMES = {
    0: "กลุ่มผิวแห้ง/แพ้ง่าย",
    1: "กลุ่มผิวผสม",
    2: "กลุ่มผิวมัน/เป็นสิว"
};

// Colors matching the Python script (K=3: Pink, Blue, Green)
const CLUSTER_COLORS = {
    0: "#f5576c",
    1: "#4facfe",
    2: "#43e97b"
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

    // Observe all sections that start hidden
    document.querySelectorAll('.unsup-section, .sup-graph-card').forEach(el => {
        observer.observe(el);
    });

    // Top section shouldn't wait for scroll since it's at the very top
    const topGrid = document.getElementById('unsup-top-grid');
    if (topGrid) {
        topGrid.classList.add('visible');
        // Add visible to children cards for animation
        topGrid.querySelectorAll('.sup-model-card').forEach((card, index) => {
            setTimeout(() => {
                card.classList.add('visible');
            }, 100 * index);
        });
    }

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

        renderDistributionBars(data.clusters, distContainer);
        renderPersonaCards(data.clusters, personaContainer, observer);

        if (data.anomaly_detection) {
            renderAnomalyData(data.anomaly_detection);
        }
        if (data.pca_analysis) {
            renderPCAMetrics(data.pca_analysis);
        }

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

// ── PCA Metrics ──
function renderPCAMetrics(pca) {
    const container = document.getElementById('pca-metrics');
    if (!container) return;

    container.innerHTML = `
        <div><strong>Total Features:</strong> <br>${pca.total_features}</div>
        <div><strong style="color: #43e97b;">2D Variance:</strong> <br>${pca.variance_explained_2d}%</div>
        <div><strong>90% Variance:</strong> <br>${pca.components_for_90_pct} components</div>
    `;
}

// ── Anomaly Detection ──
function renderAnomalyData(anomaly) {
    const pctEl = document.getElementById('anomaly-pct');
    const countEl = document.getElementById('anomaly-counts');
    const featuresEl = document.getElementById('anomaly-features-list');

    if (pctEl) pctEl.innerText = `${anomaly.anomaly_percentage}%`;
    if (countEl) countEl.innerText = `พบ ${anomaly.n_anomalies} คน จาก ${anomaly.total_samples} คน`;

    if (featuresEl && anomaly.top_anomaly_features) {
        featuresEl.innerHTML = anomaly.top_anomaly_features.slice(0, 5).map(f => {
            return `
                <div style="background: rgba(255,255,255,0.05); padding: 8px 12px; border-radius: 6px; display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 13px; color: #e5e5e7; font-weight: 600;">${f.feature}</span>
                    <div style="display: flex; gap: 12px; font-size: 12px; text-align: right;">
                        <div>
                            <div style="color: #8e8e93; font-size: 10px;">ปกติ</div>
                            <div style="color: #4facfe;">${f.normal_mean.toFixed(2)}</div>
                        </div>
                        <div>
                            <div style="color: #8e8e93; font-size: 10px;">Anomaly</div>
                            <div style="color: #ffcc00;">${f.anomaly_mean.toFixed(2)}</div>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    }
}

// ── Distribution Progress Bars ──
function renderDistributionBars(clusters, container) {
    if (!container) return;
    container.innerHTML = '';

    clusters.forEach(cluster => {
        const color = CLUSTER_COLORS[cluster.cluster_id] || "#ffffff";
        const name = CLUSTER_NAMES[cluster.cluster_id] || `Cluster ${cluster.cluster_id}`;

        const row = document.createElement('div');
        row.className = 'unsup-bar-row';
        row.innerHTML = `
            <div class="unsup-bar-header">
                <div class="unsup-bar-name">
                    <div class="unsup-bar-dot" style="background-color: ${color};"></div>
                    ${name}
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
function renderPersonaCards(clusters, container, observer) {
    if (!container) return;
    container.innerHTML = '';

    clusters.forEach((cluster, index) => {
        const color = CLUSTER_COLORS[cluster.cluster_id] || "#ffffff";
        const name = CLUSTER_NAMES[cluster.cluster_id] || `Cluster ${cluster.cluster_id}`;

        // Merge skin types and concerns, sort by percentage
        let topFeatures = [];
        cluster.top_skin_types.forEach(s => topFeatures.push({ name: s.name, pct: s.pct }));
        cluster.top_concerns.forEach(c => topFeatures.push({ name: c.name, pct: c.pct }));
        topFeatures.sort((a, b) => b.pct - a.pct);

        // Build feature bars (top 4)
        const featuresHTML = topFeatures.slice(0, 4).map(f => {
            const percent = (f.pct * 100).toFixed(0);
            return `
                <div class="unsup-feat-row">
                    <div class="unsup-feat-header">
                        <span class="unsup-feat-name">${f.name}</span>
                        <span style="color: ${color};">${percent}%</span>
                    </div>
                    <div class="unsup-feat-track">
                        <div class="unsup-feat-fill" style="background: ${color}; width: ${percent}%;"></div>
                    </div>
                </div>
            `;
        }).join('');

        // Brief description based on cluster ID
        const descriptions = {
            0: "กลุ่มที่เน้นความชุ่มชื้นเป็นหลัก มีแนวโน้มผิวแห้งและเกิดการแพ้หรือระคายเคืองได้ง่าย",
            1: "กลุ่มที่มีสภาพผิวผสมเป็นส่วนใหญ่ ประสบปัญหารูขุมขนกว้างและสิวอุดตันเป็นบางจุด",
            2: "กลุ่มที่มีความมันบนใบหน้าสูงและมีปัญหาสิวรุมเร้า ทั้งสิวอักเสบ สิวอุดตัน และสิวผด"
        };
        const desc = descriptions[cluster.cluster_id] || "";

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

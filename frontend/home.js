/**
 * Home Dashboard — 4eves Analytics
 * Fetches data from FastAPI backend and renders interactive charts.
 */

// ═══════════════════════════════════════════════════════
//  Configuration
// ═══════════════════════════════════════════════════════

const isLocal = window.location.hostname === '127.0.0.1' || window.location.hostname === 'localhost';
const API_BASE = isLocal ? 'http://127.0.0.1:8000' : '';

const COLORS = [
    '#667eea', '#764ba2', '#f093fb', '#f5576c',
    '#4facfe', '#00f2fe', '#43e97b', '#fa709a',
    '#a8edea', '#fed6e3',
];

const FACTOR_COLORS = [
    '#4facfe', '#667eea', '#43e97b', '#f093fb',
    '#00f2fe', '#f5576c', '#fa709a', '#764ba2',
    '#a8edea', '#fed6e3',
];

const BRAND_COLORS = {
    'Dermavie': '#667eea',
    'Florelle': '#f093fb',
    'Veloura': '#43e97b',
    'Kiyora': '#f5576c',
    'Eastern Belle': '#4facfe',
    'Klinor Lab': '#00f2fe',
    'Aomizu': '#fa709a',
    'DermaCerin': '#a8edea',
    'Glow in Skin': '#fed6e3',
    'Kireiha': '#764ba2',
};


// ═══════════════════════════════════════════════════════
//  Scroll-reveal Observer
// ═══════════════════════════════════════════════════════

const observer = new IntersectionObserver(
    (entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
            }
        });
    },
    { threshold: 0.12, rootMargin: '0px 0px -40px 0px' }
);

document.querySelectorAll('.home-section').forEach((el) => observer.observe(el));

// ═══════════════════════════════════════════════════════
//  Data Fetching
// ═══════════════════════════════════════════════════════

async function loadHomeSummary() {
    try {
        const res = await fetch(`${API_BASE}/api/home-summary`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        renderOverview(data.overview);
        renderDemographics(data.demographics);
        renderBrands(data.top_brands);
        renderFactors(data.factors);
    } catch (err) {
        console.error('Failed to load home summary:', err);
        document.getElementById('overview-grid').innerHTML =
            '<div class="home-error"> เชื่อม API ไม่ติด</div>';
    }
}

// ═══════════════════════════════════════════════════════
//  Renderers
// ═══════════════════════════════════════════════════════

/* ── Animated number counter ─────────────────────────── */

function animateNumber(el, target, duration = 1200) {
    let start = 0;
    const step = (ts) => {
        if (!start) start = ts;
        const progress = Math.min((ts - start) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);   // easeOutCubic
        el.textContent = Math.floor(eased * target);
        if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
}

function renderOverview(ov) {
    animateNumber(document.getElementById('stat-total'), ov.total_respondents);
    animateNumber(document.getElementById('stat-users'), ov.cleansing_water_users);
    animateNumber(document.getElementById('stat-features'), ov.total_features);
    animateNumber(document.getElementById('stat-non-users'), ov.non_cleansing_water_users);
}

/* ── Horizontal bar chart ────────────────────────────── */

function renderBarChart(containerId, dataObj, colorStart = 0) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';

    const entries = Object.entries(dataObj);
    const maxVal = Math.max(...entries.map(([, v]) => v));

    entries.forEach(([label, value], i) => {
        const pct = ((value / maxVal) * 100).toFixed(1);
        const color = COLORS[(colorStart + i) % COLORS.length];

        const row = document.createElement('div');
        row.className = 'bar-row';
        row.style.animationDelay = `${i * 0.08}s`;
        row.innerHTML = `
            <div class="bar-label">${label}</div>
            <div class="bar-track">
                <div class="bar-fill" style="width:0%; background:${color};"></div>
            </div>
            <div class="bar-value">${value}</div>
        `;
        container.appendChild(row);

        setTimeout(() => {
            row.querySelector('.bar-fill').style.width = pct + '%';
        }, 200 + i * 100);
    });
}

/* ── Donut chart (CSS conic-gradient) ────────────────── */

function renderDonut(containerId, dataObj) {
    const container = document.getElementById(containerId);
    const entries = Object.entries(dataObj);
    const total = entries.reduce((sum, [, v]) => sum + v, 0);

    let cumulative = 0;
    const segments = entries.map(([label, value], i) => {
        const pct = (value / total) * 100;
        const start = cumulative;
        cumulative += pct;
        return { label, value, pct, start, color: COLORS[i % COLORS.length] };
    });

    const gradient = segments
        .map((s) => `${s.color} ${s.start}% ${s.start + s.pct}%`)
        .join(', ');

    container.innerHTML = `
        <div class="donut" style="background:conic-gradient(${gradient});">
            <div class="donut-hole">
                <span class="donut-total">${total}</span>
                <span class="donut-label">Total</span>
            </div>
        </div>
        <div class="donut-legend">
            ${segments.map((s) => `
                <div class="legend-item">
                    <span class="legend-dot" style="background:${s.color};"></span>
                    <span class="legend-text">${s.label}</span>
                    <span class="legend-val">${s.value} (${s.pct.toFixed(0)}%)</span>
                </div>
            `).join('')}
        </div>
    `;
}

function renderDemographics(demo) {
    renderBarChart('age-bars', demo.age, 0);
    renderBarChart('skin-bars', demo.skin_type, 3);
    renderDonut('sex-donut', demo.sex);
}

/* ── Top Brands ranking ──────────────────────────────── */

function renderBrands(brands) {
    const container = document.getElementById('brand-list');
    container.innerHTML = '';

    const entries = Object.entries(brands);
    const maxVal = Math.max(...entries.map(([, v]) => v));

    entries.forEach(([name, count], i) => {
        const pct = ((count / maxVal) * 100).toFixed(1);
        const color = BRAND_COLORS[name] || COLORS[i % COLORS.length];

        let labelText = name;
        if (i === 0) labelText = '1 ' + name;
        else if (i === 1) labelText = '2 ' + name;
        else if (i === 2) labelText = '3 ' + name;

        const row = document.createElement('div');
        row.className = 'bar-row';
        row.style.animationDelay = `${i * 0.08}s`;
        row.innerHTML = `
            <div class="bar-label" title="${name}">${labelText}</div>
            <div class="bar-track">
                <div class="bar-fill" style="width:0%; background:${color};"></div>
            </div>
            <div class="bar-value">${count}</div>
        `;
        container.appendChild(row);

        setTimeout(() => {
            row.querySelector('.bar-fill').style.width = pct + '%';
        }, 150 + i * 100);
    });
}

/* ── Purchase-decision factor scores ─────────────────── */

function renderFactors(factors) {
    const container = document.getElementById('factor-list');
    container.innerHTML = '';

    const entries = Object.entries(factors);

    entries.forEach(([name, score], i) => {
        const pct = ((score / 5) * 100).toFixed(1);
        const color = FACTOR_COLORS[i % FACTOR_COLORS.length];

        const row = document.createElement('div');
        row.className = 'bar-row';
        row.style.animationDelay = `${i * 0.08}s`;
        row.innerHTML = `
            <div class="bar-label">${name}</div>
            <div class="bar-track">
                <div class="bar-fill" style="width:0%; background:${color};"></div>
            </div>
            <div class="bar-value">${score.toFixed(2)}</div>
        `;
        container.appendChild(row);

        setTimeout(() => {
            row.querySelector('.bar-fill').style.width = pct + '%';
        }, 150 + i * 80);
    });
}

// ═══════════════════════════════════════════════════════
//  Bootstrap
// ═══════════════════════════════════════════════════════

loadHomeSummary();

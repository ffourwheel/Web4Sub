// ═══════════════════════════════════════════════════════
//  Configuration
// ═══════════════════════════════════════════════════════

const isLocal = window.location.hostname === '127.0.0.1' || window.location.hostname === 'localhost' || window.location.protocol === 'file:';
const API_BASE = isLocal ? 'http://127.0.0.1:8000' : '';

const COLORS = [
    '#f5576c', '#ff8a5c', '#667eea', '#764ba2',
    '#4facfe', '#00f2fe', '#43e97b', '#fa709a',
    '#a8edea', '#fed6e3',
];

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
    { threshold: 0.1, rootMargin: '0px 0px -40px 0px' }
);

document.querySelectorAll('.bi-section').forEach((el) => observer.observe(el));


// ═══════════════════════════════════════════════════════
//  Animated Number Counter
// ═══════════════════════════════════════════════════════

function animateNumber(el, target, suffix = '', duration = 1200) {
    let start = 0;
    const isFloat = target % 1 !== 0;
    const step = (ts) => {
        if (!start) start = ts;
        const progress = Math.min((ts - start) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = isFloat ? (eased * target).toFixed(1) : Math.floor(eased * target);
        el.textContent = current + suffix;
        if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
}


// ═══════════════════════════════════════════════════════
//  Data Fetching
// ═══════════════════════════════════════════════════════

async function loadBusinessInsight() {
    try {
        const res = await fetch(`${API_BASE}/api/business-insight`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        renderKPIs(data.overview);
        renderCoreValues(data.core_values);
        renderFactorComparison(data.factors);
        renderDemographics(data.demographics);
        renderImprovementGaps(data.improvement_gaps);
        renderRecommendations(data);

    } catch (err) {
        console.error('Failed to load business insight:', err);
        document.getElementById('bi-kpi-grid').innerHTML =
            '<div class="bi-error"><i class="ph-fill ph-warning" style="font-size:24px;margin-bottom:8px;display:block;"></i>เชื่อม API ไม่ติด — กรุณาเปิด Backend Server</div>';
    }
}


// ═══════════════════════════════════════════════════════
//  KPI Cards
// ═══════════════════════════════════════════════════════

function renderKPIs(ov) {
    animateNumber(document.getElementById('kpi-total'), ov.total_respondents);
    animateNumber(document.getElementById('kpi-cleansing'), ov.cleansing_water_users);
    animateNumber(document.getElementById('kpi-kiyora'), ov.kiyora_users);
    animateNumber(document.getElementById('kpi-primary'), ov.kiyora_primary_users);
    animateNumber(document.getElementById('kpi-rate'), ov.kiyora_usage_rate, '%');
}


// ═══════════════════════════════════════════════════════
//  Core Values
// ═══════════════════════════════════════════════════════

function renderCoreValues(coreValues) {
    const container = document.getElementById('bi-core-grid');
    if (!coreValues || coreValues.length === 0) {
        container.innerHTML = '<p style="color:#8e8e93;text-align:center;">ไม่มีข้อมูล Core Values</p>';
        return;
    }

    const medals = ['', '', ''];
    const ranks = ['#1 จุดเด่นหลัก', '#2 จุดเด่นรอง', '#3 จุดเด่นเสริม'];

    container.innerHTML = coreValues.map((cv, i) => `
        <div class="bi-core-card">
            <div class="bi-core-rank">${ranks[i] || '#' + (i + 1)}</div>
            <div class="bi-core-name">${medals[i]} ${cv.factor}</div>
            <div class="bi-core-score">${cv.score.toFixed(2)}</div>
            <div class="bi-core-max">/ 5.00</div>
        </div>
    `).join('');
}


// ═══════════════════════════════════════════════════════
//  Factor Comparison (Kiyora vs Overall)
// ═══════════════════════════════════════════════════════

function renderFactorComparison(factors) {
    const container = document.getElementById('bi-factor-bars');
    container.innerHTML = '';

    const kiyora = factors.kiyora;
    const overall = factors.overall;

    // Get all factor names
    const allFactors = [...new Set([...Object.keys(kiyora), ...Object.keys(overall)])];

    allFactors.forEach((name, i) => {
        const kVal = kiyora[name] || 0;
        const oVal = overall[name] || 0;
        const kPct = ((kVal / 5) * 100).toFixed(1);
        const oPct = ((oVal / 5) * 100).toFixed(1);

        const row = document.createElement('div');
        row.className = 'bi-factor-row';
        row.style.animationDelay = `${i * 0.06}s`;
        row.innerHTML = `
            <div class="bi-factor-label">${name}</div>
            <div class="bi-factor-bars">
                <div class="bi-factor-bar-wrap">
                    <div class="bi-factor-bar-track">
                        <div class="bi-factor-bar-fill kiyora" style="width:0%;"></div>
                    </div>
                    <div class="bi-factor-val kiyora">${kVal.toFixed(2)}</div>
                </div>
                <div class="bi-factor-bar-wrap">
                    <div class="bi-factor-bar-track">
                        <div class="bi-factor-bar-fill overall" style="width:0%;"></div>
                    </div>
                    <div class="bi-factor-val overall">${oVal.toFixed(2)}</div>
                </div>
            </div>
        `;
        container.appendChild(row);

        setTimeout(() => {
            row.querySelector('.bi-factor-bar-fill.kiyora').style.width = kPct + '%';
            row.querySelector('.bi-factor-bar-fill.overall').style.width = oPct + '%';
        }, 200 + i * 80);
    });
}


// ═══════════════════════════════════════════════════════
//  Demographics Charts
// ═══════════════════════════════════════════════════════

function renderBarChart(containerId, dataObj, colorStart = 0) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';

    const entries = Object.entries(dataObj);
    if (entries.length === 0) {
        container.innerHTML = '<p style="color:#6c6c70;font-size:13px;">ไม่มีข้อมูล</p>';
        return;
    }
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

function renderDonut(containerId, dataObj) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const entries = Object.entries(dataObj);
    if (entries.length === 0) {
        container.innerHTML = '<p style="color:#6c6c70;font-size:13px;">ไม่มีข้อมูล</p>';
        return;
    }
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
    renderDonut('bi-sex-donut', demo.sex);
    renderBarChart('bi-age-bars', demo.age, 0);
    renderBarChart('bi-skin-bars', demo.skin_type, 3);
    renderBarChart('bi-income-bars', demo.income, 5);
}



// ═══════════════════════════════════════════════════════
//  Improvement Gaps
// ═══════════════════════════════════════════════════════

function renderImprovementGaps(gaps) {
    const container = document.getElementById('bi-gaps');
    if (!container) return;

    if (!gaps || gaps.length === 0) {
        container.innerHTML = `
            <div style="text-align:center;padding:20px;color:#43e97b;font-weight:600;">
                <i class="ph-fill ph-check-circle" style="font-size:24px;margin-right:8px;"></i>
                Kiyora ทำได้ดีกว่าค่าเฉลี่ยตลาดในทุกด้าน! 🎉
            </div>
        `;
        return;
    }

    container.innerHTML = gaps.map(gap => `
        <div class="bi-gap-item">
            <div class="bi-gap-info">
                <div class="bi-gap-factor">${gap.factor}</div>
                <div class="bi-gap-detail">Kiyora: ${gap.kiyora.toFixed(2)} · ตลาด: ${gap.overall.toFixed(2)}</div>
            </div>
            <div class="bi-gap-badge">-${gap.gap.toFixed(2)}</div>
        </div>
    `).join('');
}


// ═══════════════════════════════════════════════════════
//  Strategic Recommendations
// ═══════════════════════════════════════════════════════

function renderRecommendations(data) {
    const container = document.getElementById('bi-recommendations');
    if (!container) return;

    const recs = generateRecommendations(data);
    container.innerHTML = recs.map(rec => `
        <div class="bi-rec-card">
            <div class="bi-rec-icon">
                <i class="ph-fill ${rec.icon}"></i>
            </div>
            <div class="bi-rec-content">
                <div class="bi-rec-title">${rec.title}</div>
                <div class="bi-rec-desc">${rec.desc}</div>
                <span class="bi-rec-tag ${rec.tag}">${rec.tagLabel}</span>
            </div>
        </div>
    `).join('');
}

function generateRecommendations(data) {
    const recs = [];

    // Core values insight
    if (data.core_values && data.core_values.length > 0) {
        const topCV = data.core_values[0];
        recs.push({
            icon: 'ph-star',
            title: `สื่อสาร "${topCV.factor}" เป็นจุดขายหลัก`,
            desc: `ผู้ใช้ Kiyora ให้ความสำคัญกับ ${topCV.factor} สูงสุด (${topCV.score.toFixed(2)}/5.00) ควรใช้เป็น Key Message ในการสื่อสารการตลาดทุกช่องทาง`,
            tag: 'branding',
            tagLabel: 'Branding',
        });
    }

    // Gap analysis
    if (data.improvement_gaps && data.improvement_gaps.length > 0) {
        const topGap = data.improvement_gaps[0];
        recs.push({
            icon: 'ph-target',
            title: `เพิ่มจุดเด่นด้าน "${topGap.factor}"`,
            desc: `Kiyora ยังตามตลาดอยู่ ${topGap.gap.toFixed(2)} คะแนนในด้าน ${topGap.factor} — การปรับปรุงสูตรหรือเพิ่มฟีเจอร์ด้านนี้จะช่วยเพิ่มความสามารถแข่งขัน`,
            tag: 'product',
            tagLabel: 'Product',
        });
    }

    // Demographics insight
    if (data.demographics) {
        const ages = Object.entries(data.demographics.age || {});
        if (ages.length > 0) {
            const topAge = ages.sort((a, b) => b[1] - a[1])[0];
            recs.push({
                icon: 'ph-users-three',
                title: `โฟกัสกลุ่มอายุ "${topAge[0]}" เป็นหลัก`,
                desc: `กลุ่มอายุ ${topAge[0]} เป็นฐานผู้ใช้ Kiyora หลัก (${topAge[1]} คน) — ควรออกแบบแคมเปญให้ตรงกับไลฟ์สไตล์ของกลุ่มนี้`,
                tag: 'marketing',
                tagLabel: 'Marketing',
            });
        }

        const skins = Object.entries(data.demographics.skin_type || {});
        if (skins.length > 0) {
            const topSkin = skins.sort((a, b) => b[1] - a[1])[0];
            recs.push({
                icon: 'ph-sparkle',
                title: `พัฒนาสูตรเฉพาะ "${topSkin[0]}"`,
                desc: `ผู้ใช้ Kiyora ส่วนใหญ่เป็น ${topSkin[0]} (${topSkin[1]} คน) — การพัฒนาสูตรเฉพาะจะเพิ่มความภักดีต่อแบรนด์`,
                tag: 'product',
                tagLabel: 'Product',
            });
        }
    }

    // Market share
    if (data.overview) {
        const rate = data.overview.kiyora_usage_rate;
        if (rate < 20) {
            recs.push({
                icon: 'ph-megaphone',
                title: 'เพิ่ม Brand Awareness ในตลาด Cleansing Water',
                desc: `สัดส่วนผู้ใช้ Kiyora อยู่ที่ ${rate}% ของตลาด — ควรลงทุนใน Influencer Marketing และ Trial Campaign เพื่อขยายฐานลูกค้า`,
                tag: 'marketing',
                tagLabel: 'Marketing',
            });
        }
    }

    // Retention strategy
    recs.push({
        icon: 'ph-heart',
        title: 'สร้าง Loyalty Program เพื่อรักษาฐานลูกค้า',
        desc: 'สร้างโปรแกรมสมาชิกหรือ Subscription Model เพื่อรักษาลูกค้าปัจจุบันและเพิ่มโอกาสในการซื้อซ้ำ',
        tag: 'strategy',
        tagLabel: 'Strategy',
    });

    return recs;
}


// ═══════════════════════════════════════════════════════
//  Lightbox
// ═══════════════════════════════════════════════════════

function openLightbox(imgEl) {
    const lightbox = document.getElementById('bi-lightbox');
    const lightboxImg = document.getElementById('bi-lightbox-img');
    lightboxImg.src = imgEl.src;
    lightbox.classList.add('active');
}

function closeLightbox() {
    const lightbox = document.getElementById('bi-lightbox');
    lightbox.classList.remove('active');
}


// ═══════════════════════════════════════════════════════
//  Bootstrap
// ═══════════════════════════════════════════════════════

loadBusinessInsight();

#!/bin/bash

FIGURES_DIR="/Users/ankanderia/Documents/MBZUAI/Projects/CLIP-DINO/Project Page/Figures"
CONVERTED_DIR="$FIGURES_DIR/converted"
OUTPUT_FILE="/Users/ankanderia/Documents/MBZUAI/Projects/CLIP-DINO/Project Page/index.html"

# Function to convert image to base64 data URI
img_to_base64() {
    local file="$1"
    local ext="${file##*.}"
    local mime="image/$ext"
    if [ "$ext" = "jpg" ]; then mime="image/jpeg"; fi
    echo "data:$mime;base64,$(base64 -i "$file" | tr -d '\n')"
}

# Convert all needed images
echo "Converting images to base64..."
MAIN_FIGURE=$(img_to_base64 "$CONVERTED_DIR/main_figure.png")
TEASER=$(img_to_base64 "$CONVERTED_DIR/teaser_come_2.png")
COMPLEMENTARY=$(img_to_base64 "$CONVERTED_DIR/complementary_features.png")
LAYER_ANALYSIS=$(img_to_base64 "$CONVERTED_DIR/layer_analysis.png")
SEMANTIC_FT=$(img_to_base64 "$CONVERTED_DIR/SEMENTIC_FT.png")
QUALITATIVE=$(img_to_base64 "$CONVERTED_DIR/qualitative_analysis.png")
PERFORMANCE=$(img_to_base64 "$CONVERTED_DIR/performance_breakdown.png")
TASKS=$(img_to_base64 "$CONVERTED_DIR/tasks.png")
DINO_2=$(img_to_base64 "$FIGURES_DIR/dino_2.png")
SIGLIP_2=$(img_to_base64 "$FIGURES_DIR/siglip_2.png")

echo "Building HTML page..."

cat > "$OUTPUT_FILE" << HTMLEOF
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CoME-VL: Scaling Complementary Multi-Encoder Vision-Language</title>
<meta name="description" content="CoME-VL proposes a modular fusion framework integrating contrastive and self-supervised vision encoders for vision-language modeling, achieving state-of-the-art results on multiple benchmarks.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
<style>
:root {
  --primary: #4f6ef7;
  --primary-dark: #3b5de7;
  --primary-light: rgba(79,110,247,0.08);
  --accent: #06b6d4;
  --accent2: #a855f7;
  --accent3: #f472b6;
  --dark: #0a0f1e;
  --dark-card: #111827;
  --gray-900: #1e293b;
  --gray-800: #1f2937;
  --gray-700: #334155;
  --gray-600: #475569;
  --gray-500: #64748b;
  --gray-400: #94a3b8;
  --gray-300: #cbd5e1;
  --gray-200: #e2e8f0;
  --gray-100: #f1f5f9;
  --gray-50: #f8fafc;
  --white: #ffffff;
  --gradient-hero: linear-gradient(160deg, #080d1c 0%, #0f1d3d 35%, #162451 55%, #1a1a3e 80%, #0f0a2e 100%);
  --gradient-accent: linear-gradient(135deg, #4f6ef7 0%, #a855f7 50%, #f472b6 100%);
  --gradient-blue: linear-gradient(135deg, #3b82f6, #1d4ed8);
  --gradient-purple: linear-gradient(135deg, #8b5cf6, #6d28d9);
  --gradient-emerald: linear-gradient(135deg, #10b981, #059669);
  --gradient-amber: linear-gradient(135deg, #f59e0b, #d97706);
  --shadow-sm: 0 1px 3px rgba(0,0,0,0.06);
  --shadow-md: 0 4px 20px rgba(0,0,0,0.06);
  --shadow-lg: 0 10px 40px rgba(0,0,0,0.1);
  --shadow-xl: 0 20px 60px rgba(0,0,0,0.12);
  --shadow-glow: 0 0 40px rgba(79,110,247,0.15);
  --radius: 18px;
  --radius-sm: 12px;
  --radius-xs: 8px;
  --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

* { margin: 0; padding: 0; box-sizing: border-box; }
html { scroll-behavior: smooth; }

body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  color: var(--gray-700);
  background: var(--white);
  line-height: 1.7;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* ===== HERO SECTION ===== */
.hero {
  background: var(--gradient-hero);
  color: var(--white);
  padding: 110px 0 90px;
  position: relative;
  overflow: hidden;
}

.hero::before {
  content: '';
  position: absolute;
  top: -60%;
  left: -30%;
  width: 180%;
  height: 200%;
  background:
    radial-gradient(ellipse at 25% 50%, rgba(79,110,247,0.18) 0%, transparent 55%),
    radial-gradient(ellipse at 75% 30%, rgba(168,85,247,0.12) 0%, transparent 50%),
    radial-gradient(ellipse at 50% 80%, rgba(6,182,212,0.08) 0%, transparent 45%);
  pointer-events: none;
  animation: heroShimmer 15s ease-in-out infinite alternate;
}

@keyframes heroShimmer {
  0% { transform: translateX(0) translateY(0) scale(1); }
  50% { transform: translateX(-3%) translateY(2%) scale(1.02); }
  100% { transform: translateX(2%) translateY(-1%) scale(0.98); }
}

.hero::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 120px;
  background: linear-gradient(to top, var(--white), transparent);
  pointer-events: none;
}

.hero-content {
  max-width: 980px;
  margin: 0 auto;
  padding: 0 32px;
  position: relative;
  z-index: 1;
  text-align: center;
}

.paper-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 100px;
  padding: 8px 22px;
  font-size: 0.82rem;
  font-weight: 500;
  color: rgba(255,255,255,0.7);
  margin-bottom: 36px;
  backdrop-filter: blur(12px);
  letter-spacing: 0.03em;
}
.paper-badge i { color: var(--accent); }

h1.paper-title {
  font-size: 3.2rem;
  font-weight: 800;
  line-height: 1.12;
  letter-spacing: -0.035em;
  margin-bottom: 14px;
  background: linear-gradient(135deg, #fff 0%, #c7d2e8 60%, #94a3b8 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

h1.paper-title .highlight {
  background: var(--gradient-accent);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}

.paper-subtitle {
  font-size: 1.1rem;
  font-weight: 400;
  color: rgba(255,255,255,0.55);
  margin-bottom: 44px;
  max-width: 680px;
  margin-left: auto;
  margin-right: auto;
  line-height: 1.65;
}

.authors {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 6px 22px;
  margin-bottom: 14px;
}
.authors a {
  color: rgba(255,255,255,0.82);
  text-decoration: none;
  font-size: 1rem;
  font-weight: 500;
  transition: var(--transition);
}
.authors a:hover { color: var(--accent); }
.authors .equal {
  font-size: 0.72rem;
  vertical-align: super;
  color: rgba(255,255,255,0.45);
}

.affiliation {
  color: rgba(255,255,255,0.4);
  font-size: 0.88rem;
  margin-bottom: 40px;
  letter-spacing: 0.01em;
}

.hero-buttons {
  display: flex;
  justify-content: center;
  gap: 12px;
  flex-wrap: wrap;
}

.btn {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 13px 26px;
  border-radius: 12px;
  font-size: 0.92rem;
  font-weight: 600;
  text-decoration: none;
  transition: var(--transition);
  cursor: pointer;
  border: none;
  letter-spacing: 0.01em;
}
.btn-primary {
  background: var(--gradient-accent);
  color: var(--white);
  box-shadow: 0 4px 18px rgba(79,110,247,0.35);
}
.btn-primary:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 28px rgba(79,110,247,0.45);
}
.btn-outline {
  background: rgba(255,255,255,0.05);
  color: var(--white);
  border: 1.5px solid rgba(255,255,255,0.15);
  backdrop-filter: blur(10px);
}
.btn-outline:hover {
  background: rgba(255,255,255,0.1);
  border-color: rgba(255,255,255,0.3);
  transform: translateY(-2px);
}

/* ===== STICKY NAV ===== */
.sticky-nav {
  position: sticky;
  top: 0;
  background: rgba(255,255,255,0.88);
  backdrop-filter: blur(24px) saturate(180%);
  border-bottom: 1px solid rgba(0,0,0,0.06);
  z-index: 100;
  transition: var(--transition);
}
.sticky-nav .nav-inner {
  max-width: 1100px;
  margin: 0 auto;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 10px 32px;
  overflow-x: auto;
  scrollbar-width: none;
}
.sticky-nav .nav-inner::-webkit-scrollbar { display: none; }
.sticky-nav a {
  color: var(--gray-500);
  text-decoration: none;
  font-size: 0.84rem;
  font-weight: 500;
  padding: 7px 14px;
  border-radius: 8px;
  transition: var(--transition);
  white-space: nowrap;
}
.sticky-nav a:hover {
  color: var(--primary);
  background: var(--primary-light);
}

/* ===== SECTIONS ===== */
.container {
  max-width: 1100px;
  margin: 0 auto;
  padding: 0 32px;
}

section { padding: 80px 0; }
section:nth-child(even) { background: var(--gray-50); }

.section-label {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-size: 0.78rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--primary);
  margin-bottom: 14px;
}
.section-label i { font-size: 0.68rem; }

h2.section-title {
  font-size: 2.1rem;
  font-weight: 800;
  color: var(--dark);
  letter-spacing: -0.025em;
  margin-bottom: 18px;
  line-height: 1.2;
}

.section-desc {
  font-size: 1.02rem;
  color: var(--gray-500);
  max-width: 740px;
  line-height: 1.78;
  margin-bottom: 40px;
}

/* ===== FIGURE CARDS ===== */
.figure-card {
  background: var(--white);
  border-radius: var(--radius);
  box-shadow: var(--shadow-md);
  overflow: hidden;
  margin-bottom: 32px;
  border: 1px solid rgba(0,0,0,0.04);
  transition: var(--transition);
}
.figure-card:hover {
  box-shadow: var(--shadow-lg);
  transform: translateY(-2px);
}
.figure-card img { width: 100%; display: block; }
.figure-card .caption {
  padding: 18px 26px;
  font-size: 0.88rem;
  color: var(--gray-500);
  border-top: 1px solid var(--gray-100);
  line-height: 1.65;
}
.figure-card .caption strong { color: var(--dark); }

/* ===== ABSTRACT ===== */
.abstract-box {
  background: var(--white);
  border-radius: var(--radius);
  padding: 42px 50px;
  box-shadow: var(--shadow-md);
  border-left: 5px solid;
  border-image: var(--gradient-accent) 1;
  font-size: 1rem;
  line-height: 1.88;
  color: var(--gray-700);
  position: relative;
}
.abstract-box::before {
  content: '\201C';
  position: absolute;
  top: 14px;
  left: 20px;
  font-size: 4.5rem;
  color: var(--primary);
  opacity: 0.1;
  font-family: Georgia, serif;
  line-height: 1;
}

/* ===== KEY HIGHLIGHTS ===== */
.highlights-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 20px;
  margin-top: 44px;
}
.highlight-card {
  background: var(--white);
  border-radius: var(--radius-sm);
  padding: 30px 26px;
  box-shadow: var(--shadow-sm);
  border: 1px solid rgba(0,0,0,0.04);
  transition: var(--transition);
}
.highlight-card:hover {
  transform: translateY(-5px);
  box-shadow: var(--shadow-lg);
}
.highlight-icon {
  width: 50px;
  height: 50px;
  border-radius: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.2rem;
  color: var(--white);
  margin-bottom: 18px;
}
.highlight-card:nth-child(1) .highlight-icon { background: var(--gradient-blue); }
.highlight-card:nth-child(2) .highlight-icon { background: var(--gradient-purple); }
.highlight-card:nth-child(3) .highlight-icon { background: var(--gradient-emerald); }
.highlight-card:nth-child(4) .highlight-icon { background: var(--gradient-amber); }
.highlight-card h3 {
  font-size: 0.98rem;
  font-weight: 700;
  color: var(--dark);
  margin-bottom: 10px;
}
.highlight-card p {
  font-size: 0.88rem;
  color: var(--gray-500);
  line-height: 1.65;
}

/* ===== RESULTS TABLE ===== */
.table-wrapper {
  overflow-x: auto;
  border-radius: var(--radius);
  box-shadow: var(--shadow-md);
  margin: 24px 0;
  border: 1px solid rgba(0,0,0,0.04);
}
table {
  width: 100%;
  border-collapse: collapse;
  background: var(--white);
  font-size: 0.88rem;
}
thead { background: var(--dark); color: var(--white); }
th {
  padding: 14px 18px;
  text-align: center;
  font-weight: 600;
  font-size: 0.82rem;
  letter-spacing: 0.02em;
}
th:first-child { text-align: left; }
td {
  padding: 12px 18px;
  text-align: center;
  border-bottom: 1px solid var(--gray-100);
}
td:first-child {
  text-align: left;
  font-weight: 500;
  color: var(--dark);
}
tbody tr { transition: var(--transition); }
tbody tr:hover { background: var(--primary-light); }
tbody tr:last-child {
  background: rgba(79,110,247,0.06);
  font-weight: 600;
}
tbody tr:last-child td {
  color: var(--primary);
  border-bottom: none;
}

/* ===== METRIC CARDS ===== */
.metrics-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
  margin: 44px 0;
}
.metric-card {
  text-align: center;
  background: var(--white);
  border-radius: var(--radius-sm);
  padding: 30px 20px;
  box-shadow: var(--shadow-sm);
  border: 1px solid rgba(0,0,0,0.04);
  transition: var(--transition);
}
.metric-card:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow-lg);
}
.metric-value {
  font-size: 2.6rem;
  font-weight: 800;
  letter-spacing: -0.04em;
  margin-bottom: 6px;
}
.metric-value.blue { background: var(--gradient-blue); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.metric-value.green { background: var(--gradient-emerald); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.metric-value.purple { background: var(--gradient-purple); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.metric-delta {
  font-size: 0.82rem;
  font-weight: 600;
  color: #059669;
  margin-bottom: 8px;
}
.metric-label {
  font-size: 0.84rem;
  color: var(--gray-500);
  font-weight: 500;
}

/* ===== LAYOUT ===== */
.two-col {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 32px;
  align-items: start;
}
.gallery-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

/* ===== BIBTEX ===== */
.bibtex-box {
  background: var(--gray-900);
  color: #a5f3fc;
  border-radius: var(--radius);
  padding: 30px 34px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  line-height: 1.75;
  overflow-x: auto;
  position: relative;
  border: 1px solid rgba(255,255,255,0.06);
}
.bibtex-box .copy-btn {
  position: absolute;
  top: 14px;
  right: 14px;
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.12);
  color: var(--white);
  border-radius: 8px;
  padding: 8px 14px;
  font-size: 0.78rem;
  cursor: pointer;
  font-family: 'Inter', sans-serif;
  font-weight: 500;
  transition: var(--transition);
}
.bibtex-box .copy-btn:hover { background: rgba(255,255,255,0.18); }

/* ===== FOOTER ===== */
footer {
  background: var(--dark);
  color: rgba(255,255,255,0.5);
  text-align: center;
  padding: 48px 32px 36px;
  font-size: 0.86rem;
  line-height: 1.8;
}
footer a {
  color: var(--accent);
  text-decoration: none;
  transition: var(--transition);
}
footer a:hover { color: var(--accent2); }
footer .footer-label {
  font-weight: 600;
  color: rgba(255,255,255,0.7);
  text-transform: uppercase;
  font-size: 0.72rem;
  letter-spacing: 0.1em;
  margin-bottom: 4px;
}

/* ===== RESPONSIVE ===== */
@media (max-width: 768px) {
  h1.paper-title { font-size: 2rem; }
  h2.section-title { font-size: 1.55rem; }
  .two-col, .gallery-row { grid-template-columns: 1fr; }
  .abstract-box { padding: 28px 24px; }
  section { padding: 50px 0; }
  .hero { padding: 70px 0 50px; }
  .metrics-row { grid-template-columns: repeat(2, 1fr); }
  .highlights-grid { grid-template-columns: 1fr; }
}
</style>
</head>
<body>

<!-- ==================== HERO ==================== -->
<header class="hero" id="top">
  <div class="hero-content">
    <div class="paper-badge"><i class="fas fa-atom"></i> MBZUAI &middot; 2025</div>
    <h1 class="paper-title">
      <span class="highlight">CoME-VL</span>: Scaling Complementary<br>Multi-Encoder Vision-Language
    </h1>
    <p class="paper-subtitle">
      How effectively can we combine two complementary vision encoders for vision-language modeling? We show that fusing SigLIP2 (strong at understanding) with DINOv3 (strong at grounding) through entropy-guided layer selection and orthogonality-regularized mixing yields consistent gains on both tasks.
    </p>
    <div class="authors">
      <a href="#">Ankan Deria<span class="equal">*</span></a>
      <a href="#">Komal Kumar<span class="equal">*</span></a>
      <a href="#">Xilin He</a>
      <a href="#">Imran Razzak</a>
      <a href="#">Hisham Cholakkal</a>
      <a href="#">Fahad Shahbaz Khan</a>
      <a href="#">Salman Khan</a>
    </div>
    <p class="affiliation">Mohamed bin Zayed University of Artificial Intelligence, Abu Dhabi, UAE</p>
    <div class="hero-buttons">
      <a href="COME_VL_Arxiv-2.pdf" class="btn btn-primary"><i class="fas fa-file-pdf"></i> Paper (PDF)</a>
      <a href="https://github.com/mbzuai-oryx/CoME-VL" class="btn btn-outline"><i class="fab fa-github"></i> Code</a>
      <a href="https://huggingface.co/MBZUAI/CoME-VL" class="btn btn-outline"><i class="fas fa-cube"></i> HF Checkpoints</a>
      <a href="#abstract" class="btn btn-outline"><i class="fas fa-book-open"></i> Abstract</a>
    </div>
  </div>
</header>

<!-- ==================== NAV ==================== -->
<nav class="sticky-nav">
  <div class="nav-inner">
    <a href="#abstract">Abstract</a>
    <a href="#teaser">Overview</a>
    <a href="#why-two-encoders">Why Two Encoders?</a>
    <a href="#method">Method</a>
    <a href="#analysis">Analysis</a>
    <a href="#results">Results</a>
    <a href="#qualitative">Qualitative</a>
    <a href="#tasks">Tasks</a>
    <a href="#citation">Citation</a>
  </div>
</nav>

<!-- ==================== ABSTRACT ==================== -->
<section id="abstract">
  <div class="container">
    <div class="section-label"><i class="fas fa-circle"></i> Abstract</div>
    <h2 class="section-title">TL;DR</h2>
    <div class="abstract-box">
      Recent vision-language models (VLMs) typically rely on a single vision encoder trained with contrastive image-text objectives, such as CLIP-style pretraining. While contrastive encoders are effective for cross-modal alignment and retrieval, self-supervised visual encoders often capture richer dense semantics and exhibit stronger robustness on recognition and understanding tasks. In this work, we investigate how to scale the fusion of these complementary visual representations for vision-language modeling. We propose <strong>CoME-VL</strong> (Complementary Multi-Encoder Vision-Language), a modular fusion framework that integrates a contrastively trained vision encoder with DINO as self-supervised encoders. Our approach performs representation-level fusion by <strong>(i)</strong> entropy-guided multi-layer aggregation with orthogonality-constrained projections to reduce redundancy, and <strong>(ii)</strong> RoPE-enhanced cross-attention to align heterogeneous token grids and produce compact fused visual tokens. Extensive experiments across diverse vision-language benchmarks demonstrate that CoME-VL consistently outperforms single-encoder baselines. In particular, we observe an average improvement of <strong>4.9%</strong> on visual understanding tasks and <strong>5.4%</strong> on grounding tasks. Our method achieves state-of-the-art performance on RefCOCO for detection while improving over the baseline by a large margin.
    </div>
    <div class="highlights-grid">
      <div class="highlight-card">
        <div class="highlight-icon"><i class="fas fa-eye"></i></div>
        <h3>SigLIP2 &rarr; Understanding</h3>
        <p>Contrastive encoder excels at semantic understanding &mdash; chart, diagram, table, and document comprehension tasks.</p>
      </div>
      <div class="highlight-card">
        <div class="highlight-icon"><i class="fas fa-crosshairs"></i></div>
        <h3>DINOv3 &rarr; Grounding</h3>
        <p>Self-supervised encoder captures fine-grained spatial cues crucial for pointing, counting, and object localization.</p>
      </div>
      <div class="highlight-card">
        <div class="highlight-icon"><i class="fas fa-chart-line"></i></div>
        <h3>Entropy-Guided Selection</h3>
        <p>Layer-wise entropy analysis reveals which layers are informative &mdash; guiding optimal multi-scale feature selection from each encoder.</p>
      </div>
      <div class="highlight-card">
        <div class="highlight-icon"><i class="fas fa-compress-arrows-alt"></i></div>
        <h3>Orthogonal Fusion + RoPE</h3>
        <p>Orthogonality-regularized mixing removes redundancy; RoPE cross-attention spatially aligns heterogeneous token grids.</p>
      </div>
    </div>
  </div>
</section>

<!-- ==================== TEASER ==================== -->
<section id="teaser">
  <div class="container">
    <div class="section-label"><i class="fas fa-circle"></i> Overview</div>
    <h2 class="section-title">Entropy Analysis &amp; Performance Gains</h2>
    <p class="section-desc">
      SigLIP2 and DINOv3 exhibit distinct entropy profiles across depth &mdash; SigLIP2 maintains high entropy (rich semantic diversity) while DINOv3&rsquo;s deeper layers concentrate on spatially discriminative regions. By leveraging all SigLIP2 layers for understanding and DINOv3 layers 10&ndash;23 for grounding, CoME-VL harnesses the best of both worlds.
    </p>
    <div class="figure-card">
      <img src="${TEASER}" alt="CoME-VL Teaser: Entropy analysis and performance comparison">
      <div class="caption"><strong>Figure 1.</strong> CoME-VL overview &mdash; complementary encoder fusion guided by layer-wise entropy analysis yields consistent improvements across understanding and grounding benchmarks.</div>
    </div>
  </div>
</section>

<!-- ==================== WHY TWO ENCODERS? ==================== -->
<section id="why-two-encoders">
  <div class="container">
    <div class="section-label"><i class="fas fa-circle"></i> Motivation</div>
    <h2 class="section-title">Why Two Encoders?</h2>
    <p class="section-desc">
      Contrastive (SigLIP2) and self-supervised (DINOv3) encoders learn fundamentally different visual representations. SigLIP2 excels at semantic alignment with language, while DINOv3 captures spatially coherent, fine-grained features ideal for grounding. Combining both unlocks complementary strengths.
    </p>
    <div class="figure-card">
      <img src="${COMPLEMENTARY}" alt="Complementary features analysis of SigLIP2 and DINOv3">
      <div class="caption"><strong>Figure 2.</strong> Complementary feature analysis &mdash; SigLIP2 and DINOv3 encode qualitatively different information. Their fusion provides richer visual representations for downstream vision-language tasks.</div>
    </div>
  </div>
</section>

<!-- ==================== METHOD ==================== -->
<section id="method">
  <div class="container">
    <div class="section-label"><i class="fas fa-circle"></i> Architecture</div>
    <h2 class="section-title">CoME-VL Framework</h2>
    <p class="section-desc">
      CoME-VL integrates SigLIP2 and DINOv3 through a modular fusion pipeline: entropy-guided layer selection identifies the most informative features, orthogonality-regularized projections reduce redundancy, and RoPE-enhanced cross-attention aligns heterogeneous token grids into compact visual tokens.
    </p>
    <div class="figure-card">
      <img src="${MAIN_FIGURE}" alt="CoME-VL architecture diagram">
      <div class="caption"><strong>Figure 3.</strong> CoME-VL architecture &mdash; a modular fusion framework that combines contrastive and self-supervised vision encoders through entropy-guided layer selection, orthogonality-constrained projections, and RoPE cross-attention.</div>
    </div>
  </div>
</section>

<!-- ==================== ANALYSIS ==================== -->
<section id="analysis">
  <div class="container">
    <div class="section-label"><i class="fas fa-circle"></i> Analysis</div>
    <h2 class="section-title">Semantic Feature Analysis</h2>
    <p class="section-desc">
      DINOv3 maintains spatially coherent object-level attention throughout, while SigLIP2 transitions from broad spatial coverage in early layers to focused semantic discrimination in deeper layers.
    </p>
    <div class="gallery-row">
      <div class="figure-card">
        <img src="${DINO_2}" alt="DINOv3 layer-wise attention rollout">
        <div class="caption"><strong>DINOv3</strong> &mdash; Layer-wise attention rollout showing spatially coherent object-level focus.</div>
      </div>
      <div class="figure-card">
        <img src="${SIGLIP_2}" alt="SigLIP2 layer-wise attention rollout">
        <div class="caption"><strong>SigLIP2</strong> &mdash; Layer-wise attention rollout showing transition from spatial to semantic focus.</div>
      </div>
    </div>
    <div style="margin-top: 36px">
      <div class="figure-card">
        <img src="${LAYER_ANALYSIS}" alt="Layer-wise entropy analysis">
        <div class="caption"><strong>Figure 4.</strong> Layer-wise entropy analysis &mdash; reveals distinct entropy profiles for SigLIP2 (high entropy, semantic diversity) and DINOv3 (decreasing entropy, spatial concentration), guiding optimal layer selection.</div>
      </div>
    </div>
  </div>
</section>

<!-- ==================== RESULTS ==================== -->
<section id="results">
  <div class="container">
    <div class="section-label"><i class="fas fa-circle"></i> Experiments</div>
    <h2 class="section-title">Benchmark Results</h2>
    <p class="section-desc">
      CoME-VL achieves consistent improvements across diverse vision-language benchmarks, outperforming both single-encoder baselines and competitive multi-encoder alternatives.
    </p>

    <div class="metrics-row">
      <div class="metric-card">
        <div class="metric-value blue">+4.9%</div>
        <div class="metric-delta">&uarr; avg. improvement</div>
        <div class="metric-label">Understanding Tasks</div>
      </div>
      <div class="metric-card">
        <div class="metric-value green">+5.4%</div>
        <div class="metric-delta">&uarr; avg. improvement</div>
        <div class="metric-label">Grounding Tasks</div>
      </div>
      <div class="metric-card">
        <div class="metric-value purple">SOTA</div>
        <div class="metric-delta">State-of-the-Art</div>
        <div class="metric-label">RefCOCO Detection</div>
      </div>
    </div>

    <h3 style="font-size:1.1rem; font-weight:700; color:var(--dark); margin-bottom:12px;">Main Comparison</h3>
    <div class="table-wrapper">
      <table>
        <thead><tr><th>Model</th><th>CountQ</th><th>Pointing</th><th>RealWorldQ</th><th>ChartQ</th><th>DocVQA</th><th>RefCOCO</th></tr></thead>
        <tbody>
          <tr><td>IDEFICS3 8B</td><td>15.87</td><td>28.93</td><td>50.46</td><td>72.17</td><td>64.41</td><td>NS</td></tr>
          <tr><td>InternVL2 8B</td><td>44.10</td><td>57.77</td><td>64.46</td><td>83.30</td><td>91.64</td><td>NS</td></tr>
          <tr><td>MiniCPM 8B</td><td>35.85</td><td>57.03</td><td>67.29</td><td>73.02</td><td>80.10</td><td>NS</td></tr>
          <tr><td>QWEN2-VL 7B</td><td>45.21</td><td>64.25</td><td>61.13</td><td>86.91</td><td>57.42</td><td>NS</td></tr>
          <tr><td>Pixtral-12B</td><td>38.28</td><td>54.00</td><td>63.96</td><td>64.94</td><td>71.66</td><td>NS</td></tr>
          <tr><td>GLM-4V 9B</td><td>40.23</td><td>58.65</td><td>54.12</td><td>84.37</td><td>84.76</td><td>NS</td></tr>
          <tr><td>Molmo</td><td>52.39</td><td>62.41</td><td>66.25</td><td>76.26</td><td>83.31</td><td>53.79 / 68.94</td></tr>
          <tr><td>CoME-VL (Ours)</td><td>57.24</td><td>66.94</td><td>70.75</td><td>81.84</td><td>87.83</td><td>58.56 / 75.94</td></tr>
        </tbody>
      </table>
    </div>

    <div class="two-col" style="margin-top: 48px;">
      <div>
        <h3 style="font-size:1.1rem; font-weight:700; color:var(--dark); margin-bottom:12px;">RefCOCO Benchmark</h3>
        <div class="table-wrapper">
          <table>
            <thead><tr><th>RefCOCO</th><th>val</th><th>testA</th><th>testB</th></tr></thead>
            <tbody>
              <tr><td>Molmo</td><td>0.10</td><td>0.27</td><td>0.27</td></tr>
              <tr><td>Clip-to-DINO</td><td>91.73</td><td>94.06</td><td>88.85</td></tr>
              <tr><td>Qwen-VL</td><td>89.36</td><td>92.23</td><td>85.36</td></tr>
              <tr><td>CoME-VL (Ours)</td><td>92.57</td><td>95.36</td><td>90.51</td></tr>
            </tbody>
          </table>
        </div>
      </div>
      <div>
        <h3 style="font-size:1.1rem; font-weight:700; color:var(--dark); margin-bottom:12px;">Component Contribution Analysis</h3>
        <div class="figure-card">
          <img src="${PERFORMANCE}" alt="Performance breakdown of each component">
          <div class="caption"><strong>Figure 5.</strong> Ablation study showing the contribution of each component to the overall performance gains.</div>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- ==================== QUALITATIVE ==================== -->
<section id="qualitative">
  <div class="container">
    <div class="section-label"><i class="fas fa-circle"></i> Qualitative</div>
    <h2 class="section-title">Qualitative Results</h2>
    <p class="section-desc">
      CoME-VL produces more accurate and spatially precise predictions across diverse vision-language tasks, including visual question answering, object grounding, and document understanding.
    </p>
    <div class="figure-card">
      <img src="${QUALITATIVE}" alt="Qualitative comparison of CoME-VL results">
      <div class="caption"><strong>Figure 6.</strong> Qualitative comparison &mdash; CoME-VL generates more contextually accurate and spatially grounded responses compared to single-encoder baselines.</div>
    </div>
  </div>
</section>

<!-- ==================== TASKS ==================== -->
<section id="tasks">
  <div class="container">
    <div class="section-label"><i class="fas fa-circle"></i> Applications</div>
    <h2 class="section-title">Supported Tasks</h2>
    <p class="section-desc">
      CoME-VL supports a wide range of vision-language tasks, leveraging the complementary strengths of both encoders for superior performance across understanding and grounding domains.
    </p>
    <div class="figure-card">
      <img src="${TASKS}" alt="Downstream tasks supported by CoME-VL">
      <div class="caption"><strong>Figure 7.</strong> CoME-VL supports diverse downstream tasks including visual question answering, document understanding, chart comprehension, counting, pointing, and object detection.</div>
    </div>
  </div>
</section>

<!-- ==================== CITATION ==================== -->
<section id="citation">
  <div class="container">
    <div class="section-label"><i class="fas fa-circle"></i> Citation</div>
    <h2 class="section-title">BibTeX</h2>
    <p class="section-desc">If you find our work useful, please cite:</p>
    <div class="bibtex-box">
      <button class="copy-btn" onclick="navigator.clipboard.writeText(this.parentElement.querySelector('pre').textContent).then(()=>{this.textContent='Copied!';setTimeout(()=>this.textContent='Copy',1500)})"><i class="far fa-copy"></i> Copy</button>
<pre>@article{comevl2025,
  title={CoME-VL: Scaling Complementary Multi-Encoder Vision-Language},
  author={Deria, Ankan and Kumar, Komal and He, Xilin and Razzak, Imran and Cholakkal, Hisham and Khan, Fahad Shahbaz and Khan, Salman},
  journal={arXiv preprint},
  year={2025}
}</pre>
    </div>
  </div>
</section>

<!-- ==================== FOOTER ==================== -->
<footer>
  <div class="container">
    <div style="margin-bottom: 20px;">
      <div class="footer-label">Correspondence</div>
      <a href="mailto:salman.khan@mbzuai.ac.ae">salman.khan@mbzuai.ac.ae</a>
    </div>
    <div style="margin-bottom: 20px;">
      <div class="footer-label">Contact</div>
      <a href="mailto:ankan.deria@mbzuai.ac.ae">ankan.deria@mbzuai.ac.ae</a> &middot;
      <a href="mailto:komal.kumar@mbzuai.ac.ae">komal.kumar@mbzuai.ac.ae</a>
    </div>
    <div style="color: rgba(255,255,255,0.3); font-size: 0.78rem; margin-top: 24px;">
      &copy; 2025 MBZUAI &middot; CoME-VL Project
    </div>
  </div>
</footer>

</body>
</html>
HTMLEOF

echo "Done! Page built at: $OUTPUT_FILE"

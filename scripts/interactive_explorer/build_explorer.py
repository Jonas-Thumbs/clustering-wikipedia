"""Interactive Cluster Explorer Builder

Systematic calibration with CSS coordinate-isolated viewport:
- Fixed non-stretched 1:1 internal buffer matching CSS layout dimensions
- Dynamic client-to-internal scaling factor (scaleX, scaleY) to eradicate CSS flexbox aspect distortion
- Aspect-ratio preserving isometric projection
- Visual reticle and diagnostic telemetry
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
import numpy as np

CORPORA = ["arizona", "computer_science", "literature", "vital_level_3"]

ROOT_DIR = Path(__file__).parent.parent.parent
PROJECTIONS_DIR = ROOT_DIR / "data" / "derived" / "projections"
CLUSTERS_DIR = ROOT_DIR / "data" / "derived" / "clusters"
CLEANED_DIR = ROOT_DIR / "data" / "derived" / "cleaned"
OUT_DIR = Path(__file__).parent / "web"


def build_dataset() -> dict:
    dataset = {}
    for corpus in CORPORA:
        print(f"Loading data for {corpus}...")
        svd_umap = np.load(PROJECTIONS_DIR / f"{corpus}_umap_svd.npy")
        sbert_umap = np.load(PROJECTIONS_DIR / f"{corpus}_umap_sbert.npy")

        snippets: dict[str, str] = {}
        cleaned_path = CLEANED_DIR / f"{corpus}.cleaned.jsonl"
        if cleaned_path.exists():
            with open(cleaned_path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    lead = rec.get("lead_text", "") or ""
                    if len(lead) > 220:
                        lead = lead[:217] + "..."
                    snippets[str(rec["page_id"])] = lead

        clusters_file = CLUSTERS_DIR / f"{corpus}_clusters.csv"
        points = []
        with open(clusters_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, r in enumerate(reader):
                pid = r["page_id"]
                title = r["title"]
                points.append({
                    "id": pid,
                    "title": title,
                    "snippet": snippets.get(pid, ""),
                    "svd_x": round(float(svd_umap[idx, 0]), 4),
                    "svd_y": round(float(svd_umap[idx, 1]), 4),
                    "sbert_x": round(float(sbert_umap[idx, 0]), 4),
                    "sbert_y": round(float(sbert_umap[idx, 1]), 4),
                    "kmeans_svd": int(r["kmeans_svd"]),
                    "kmeans_sbert": int(r["kmeans_sbert"]),
                    "hdbscan_svd": int(r["hdbscan_svd"]),
                    "hdbscan_sbert": int(r["hdbscan_sbert"]),
                })

        summary_path = CLUSTERS_DIR / f"{corpus}_cluster_summary.json"
        summary = {}
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)

        dataset[corpus] = {
            "name": corpus.replace("_", " ").title(),
            "count": len(points),
            "summary": summary,
            "points": points,
        }
    return dataset


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Wikipedia Semantic Clusters - Interactive Explorer</title>
  <style>
    :root {
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --plot-bg: #fdfdfd;
      --border: #e2e8f0;
      --border-dark: #cbd5e1;
      --text: #0f172a;
      --text-muted: #64748b;
      --accent: #2563eb;
      --accent-hover: #1d4ed8;
      --accent-soft: rgba(37, 99, 235, 0.08);
      --noise-color: #94a3b8;
      --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
      --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
      --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.05);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body {
      width: 100%;
      height: 100%;
      overflow: hidden;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    body {
      display: flex;
      flex-direction: column;
    }
    header {
      background: var(--card-bg);
      border-bottom: 1px solid var(--border);
      padding: 12px 24px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      box-shadow: var(--shadow-sm);
      z-index: 10;
      flex-shrink: 0;
    }
    .brand { display: flex; flex-direction: column; }
    .brand h1 { font-size: 1.15rem; font-weight: 700; letter-spacing: -0.02em; color: var(--text); }
    .brand span { font-size: 0.78rem; color: var(--text-muted); }
    .controls { display: flex; flex-wrap: wrap; align-items: center; gap: 12px; }
    .control-group { display: flex; align-items: center; gap: 6px; }
    label { font-size: 0.76rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.04em; }
    select, input {
      background: #ffffff;
      border: 1px solid var(--border-dark);
      color: var(--text);
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 0.85rem;
      outline: none;
      transition: all 0.15s ease;
      box-shadow: var(--shadow-sm);
    }
    select:focus, input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15); }
    .stats-badge {
      font-size: 0.8rem;
      font-weight: 600;
      padding: 5px 12px;
      border-radius: 9999px;
      background: var(--accent-soft);
      color: var(--accent);
      border: 1px solid rgba(37, 99, 235, 0.2);
    }
    main {
      flex: 1;
      display: flex;
      position: relative;
      overflow: hidden;
      min-height: 0; /* Prevents flex overflow */
    }
    #plot-container {
      flex: 1;
      position: relative;
      background: var(--plot-bg);
      overflow: hidden;
      min-width: 0;
      min-height: 0;
    }
    canvas {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      display: block;
      cursor: crosshair;
    }
    #sidebar {
      width: 320px;
      background: var(--card-bg);
      border-left: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      overflow-y: auto;
      padding: 18px;
      gap: 16px;
      box-shadow: -2px 0 6px rgba(0, 0, 0, 0.02);
      flex-shrink: 0;
    }
    .panel-box {
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px;
    }
    .panel-box h3 {
      font-size: 0.76rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 10px;
      font-weight: 700;
    }
    .stat-row { display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 6px; }
    .stat-row span:first-child { color: var(--text-muted); }
    .stat-row span:last-child { font-weight: 600; color: var(--text); }
    #legend-list {
      display: flex;
      flex-direction: column;
      gap: 6px;
      max-height: 280px;
      overflow-y: auto;
      padding-right: 4px;
    }
    .legend-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 0.82rem;
      cursor: pointer;
      padding: 4px 8px;
      border-radius: 6px;
      transition: background 0.12s;
    }
    .legend-item:hover { background: rgba(0, 0, 0, 0.04); }
    .legend-color {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
      margin-right: 8px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .tooltip {
      position: absolute;
      pointer-events: none;
      background: rgba(255, 255, 255, 0.96);
      border: 1px solid var(--border-dark);
      padding: 12px 16px;
      border-radius: 8px;
      font-size: 0.82rem;
      color: var(--text);
      max-width: 340px;
      box-shadow: var(--shadow-lg);
      transition: opacity 0.05s ease;
      opacity: 0;
      z-index: 50;
      backdrop-filter: blur(8px);
    }
    .tooltip-title { font-weight: 700; font-size: 0.95rem; color: var(--accent); margin-bottom: 4px; }
    .tooltip-meta { font-size: 0.75rem; font-weight: 600; color: var(--text-muted); margin-bottom: 6px; }
    .tooltip-snippet { font-size: 0.78rem; line-height: 1.45; color: #334155; border-top: 1px solid var(--border); padding-top: 6px; }
    .reset-btn {
      background: #ffffff;
      border: 1px solid var(--border-dark);
      color: var(--text);
      padding: 6px 12px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.8rem;
      font-weight: 600;
      transition: all 0.15s;
      box-shadow: var(--shadow-sm);
    }
    .reset-btn:hover { background: #f1f5f9; border-color: #94a3b8; }
    #diagnostics {
      font-family: monospace;
      font-size: 0.74rem;
      color: #334155;
      line-height: 1.5;
      padding: 10px;
      background: #f1f5f9;
      border-radius: 6px;
      border: 1px solid var(--border);
    }
    .key-badge {
      display: inline-block;
      background: #e2e8f0;
      color: #0f172a;
      padding: 1px 5px;
      border-radius: 3px;
      font-size: 0.7rem;
      font-weight: 700;
      border: 1px solid #cbd5e1;
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <h1>Wikipedia Semantic Cluster Explorer</h1>
      <span>Interactive UMAP 2D Manifolds with Article Inspection</span>
    </div>

    <div class="controls">
      <div class="control-group">
        <label for="corpus-select">Corpus</label>
        <select id="corpus-select">
          <option value="vital_level_3">Vital Level 3 (999)</option>
          <option value="computer_science">Computer Science (1,597)</option>
          <option value="literature">Literature (1,424)</option>
          <option value="arizona">Arizona (1,455)</option>
        </select>
      </div>

      <div class="control-group">
        <label for="feature-select">Representation</label>
        <select id="feature-select">
          <option value="sbert">Sentence-BERT (384-d)</option>
          <option value="svd">TF-IDF + SVD (100-d)</option>
        </select>
      </div>

      <div class="control-group">
        <label for="algo-select">Clustering</label>
        <select id="algo-select">
          <option value="kmeans">K-Means</option>
          <option value="hdbscan">HDBSCAN</option>
        </select>
      </div>

      <div class="control-group">
        <input type="text" id="search-input" placeholder="Search article title..." />
      </div>

      <button id="reset-zoom" class="reset-btn">Reset View</button>
      <div id="stats-badge" class="stats-badge">999 articles</div>
    </div>
  </header>

  <main>
    <div id="plot-container">
      <canvas id="plot"></canvas>
      <div id="tooltip" class="tooltip">
        <div id="tt-title" class="tooltip-title"></div>
        <div id="tt-meta" class="tooltip-meta"></div>
        <div id="tt-snippet" class="tooltip-snippet"></div>
      </div>
    </div>

    <aside id="sidebar">
      <div class="panel-box">
        <h3>Current Configuration</h3>
        <div class="stat-row"><span>Articles:</span><span id="sb-count">-</span></div>
        <div class="stat-row"><span>Clusters:</span><span id="sb-clusters">-</span></div>
        <div class="stat-row"><span>Noise:</span><span id="sb-noise">-</span></div>
        <div class="stat-row"><span>Silhouette:</span><span id="sb-sil">-</span></div>
      </div>

      <div class="panel-box" style="flex: 1; display: flex; flex-direction: column;">
        <h3>Cluster Legend & Counts</h3>
        <div id="legend-list"></div>
      </div>

      <div class="panel-box">
        <h3>Inspection Diagnostic <span class="key-badge">Press C</span></h3>
        <div id="diagnostics">
          Cursor: (<span id="diag-cx">-</span>, <span id="diag-cy">-</span>)<br>
          Target: (<span id="diag-tx">-</span>, <span id="diag-ty">-</span>)<br>
          Delta: <span id="diag-delta" style="font-weight:700;">-</span> px<br>
          Crosshair: <span id="diag-mode" style="color:var(--text-muted);font-weight:700;">OFF</span>
        </div>
      </div>
    </aside>
  </main>

  <script>
    const DATA = __DATA_PLACEHOLDER__;

    const PALETTE = [
      "#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706",
      "#0891b2", "#db2777", "#65a30d", "#4f46e5", "#ca8a04",
      "#0d9488", "#c026d3", "#16a34a", "#9333ea", "#ea580c",
      "#0284c7", "#e11d48", "#1d4ed8", "#b91c1c", "#047857"
    ];
    const NOISE_COLOR = "#94a3b8";

    // DOM Elements
    const corpusSelect = document.getElementById("corpus-select");
    const featureSelect = document.getElementById("feature-select");
    const algoSelect = document.getElementById("algo-select");
    const searchInput = document.getElementById("search-input");
    const resetZoomBtn = document.getElementById("reset-zoom");
    const statsBadge = document.getElementById("stats-badge");

    const sbCount = document.getElementById("sb-count");
    const sbClusters = document.getElementById("sb-clusters");
    const sbNoise = document.getElementById("sb-noise");
    const sbSil = document.getElementById("sb-sil");
    const legendList = document.getElementById("legend-list");

    const diagCx = document.getElementById("diag-cx");
    const diagCy = document.getElementById("diag-cy");
    const diagTx = document.getElementById("diag-tx");
    const diagTy = document.getElementById("diag-ty");
    const diagDelta = document.getElementById("diag-delta");
    const diagMode = document.getElementById("diag-mode");

    const canvas = document.getElementById("plot");
    const ctx = canvas.getContext("2d");
    const container = document.getElementById("plot-container");
    const tooltip = document.getElementById("tooltip");
    const ttTitle = document.getElementById("tt-title");
    const ttMeta = document.getElementById("tt-meta");
    const ttSnippet = document.getElementById("tt-snippet");

    // Camera view state
    let zoom = 1;
    let cameraX = 0;
    let cameraY = 0;
    let isDragging = false;
    let lastMouseX = 0, lastMouseY = 0;
    let hoveredPoint = null;
    let hoveredScreen = null;
    let mousePos = { x: -1, y: -1 };
    let showCrosshair = false;
    let bounds = { minX: 0, maxX: 1, minY: 0, maxY: 1 };

    window.addEventListener("keydown", (e) => {
      if (e.key === "c" || e.key === "C") {
        if (document.activeElement === searchInput) return;
        showCrosshair = !showCrosshair;
        diagMode.textContent = showCrosshair ? "ON" : "OFF";
        render();
      }
    });

    function getActiveKey() {
      const feat = featureSelect.value;
      const algo = algoSelect.value;
      return `${algo}_${feat}`;
    }

    function calculateBounds(corpusKey, featKey) {
      const pts = DATA[corpusKey].points;
      const xKey = featKey === "svd" ? "svd_x" : "sbert_x";
      const yKey = featKey === "svd" ? "svd_y" : "sbert_y";

      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (const p of pts) {
        const x = p[xKey];
        const y = p[yKey];
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
      const padX = (maxX - minX) * 0.08 || 1;
      const padY = (maxY - minY) * 0.08 || 1;
      bounds = { minX: minX - padX, maxX: maxX + padX, minY: minY - padY, maxY: maxY + padY };
      cameraX = (bounds.minX + bounds.maxX) / 2;
      cameraY = (bounds.minY + bounds.maxY) / 2;
    }

    // Mathematical calibration: get CSS pixel viewport matching container layout
    function getViewport() {
      const rect = container.getBoundingClientRect();
      const cssW = rect.width;
      const cssH = rect.height;

      // Aspect-ratio preserving isometric scale in CSS units
      const spanX = bounds.maxX - bounds.minX;
      const spanY = bounds.maxY - bounds.minY;
      const baseScale = Math.min((cssW * 0.88) / spanX, (cssH * 0.88) / spanY);
      const currentScale = baseScale * zoom;

      return { rect, cssW, cssH, currentScale };
    }

    // Direct client coordinate conversion relative to plot container
    function getCanvasCoordinates(e) {
      const rect = container.getBoundingClientRect();
      return [e.clientX - rect.left, e.clientY - rect.top];
    }

    function syncBufferDimensions() {
      const rect = container.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const targetW = Math.round(rect.width * dpr);
      const targetH = Math.round(rect.height * dpr);
      if (canvas.width !== targetW || canvas.height !== targetH) {
        canvas.width = targetW;
        canvas.height = targetH;
        return true;
      }
      return false;
    }

    function resizeCanvas() {
      syncBufferDimensions();
      render();
    }

    // Track container resizing automatically (handles window resize, header wrap, font load, devtools, sidebar toggle)
    if (window.ResizeObserver) {
      const ro = new ResizeObserver(() => {
        resizeCanvas();
      });
      ro.observe(container);
    }
    window.addEventListener("resize", resizeCanvas);

    function resetView() {
      zoom = 1;
      cameraX = (bounds.minX + bounds.maxX) / 2;
      cameraY = (bounds.minY + bounds.maxY) / 2;
      render();
    }

    resetZoomBtn.addEventListener("click", resetView);

    // Forward transformation: Data (x, y) -> Screen (sx, sy) in CSS pixels
    function toScreen(x, y, vp) {
      if (!vp) vp = getViewport();
      const sx = (vp.cssW / 2) + (x - cameraX) * vp.currentScale;
      const sy = (vp.cssH / 2) - (y - cameraY) * vp.currentScale;
      return [sx, sy];
    }

    // Inverse transformation: Screen (sx, sy) in CSS pixels -> Data (x, y)
    function toData(sx, sy, vp) {
      if (!vp) vp = getViewport();
      const x = cameraX + (sx - vp.cssW / 2) / vp.currentScale;
      const y = cameraY - (sy - vp.cssH / 2) / vp.currentScale;
      return [x, y];
    }

    function updateSidebar() {
      const corpusKey = corpusSelect.value;
      const featKey = featureSelect.value;
      const algoKey = algoSelect.value;
      const clusterKey = getActiveKey();
      const corpusData = DATA[corpusKey];
      const pts = corpusData.points;
      const summary = corpusData.summary.runs ? corpusData.summary.runs[clusterKey] : null;

      sbCount.textContent = pts.length;
      statsBadge.textContent = `${pts.length} articles • ${corpusData.name}`;

      const counts = {};
      pts.forEach(p => {
        const c = p[clusterKey];
        counts[c] = (counts[c] || 0) + 1;
      });

      const clusterIds = Object.keys(counts).map(Number).sort((a, b) => a - b);
      const nonNoiseCount = clusterIds.filter(id => id !== -1).length;
      sbClusters.textContent = nonNoiseCount;

      if (algoKey === "hdbscan") {
        const noiseCount = counts[-1] || 0;
        const pct = ((noiseCount / pts.length) * 100).toFixed(1);
        sbNoise.textContent = `${pct}% (${noiseCount})`;
      } else {
        sbNoise.textContent = "0.0% (None)";
      }

      if (summary && summary.silhouette) {
        sbSil.textContent = summary.silhouette.toFixed(4);
      } else if (summary && summary.silhouette_non_noise) {
        sbSil.textContent = `${summary.silhouette_non_noise.toFixed(4)} (non-noise)`;
      } else {
        sbSil.textContent = "N/A";
      }

      legendList.innerHTML = "";
      clusterIds.forEach(cid => {
        const item = document.createElement("div");
        item.className = "legend-item";
        const color = cid === -1 ? NOISE_COLOR : PALETTE[cid % PALETTE.length];
        const label = cid === -1 ? "Noise / Outlier" : `Cluster ${cid}`;
        item.innerHTML = `
          <div style="display: flex; align-items: center;">
            <span class="legend-color" style="background: ${color}"></span>
            <span style="font-weight: 500;">${label}</span>
          </div>
          <span style="color: var(--text-muted); font-size: 0.78rem; font-weight: 600;">${counts[cid]}</span>
        `;
        legendList.appendChild(item);
      });
    }

    function render() {
      syncBufferDimensions();
      const dpr = window.devicePixelRatio || 1;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const vp = getViewport();
      const corpusKey = corpusSelect.value;
      const featKey = featureSelect.value;
      const clusterKey = getActiveKey();
      const pts = DATA[corpusKey].points;
      const xKey = featKey === "svd" ? "svd_x" : "sbert_x";
      const yKey = featKey === "svd" ? "svd_y" : "sbert_y";

      const query = searchInput.value.trim().toLowerCase();

      // Background grid
      ctx.strokeStyle = "rgba(0, 0, 0, 0.04)";
      ctx.lineWidth = 1;
      const gridSize = 50;
      for (let x = 0; x < vp.cssW; x += gridSize) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, vp.cssH); ctx.stroke();
      }
      for (let y = 0; y < vp.cssH; y += gridSize) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(vp.cssW, y); ctx.stroke();
      }

      const radius = Math.max(3.8, Math.min(8.5, 4.6 * Math.sqrt(zoom)));

      // Draw regular points
      for (let i = 0; i < pts.length; i++) {
        const p = pts[i];
        if (p === hoveredPoint) continue;

        const [sx, sy] = toScreen(p[xKey], p[yKey], vp);
        if (sx < -20 || sx > vp.cssW + 20 || sy < -20 || sy > vp.cssH + 20) continue;

        const cid = p[clusterKey];
        const isNoise = cid === -1;
        const color = isNoise ? NOISE_COLOR : PALETTE[cid % PALETTE.length];
        const matchesSearch = query && p.title.toLowerCase().includes(query);

        ctx.beginPath();
        ctx.arc(sx, sy, matchesSearch ? radius + 2.5 : radius, 0, Math.PI * 2);

        if (query) {
          if (matchesSearch) {
            ctx.fillStyle = "#eab308";
            ctx.shadowColor = "rgba(234, 179, 8, 0.5)";
            ctx.shadowBlur = 8;
          } else {
            ctx.fillStyle = "rgba(203, 213, 225, 0.35)";
            ctx.shadowBlur = 0;
          }
        } else {
          ctx.fillStyle = color;
          ctx.shadowBlur = 0;
        }

        ctx.globalAlpha = isNoise ? 0.45 : (query && !matchesSearch ? 0.25 : 0.88);
        ctx.fill();

        if (matchesSearch) {
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 2.5;
          ctx.stroke();
          ctx.strokeStyle = "#ca8a04";
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }

      // Draw hovered point on top
      if (hoveredPoint && hoveredScreen) {
        const [sx, sy] = hoveredScreen;
        const cid = hoveredPoint[clusterKey];
        const isNoise = cid === -1;
        const color = isNoise ? NOISE_COLOR : PALETTE[cid % PALETTE.length];

        ctx.globalAlpha = 1.0;
        ctx.beginPath();
        ctx.arc(sx, sy, radius + 4.5, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.shadowBlur = 8;
        ctx.shadowColor = "rgba(0,0,0,0.25)";
        ctx.fill();

        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 3;
        ctx.stroke();

        ctx.strokeStyle = "#0f172a";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      // Visual crosshair
      if (showCrosshair && mousePos.x >= 0 && mousePos.y >= 0) {
        ctx.save();
        ctx.strokeStyle = "rgba(37, 99, 235, 0.5)";
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);

        ctx.beginPath();
        ctx.moveTo(mousePos.x - 18, mousePos.y); ctx.lineTo(mousePos.x + 18, mousePos.y);
        ctx.moveTo(mousePos.x, mousePos.y - 18); ctx.lineTo(mousePos.x, mousePos.y + 18);
        ctx.stroke();

        if (hoveredPoint && hoveredScreen) {
          ctx.beginPath();
          ctx.moveTo(mousePos.x, mousePos.y);
          ctx.lineTo(hoveredScreen[0], hoveredScreen[1]);
          ctx.strokeStyle = "rgba(220, 38, 38, 0.7)";
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
        ctx.restore();
      }

      ctx.globalAlpha = 1.0;
      ctx.shadowBlur = 0;
    }

    function findNearestPoint(mx, my) {
      const vp = getViewport();
      const corpusKey = corpusSelect.value;
      const featKey = featureSelect.value;
      const pts = DATA[corpusKey].points;
      const xKey = featKey === "svd" ? "svd_x" : "sbert_x";
      const yKey = featKey === "svd" ? "svd_y" : "sbert_y";

      const radius = Math.max(3.8, Math.min(8.5, 4.6 * Math.sqrt(zoom)));
      const hitRadius = radius + 2.0;
      const maxCapture = 18;

      let directHit = null;
      let directHitDist = Infinity;
      let directHitScreen = null;

      let closestFallback = null;
      let minFallbackDist = maxCapture;
      let closestScreen = null;

      for (let i = 0; i < pts.length; i++) {
        const p = pts[i];
        const [sx, sy] = toScreen(p[xKey], p[yKey], vp);
        const dist = Math.hypot(sx - mx, sy - my);

        if (dist <= hitRadius && dist < directHitDist) {
          directHit = p;
          directHitDist = dist;
          directHitScreen = [sx, sy];
        }

        if (dist < minFallbackDist) {
          minFallbackDist = dist;
          closestFallback = p;
          closestScreen = [sx, sy];
        }
      }

      if (directHit) {
        return { point: directHit, distance: directHitDist, screen: directHitScreen };
      }
      return { point: closestFallback, distance: minFallbackDist, screen: closestScreen };
    }

    canvas.addEventListener("mousedown", (e) => {
      isDragging = true;
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
      canvas.style.cursor = "grabbing";
    });

    window.addEventListener("mouseup", () => {
      isDragging = false;
      canvas.style.cursor = "crosshair";
    });

    canvas.addEventListener("mousemove", (e) => {
      const [mx, my] = getCanvasCoordinates(e);
      mousePos = { x: mx, y: my };

      diagCx.textContent = Math.round(mx);
      diagCy.textContent = Math.round(my);

      if (isDragging) {
        const dx = e.clientX - lastMouseX;
        const dy = e.clientY - lastMouseY;
        lastMouseX = e.clientX;
        lastMouseY = e.clientY;

        const vp = getViewport();
        cameraX -= dx / vp.currentScale;
        cameraY += dy / vp.currentScale;

        render();
        tooltip.style.opacity = "0";
        return;
      }

      const hit = findNearestPoint(mx, my);
      const p = hit.point;
      hoveredScreen = hit.screen;

      if (p !== hoveredPoint) {
        hoveredPoint = p;
        render();
      } else if (showCrosshair) {
        render();
      }

      if (p) {
        const clusterKey = getActiveKey();
        const cid = p[clusterKey];
        const clusterLabel = cid === -1 ? "Noise / Outlier" : `Cluster ${cid}`;

        ttTitle.textContent = p.title;
        ttMeta.textContent = `Page ID: ${p.id} • ${clusterLabel}`;
        ttSnippet.textContent = p.snippet || "No lead paragraph snippet available.";

        diagTx.textContent = Math.round(hit.screen[0]);
        diagTy.textContent = Math.round(hit.screen[1]);
        diagDelta.textContent = hit.distance.toFixed(1);

        const vp = getViewport();
        let tx = mx + 16;
        let ty = my + 16;
        if (tx + 340 > vp.cssW) tx = mx - 350;
        if (ty + 140 > vp.cssH) ty = my - 130;

        tooltip.style.left = `${tx}px`;
        tooltip.style.top = `${ty}px`;
        tooltip.style.opacity = "1";
      } else {
        tooltip.style.opacity = "0";
        diagTx.textContent = "-";
        diagTy.textContent = "-";
        diagDelta.textContent = "-";
      }
    });

    canvas.addEventListener("mouseleave", () => {
      hoveredPoint = null;
      hoveredScreen = null;
      mousePos = { x: -1, y: -1 };
      tooltip.style.opacity = "0";
      diagCx.textContent = "-";
      diagCy.textContent = "-";
      diagTx.textContent = "-";
      diagTy.textContent = "-";
      diagDelta.textContent = "-";
      render();
    });

    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();

      const [mx, my] = getCanvasCoordinates(e);

      const vpBefore = getViewport();
      const dataBefore = toData(mx, my, vpBefore);

      const zoomFactor = e.deltaY < 0 ? 1.12 : 0.89;
      const newZoom = Math.max(0.5, Math.min(45, zoom * zoomFactor));
      if (newZoom === zoom) return;

      zoom = newZoom;

      const vpAfter = getViewport();
      cameraX = dataBefore[0] - (mx - vpAfter.cssW / 2) / vpAfter.currentScale;
      cameraY = dataBefore[1] + (my - vpAfter.cssH / 2) / vpAfter.currentScale;

      render();
    }, { passive: false });

    function onFilterChange() {
      const corpusKey = corpusSelect.value;
      const featKey = featureSelect.value;
      calculateBounds(corpusKey, featKey);
      updateSidebar();
      resizeCanvas();
      resetView();
    }

    corpusSelect.addEventListener("change", onFilterChange);
    featureSelect.addEventListener("change", onFilterChange);
    algoSelect.addEventListener("change", () => {
      updateSidebar();
      render();
    });
    searchInput.addEventListener("input", render);

    // Initial boot
    resizeCanvas();
    onFilterChange();
  </script>
</body>
</html>
"""


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset()

    print("Generating standalone interactive explorer HTML...")
    data_json = json.dumps(dataset, separators=(",", ":"))
    html_content = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", data_json)

    out_file = OUT_DIR / "index.html"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Interactive explorer updated at: {out_file} (Size: {out_file.stat().st_size / (1024*1024):.2f} MB in {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()

const pptx = require("pptxgenjs");
const pres = new pptx();

pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.author = "Math AI Agent";
pres.company = "GitHub: madfish89/Math-agent";
pres.subject = "Math AI Agent — Technical Overview";

// ─── Palette: Midnight Executive ───
const C = {
  navy:   "0D1117",
  surface:"161B22",
  surface2:"21262D",
  border: "30363D",
  text:   "E6EDF3",
  dim:    "8B949E",
  blue:   "58A6FF",
  green:  "3FB950",
  purple: "BC8CFF",
  orange: "D29922",
  red:    "F85149",
  white:  "FFFFFF",
};

// Helper: dark background + footer
function darkSlide(slide, section, pageNum) {
  slide.background = { color: C.navy };
  // Section tag top-right
  slide.addText(section, {
    x: 11.0, y: 0.15, w: 2.0, h: 0.35,
    fontSize: 9, color: C.dim, align: "right", fontFace: "Arial",
  });
  // Page number bottom-right
  slide.addText(`${pageNum} / 10`, {
    x: 11.0, y: 7.05, w: 2.0, h: 0.3,
    fontSize: 9, color: C.dim, align: "right", fontFace: "Arial",
  });
}

// Helper: card
function card(slide, x, y, w, h, opts = {}) {
  slide.addShape("rect", {
    x, y, w, h,
    fill: { color: opts.fill || C.surface },
    line: { color: opts.border || C.border, width: 1 },
    rectRadius: 0.08,
    shadow: { type: "outer", color: "000000", blur: 6, offset: 2, angle: 90, opacity: 0.5 },
  });
}

// ─── SLIDE 1: Title ───
{
  const s = pres.addSlide();
  darkSlide(s, "INTRO", 1);
  // Left: title
  s.addText("Math AI Agent", {
    x: 0.8, y: 1.5, w: 7, h: 0.9,
    fontSize: 44, bold: true, color: C.white, fontFace: "Arial",
  });
  s.addText("An AI-powered math solver that runs entirely in your browser.", {
    x: 0.8, y: 2.5, w: 6.5, h: 0.5,
    fontSize: 16, color: C.dim, fontFace: "Arial",
  });
  // Feature bullets
  const bullets = [
    "GPT-4o-mini solves problems step-by-step with proofs",
    "Python (Pyodide + SymPy) generates 2D & 3D graphs",
    "Every answer is double-checked by re-derivation",
    "Three specialist agents debate complex problems",
    "Desmos calculator for interactive exploration",
    "No backend — runs on GitHub Pages",
  ];
  bullets.forEach((b, i) => {
    s.addText("▸ " + b, {
      x: 0.8, y: 3.2 + i * 0.45, w: 7, h: 0.4,
      fontSize: 13, color: C.text, fontFace: "Arial",
    });
  });
  // Tech badges
  s.addText("Pyodide  •  SymPy  •  Matplotlib  •  OpenAI  •  Desmos API  •  Vanilla JS", {
    x: 0.8, y: 6.2, w: 8, h: 0.35,
    fontSize: 11, color: C.blue, fontFace: "Arial",
  });
  // Right: mock chat bubble
  card(s, 8.8, 1.8, 3.8, 4.5);
  s.addText("🧮", { x: 9.0, y: 2.0, w: 1, h: 0.6, fontSize: 28, color: C.white });
  s.addText("Graph y = sin(x)\nfrom -6 to 6", {
    x: 9.0, y: 2.7, w: 3.4, h: 0.6, fontSize: 12, color: C.blue, fontFace: "Arial",
    fill: { color: C.blue }, fontFace: "Arial",
  });
  s.addText("Answer: sin(x)\nConfidence: 95%\n✓ Double-checked", {
    x: 9.0, y: 4.0, w: 3.4, h: 1.2, fontSize: 11, color: C.text, fontFace: "Arial",
  });
  s.addShape("rect", {
    x: 9.0, y: 5.4, w: 3.4, h: 0.6,
    fill: { color: C.green, transparency: 80 },
    line: { color: C.green, width: 1 },
    rectRadius: 0.05,
  });
  s.addText("📈 Sine wave rendered", {
    x: 9.1, y: 5.45, w: 3.2, h: 0.5, fontSize: 11, color: C.green, fontFace: "Arial",
  });
}

// ─── SLIDE 2: What It Does ───
{
  const s = pres.addSlide();
  darkSlide(s, "OVERVIEW", 2);
  s.addText("What it does", {
    x: 0.8, y: 0.4, w: 6, h: 0.6,
    fontSize: 32, bold: true, color: C.white, fontFace: "Arial",
  });
  const features = [
    { icon: "🧠", title: "Solves it", desc: "GPT-4o-mini breaks problems into steps with a proof for each one.", color: C.blue },
    { icon: "📐", title: "Cites theorems", desc: "Every step names the rule it used — power rule, quadratic formula, etc.", color: C.orange },
    { icon: "📈", title: "Draws it", desc: "Matplotlib in the browser: 2D, 3D surface, contour, histogram, scatter plots.", color: C.green },
    { icon: "🔁", title: "Re-derives it", desc: "SymPy regenerates the answer independently and flags any disagreement.", color: C.purple },
    { icon: "👥", title: "Multi-agent", desc: "Three specialist agents solve in parallel, then a synthesizer combines them.", color: C.red },
    { icon: "📊", title: "Desmos built-in", desc: "Interactive Desmos calculator for exploring functions and parameters.", color: C.blue },
  ];
  features.forEach((f, i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = 0.8 + col * 4.1, y = 1.5 + row * 2.6;
    card(s, x, y, 3.7, 2.2);
    s.addText(f.icon, { x: x + 0.2, y: y + 0.15, w: 0.6, h: 0.6, fontSize: 24 });
    s.addText(f.title, { x: x + 0.2, y: y + 0.8, w: 3.3, h: 0.4, fontSize: 16, bold: true, color: f.color, fontFace: "Arial" });
    s.addText(f.desc, { x: x + 0.2, y: y + 1.3, w: 3.3, h: 0.8, fontSize: 12, color: C.dim, fontFace: "Arial" });
  });
}

// ─── SLIDE 3: Architecture ───
{
  const s = pres.addSlide();
  darkSlide(s, "ARCHITECTURE", 3);
  s.addText("How it works", {
    x: 0.8, y: 0.4, w: 6, h: 0.6,
    fontSize: 32, bold: true, color: C.white, fontFace: "Arial",
  });
  s.addText("A problem flows through four stages. The browser is the runtime — there is no server.", {
    x: 0.8, y: 1.1, w: 10, h: 0.4,
    fontSize: 13, color: C.dim, fontFace: "Arial",
  });
  // Pipeline boxes
  const stages = [
    { label: "Type a problem", sub: "User input", color: C.blue },
    { label: "GPT-4o-mini", sub: "Solve + prove", color: C.purple },
    { label: "Pyodide + SymPy", sub: "Graph + verify", color: C.green },
    { label: "Verified answer", sub: "Steps + graph", color: C.orange },
  ];
  stages.forEach((st, i) => {
    const x = 0.8 + i * 3.1;
    card(s, x, 2.2, 2.7, 1.5, { border: st.color });
    s.addText(st.label, { x: x + 0.15, y: 2.35, w: 2.4, h: 0.5, fontSize: 14, bold: true, color: st.color, fontFace: "Arial" });
    s.addText(st.sub, { x: x + 0.15, y: 2.85, w: 2.4, h: 0.4, fontSize: 11, color: C.dim, fontFace: "Arial" });
    // Arrow
    if (i < 3) {
      s.addText("→", { x: x + 2.7, y: 2.5, w: 0.5, h: 0.6, fontSize: 24, color: C.dim, align: "center" });
    }
  });
  // Bottom callout
  card(s, 0.8, 4.5, 11.7, 2.2);
  s.addText("Everything runs in the browser.", {
    x: 1.0, y: 4.7, w: 8, h: 0.5, fontSize: 18, bold: true, color: C.green, fontFace: "Arial",
  });
  s.addText([
    { text: "No backend. The OpenAI call is the only network request — your key stays in the page. ", options: { color: C.text } },
    { text: "Python is compiled to WebAssembly by Pyodide, running real SymPy and Matplotlib in the browser. ", options: { color: C.text } },
    { text: "Skills are embedded as a Python string inside the HTML — no fetch, no server, works on GitHub Pages.", options: { color: C.blue } },
  ], { x: 1.0, y: 5.3, w: 11.2, h: 1.2, fontSize: 13, color: C.text, fontFace: "Arial", lineSpacingMultiple: 1.4 });
}

// ─── SLIDE 4: Solve Demo ───
{
  const s = pres.addSlide();
  darkSlide(s, "SOLVE", 4);
  s.addText("Step by step. With a proof on every step.", {
    x: 0.8, y: 0.4, w: 10, h: 0.6,
    fontSize: 28, bold: true, color: C.white, fontFace: "Arial",
  });
  s.addText("Each step names the rule it used. Re-derivation in SymPy confirms the final answer.", {
    x: 0.8, y: 1.05, w: 10, h: 0.4, fontSize: 13, color: C.dim, fontFace: "Arial",
  });
  // Left: mock solution card
  card(s, 0.8, 1.8, 5.5, 5.0);
  s.addText("ANSWER", { x: 1.0, y: 1.95, w: 3, h: 0.3, fontSize: 10, bold: true, color: C.green, fontFace: "Arial" });
  s.addText("f′(x) = 12x³ − 4x + 5", { x: 1.0, y: 2.3, w: 5, h: 0.5, fontSize: 20, bold: true, color: C.white, fontFace: "Arial" });
  // Verify badge
  s.addShape("rect", { x: 1.0, y: 2.9, w: 3, h: 0.35, fill: { color: C.green, transparency: 85 }, line: { color: C.green, width: 1 }, rectRadius: 0.04 });
  s.addText("✓ Double-checked with Python", { x: 1.1, y: 2.93, w: 2.8, h: 0.3, fontSize: 11, bold: true, color: C.green, fontFace: "Arial" });
  // Steps
  s.addText("Steps", { x: 1.0, y: 3.5, w: 3, h: 0.3, fontSize: 12, bold: true, color: C.blue, fontFace: "Arial" });
  const steps = [
    "Step 1: Apply the power rule: d/dx(xⁿ) = nxⁿ⁻¹",
    "Step 2: d/dx(3x⁴) = 12x³",
    "Step 3: d/dx(−2x²) = −4x",
    "Step 4: d/dx(5x) = 5 (constant rule)",
    "Step 5: Combine: 12x³ − 4x + 5",
  ];
  steps.forEach((st, i) => {
    card(s, 1.0, 3.9 + i * 0.5, 5.1, 0.42, { fill: C.surface2 });
    s.addText(st, { x: 1.15, y: 3.92 + i * 0.5, w: 4.8, h: 0.38, fontSize: 11, color: C.text, fontFace: "Arial" });
  });
  // Right: confidence + graph
  card(s, 6.7, 1.8, 5.8, 2.3);
  s.addText("Confidence", { x: 6.9, y: 1.95, w: 3, h: 0.3, fontSize: 12, color: C.dim, fontFace: "Arial" });
  // Progress bar
  s.addShape("rect", { x: 6.9, y: 2.4, w: 5.4, h: 0.12, fill: { color: C.surface2 }, rectRadius: 0.03 });
  s.addShape("rect", { x: 6.9, y: 2.4, w: 5.2, h: 0.12, fill: { color: C.green }, rectRadius: 0.03 });
  s.addText("97% — High confidence", { x: 10.5, y: 2.6, w: 2.0, h: 0.3, fontSize: 12, bold: true, color: C.green, align: "right", fontFace: "Arial" });
  s.addText("Python re-derived: 12·x³ − 4·x + 5 ✓", { x: 6.9, y: 3.0, w: 5.4, h: 0.4, fontSize: 12, color: C.text, fontFace: "Arial" });
  // Graph mock
  card(s, 6.7, 4.3, 5.8, 2.5);
  s.addText("📈 Graph: y = 12x³ − 4x + 5", { x: 6.9, y: 4.45, w: 5.4, h: 0.35, fontSize: 12, bold: true, color: C.blue, fontFace: "Arial" });
  s.addShape("rect", { x: 6.9, y: 4.9, w: 5.4, h: 1.7, fill: { color: C.surface2 }, rectRadius: 0.05 });
  s.addText("[ matplotlib cubic curve render ]", { x: 7.5, y: 5.4, w: 4, h: 0.5, fontSize: 12, color: C.dim, align: "center", fontFace: "Arial" });
}

// ─── SLIDE 5: Graphing (2D + 3D) ───
{
  const s = pres.addSlide();
  darkSlide(s, "VISUALIZE", 5);
  s.addText("2D, 3D, and interactive graphs.", {
    x: 0.8, y: 0.4, w: 10, h: 0.6,
    fontSize: 28, bold: true, color: C.white, fontFace: "Arial",
  });
  s.addText("Matplotlib renders in-page via Pyodide. Desmos opens on demand for exploration.", {
    x: 0.8, y: 1.05, w: 10, h: 0.4, fontSize: 13, color: C.dim, fontFace: "Arial",
  });
  // 6 graph type cards in 3x2 grid
  const graphs = [
    { title: "Line Graph", desc: "y = f(x) over a range", icon: "📊", color: C.blue },
    { title: "Multi-Line", desc: "Compare two functions", icon: "📈", color: C.purple },
    { title: "3D Surface", desc: "z = f(x,y) meshgrid plot", icon: "🏔️", color: C.green },
    { title: "Contour Plot", desc: "Filled contour heat map", icon: "🗺️", color: C.orange },
    { title: "Histogram", desc: "Distribution of data", icon: "📊", color: C.red },
    { title: "Scatter Plot", desc: "Point distribution", icon: "🔵", color: C.blue },
  ];
  graphs.forEach((g, i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = 0.8 + col * 4.1, y = 1.8 + row * 2.5;
    card(s, x, y, 3.7, 2.1);
    s.addText(g.icon, { x: x + 0.15, y: y + 0.1, w: 0.7, h: 0.6, fontSize: 24 });
    s.addText(g.title, { x: x + 0.9, y: y + 0.15, w: 2.6, h: 0.4, fontSize: 16, bold: true, color: g.color, fontFace: "Arial" });
    s.addText(g.desc, { x: x + 0.9, y: y + 0.6, w: 2.6, h: 0.4, fontSize: 12, color: C.dim, fontFace: "Arial" });
    // Mock graph area
    s.addShape("rect", { x: x + 0.2, y: y + 1.1, w: 3.3, h: 0.8, fill: { color: C.surface2 }, rectRadius: 0.04 });
    s.addText("[ rendered plot ]", { x: x + 0.2, y: y + 1.25, w: 3.3, h: 0.4, fontSize: 10, color: C.dim, align: "center", fontFace: "Arial" });
  });
}

// ─── SLIDE 6: Multi-Agent ───
{
  const s = pres.addSlide();
  darkSlide(s, "MULTI-AGENT", 6);
  s.addText("Hard problem? Three agents, then a synthesis.", {
    x: 0.8, y: 0.4, w: 10, h: 0.6,
    fontSize: 26, bold: true, color: C.white, fontFace: "Arial",
  });
  s.addText("Three specialist agents solve in PARALLEL (Promise.all). A fourth combines the best of each.", {
    x: 0.8, y: 1.05, w: 10, h: 0.4, fontSize: 13, color: C.dim, fontFace: "Arial",
  });
  // Three agent cards side by side
  const agents = [
    { name: "Agent 1", role: "Algebra & Calculus", color: C.blue, icon: "🧮" },
    { name: "Agent 2", role: "Number Theory & Proofs", color: C.purple, icon: "🔢" },
    { name: "Agent 3", role: "Geometry & Numerics", color: C.green, icon: "📐" },
  ];
  agents.forEach((a, i) => {
    const x = 0.8 + i * 4.1;
    card(s, x, 1.8, 3.7, 2.5, { border: a.color });
    s.addText(a.icon, { x: x + 0.2, y: 1.95, w: 0.8, h: 0.6, fontSize: 28 });
    s.addText(a.name, { x: x + 1.0, y: 1.95, w: 2.5, h: 0.4, fontSize: 16, bold: true, color: a.color, fontFace: "Arial" });
    s.addText(a.role, { x: x + 1.0, y: 2.4, w: 2.5, h: 0.4, fontSize: 12, color: C.dim, fontFace: "Arial" });
    s.addText("Solves independently\nReturns JSON with\nanswer + approach +\nconfidence + concerns", {
      x: x + 0.2, y: 3.0, w: 3.3, h: 1.2, fontSize: 11, color: C.text, fontFace: "Arial", lineSpacingMultiple: 1.3,
    });
  });
  // Arrows down
  s.addText("↓ ↓ ↓", { x: 4.5, y: 4.4, w: 4, h: 0.4, fontSize: 20, color: C.dim, align: "center", fontFace: "Arial" });
  // Synthesis box
  card(s, 2.5, 5.0, 8.3, 1.8, { border: C.orange });
  s.addText("🔀 Synthesizer", { x: 2.7, y: 5.15, w: 4, h: 0.4, fontSize: 16, bold: true, color: C.orange, fontFace: "Arial" });
  s.addText("Combines the three answers into one rigorous solution with proofs, steps, and a unified confidence score. Runs 3x faster with parallel execution.", {
    x: 2.7, y: 5.6, w: 7.8, h: 1.0, fontSize: 12, color: C.text, fontFace: "Arial",
  });
}

// ─── SLIDE 7: Skills (updated) ───
{
  const s = pres.addSlide();
  darkSlide(s, "SKILLS", 7);
  s.addText("Skills", {
    x: 0.8, y: 0.4, w: 6, h: 0.6,
    fontSize: 32, bold: true, color: C.white, fontFace: "Arial",
  });
  s.addText("Each skill is one function in skills.py — embedded in the HTML, loaded by Pyodide.", {
    x: 0.8, y: 1.05, w: 10, h: 0.4, fontSize: 13, color: C.dim, fontFace: "Arial",
  });
  const skills = [
    { num: "1", name: "generate_code", desc: "SymPy code for derivatives, integrals, solves, limits, matrices, Taylor series, ODEs, statistics", color: C.blue },
    { num: "2", name: "present_graph", desc: "2D + 3D graphs: line, multi-line, surface, contour, histogram, scatter (matplotlib)", color: C.green },
    { num: "3", name: "double_check", desc: "Re-executes SymPy, compares ALL numbers in the answer, flags mismatches", color: C.purple },
    { num: "4", name: "three_agents", desc: "Returns prompts for 3 parallel specialists + 1 synthesizer", color: C.orange },
    { num: "5", name: "search_web", desc: "Wikipedia REST API for theorem summaries and source links", color: C.red },
    { num: "6", name: "memory", desc: "localStorage persistence — solved problems survive page reloads", color: C.blue },
    { num: "7", name: "handle()", desc: "Single entry point — returns JSON with graph image if requested", color: C.green },
    { num: "8", name: "Desmos API", desc: "Interactive calculator modal for exploring functions live", color: C.purple },
  ];
  skills.forEach((sk, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.8 + col * 6.0, y = 1.6 + row * 1.35;
    card(s, x, y, 5.7, 1.15, { fill: C.surface });
    // Number badge
    s.addShape("ellipse", { x: x + 0.1, y: y + 0.15, w: 0.5, h: 0.5, fill: { color: sk.color } });
    s.addText(sk.num, { x: x + 0.1, y: y + 0.15, w: 0.5, h: 0.5, fontSize: 14, bold: true, color: C.white, align: "center", valign: "middle", fontFace: "Arial" });
    s.addText(sk.name, { x: x + 0.8, y: y + 0.1, w: 4.7, h: 0.35, fontSize: 14, bold: true, color: sk.color, fontFace: "Arial" });
    s.addText(sk.desc, { x: x + 0.8, y: y + 0.5, w: 4.7, h: 0.5, fontSize: 11, color: C.dim, fontFace: "Arial" });
  });
}

// ─── SLIDE 8: Math Skills (NEW) ───
{
  const s = pres.addSlide();
  darkSlide(s, "MATH", 8);
  s.addText("Math skills", {
    x: 0.8, y: 0.4, w: 6, h: 0.6,
    fontSize: 32, bold: true, color: C.white, fontFace: "Arial",
  });
  s.addText("13 problem types detected and solved with SymPy. New: 3D graphs, limits, matrices, Taylor series, ODEs.", {
    x: 0.8, y: 1.05, w: 11, h: 0.4, fontSize: 13, color: C.dim, fontFace: "Arial",
  });
  const types = [
    "Derivatives", "Integrals (definite & indefinite)", "Solve equations",
    "Limits (as x approaches value)", "Factorials", "Combinations (n choose k)",
    "Permutations (nPk)", "Matrix determinant", "Matrix inverse",
    "Matrix transpose", "Taylor series expansion", "Differential equations (ODE)",
    "Statistics: mean, median, std, variance", "3D surface plots", "Contour plots",
  ];
  // 3 columns x 5 rows
  types.forEach((t, i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = 0.8 + col * 4.1, y = 1.7 + row * 1.0;
    card(s, x, y, 3.8, 0.8, { fill: C.surface });
    s.addText("✓ " + t, { x: x + 0.15, y: y + 0.1, w: 3.5, h: 0.55, fontSize: 12, color: C.text, fontFace: "Arial", valign: "middle" });
  });
}

// ─── SLIDE 9: Tech Stack ───
{
  const s = pres.addSlide();
  darkSlide(s, "STACK", 9);
  s.addText("Tech stack", {
    x: 0.8, y: 0.4, w: 6, h: 0.6,
    fontSize: 32, bold: true, color: C.white, fontFace: "Arial",
  });
  const stacks = [
    { category: "AI", color: C.purple, items: [
      { name: "GPT-4o-mini", desc: "OpenAI chat completions, streaming" },
      { name: "GPT-4o / GPT-4 Turbo", desc: "Optional model selector" },
      { name: "JSON contracts", desc: "Strict JSON for answers, steps, confidence" },
    ]},
    { category: "Python", color: C.green, items: [
      { name: "Pyodide 0.26", desc: "CPython compiled to WebAssembly" },
      { name: "SymPy", desc: "Symbolic algebra, calculus, matrices, ODEs" },
      { name: "NumPy", desc: "Numeric evaluation + meshgrid for 3D" },
      { name: "Matplotlib", desc: "Headless 2D + 3D chart rendering" },
      { name: "mpl_toolkits.mplot3d", desc: "Surface, parametric, contour plots" },
    ]},
    { category: "Web", color: C.blue, items: [
      { name: "Vanilla JS", desc: "No framework, no build step" },
      { name: "Desmos API v1.9", desc: "Interactive graphing calculator" },
      { name: "localStorage", desc: "Cross-session memory of solved problems" },
      { name: "Wikipedia REST API", desc: "Theorem summaries & source links" },
      { name: "GitHub Pages", desc: "Static hosting, .nojekyll, no backend" },
    ]},
  ];
  stacks.forEach((st, i) => {
    const x = 0.8 + i * 4.1;
    card(s, x, 1.3, 3.8, 5.5, { border: st.color });
    s.addText(st.category, { x: x + 0.2, y: 1.45, w: 3.4, h: 0.4, fontSize: 18, bold: true, color: st.color, fontFace: "Arial" });
    st.items.forEach((item, j) => {
      const iy = 2.0 + j * 0.85;
      s.addText(item.name, { x: x + 0.2, y: iy, w: 3.4, h: 0.3, fontSize: 13, bold: true, color: C.text, fontFace: "Arial" });
      s.addText(item.desc, { x: x + 0.2, y: iy + 0.3, w: 3.4, h: 0.4, fontSize: 11, color: C.dim, fontFace: "Arial" });
    });
  });
}

// ─── SLIDE 10: Try It ───
{
  const s = pres.addSlide();
  darkSlide(s, "TRY IT", 10);
  s.addText("Try it now", {
    x: 0.8, y: 0.8, w: 6, h: 0.7,
    fontSize: 36, bold: true, color: C.white, fontFace: "Arial",
  });
  s.addText("Open the link, paste your OpenAI API key, and start solving.", {
    x: 0.8, y: 1.6, w: 8, h: 0.5,
    fontSize: 16, color: C.dim, fontFace: "Arial",
  });
  // URL box
  card(s, 0.8, 2.4, 11.7, 0.8, { border: C.blue });
  s.addText("🔗  https://madfish89.github.io/Math-agent/", {
    x: 1.0, y: 2.5, w: 11, h: 0.5, fontSize: 18, bold: true, color: C.blue, fontFace: "Arial",
  });
  // Example problems
  s.addText("Try these:", { x: 0.8, y: 3.6, w: 4, h: 0.4, fontSize: 16, bold: true, color: C.white, fontFace: "Arial" });
  const examples = [
    "Graph y = sin(x) from -6 to 6",
    "3d graph z = sin(x) + cos(y) from -3 to 3",
    "Find the derivative of x^2 + 3x",
    "Determinant of [[1,2],[3,4]]",
    "Taylor series of cos(x) around 0 order 5",
    "Limit of sin(x)/x as x approaches 0",
    "Mean of [1, 2, 3, 4, 5, 6, 7]",
    "Solve x^2 + 5x + 6 = 0",
  ];
  examples.forEach((ex, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.8 + col * 6.0, y = 4.2 + row * 0.55;
    card(s, x, y, 5.7, 0.42, { fill: C.surface });
    s.addText("▸ " + ex, { x: x + 0.15, y: y + 0.02, w: 5.4, h: 0.38, fontSize: 12, color: C.text, fontFace: "Arial", valign: "middle" });
  });
  // GitHub link
  s.addText("GitHub: madfish89/Math-agent  |  2 files: index.html + skills.py  |  No backend", {
    x: 0.8, y: 6.8, w: 11, h: 0.4, fontSize: 12, color: C.dim, fontFace: "Arial",
  });
}

// ─── Write ───
pres.writeFile({ fileName: "/Users/home/Downloads/Math AI Agent.pptx" }).then(() => {
  console.log("Written: /Users/home/Downloads/Math AI Agent.pptx");
});
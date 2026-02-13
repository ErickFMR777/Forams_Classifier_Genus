import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io
import os
import tempfile
import numpy as np
from datetime import datetime
from fpdf import FPDF
from translations import TRANSLATIONS, LANGUAGES, GENUS_TRANSLATIONS

# ──────────────────────────────────────────────
# Configuración de página
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Clasificador de Foraminíferos",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CSS profesional – dark theme
# ──────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Base ── */
:root { color-scheme: dark !important; }
.stApp, .stApp > * {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
.stApp { background-color: #0d1117 !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    border-right: 1px solid #21262d !important;
}
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] .stSelectbox label {
    color: #58a6ff !important;
    font-weight: 700 !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] li {
    color: #c9d1d9 !important;
}
section[data-testid="stSidebar"] .stExpander {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
}
section[data-testid="stSidebar"] summary span {
    color: #e6edf3 !important;
    font-weight: 600 !important;
}
section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small {
    color: #8b949e !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #21262d !important;
}

/* ── Hero header ── */
.hero-container {
    background: #161b22;
    border-radius: 18px;
    padding: 2.75rem 2.75rem 2.25rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    border: 1px solid #21262d;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: rgba(88,166,255,0.1);
    border: 1px solid rgba(88,166,255,0.25);
    border-radius: 50px;
    padding: 0.35rem 1rem;
    margin-bottom: 1.25rem;
    font-size: 0.72rem;
    font-weight: 700;
    color: #58a6ff !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.hero-badge-dot {
    width: 7px; height: 7px;
    background: #3fb950;
    border-radius: 50%;
    display: inline-block;
}
.hero-title {
    color: #e6edf3 !important;
    font-size: 2rem;
    font-weight: 800;
    margin: 0 0 1rem 0;
    letter-spacing: -0.02em;
    line-height: 1.15;
}
.hero-line {
    width: 50px;
    height: 3px;
    background: #58a6ff;
    border-radius: 3px;
    margin-bottom: 1rem;
}
.hero-subtitle {
    color: #8b949e !important;
    font-size: 0.92rem;
    line-height: 1.7;
    margin: 0 0 1.5rem 0;
    max-width: 680px;
}
.hero-subtitle strong { color: #c9d1d9 !important; }
.hero-subtitle em { color: #c9d1d9 !important; font-style: italic; }

/* ── Hero stats row ── */
.hero-stats {
    display: flex;
    gap: 2.5rem;
    margin-top: 0.5rem;
}
.hero-stat-item { text-align: left; }
.hero-stat-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #e6edf3 !important;
    line-height: 1;
}
.hero-stat-label {
    font-size: 0.68rem;
    font-weight: 600;
    color: #8b949e !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 0.2rem;
}

/* ── Tip banner ── */
.tip-banner {
    background: rgba(210,153,34,0.08);
    border: 1px solid rgba(210,153,34,0.25);
    border-left: 4px solid #d29922;
    border-radius: 10px;
    padding: 0.85rem 1.15rem;
    margin: 0.5rem 0 1.5rem;
    font-size: 0.88rem;
    color: #e3b341 !important;
    line-height: 1.6;
}
.tip-banner strong { font-weight: 700; color: #f0c048 !important; }

/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    margin: 2.25rem 0 1rem;
    padding-bottom: 0.65rem;
    border-bottom: 1px solid #21262d;
}
.section-icon {
    width: 38px; height: 38px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.15rem; flex-shrink: 0;
}
.section-icon.blue { background: rgba(88,166,255,0.12); }
.section-icon.green { background: rgba(63,185,80,0.12); }
.section-icon.purple { background: rgba(188,140,255,0.12); }
.section-icon.orange { background: rgba(210,153,34,0.12); }
.section-icon.red { background: rgba(248,81,73,0.12); }
.section-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #e6edf3 !important;
    margin: 0;
    letter-spacing: -0.01em;
}

/* ── Feature cards row (genus cards) ── */
.genus-card {
    background: #161b22;
    border-radius: 14px;
    padding: 1.5rem 1rem;
    text-align: center;
    border: 1px solid #21262d;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}
.genus-card:hover {
    border-color: #30363d;
    transform: translateY(-2px);
}
.genus-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.genus-card.ammonia::before { background: #3fb950; }
.genus-card.bolivina::before { background: #58a6ff; }
.genus-card.cibicides::before { background: #f85149; }
.genus-card.elphidium::before { background: #d29922; }
.genus-count {
    font-size: 2.25rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.genus-count.ammonia { color: #3fb950 !important; }
.genus-count.bolivina { color: #58a6ff !important; }
.genus-count.cibicides { color: #f85149 !important; }
.genus-count.elphidium { color: #d29922 !important; }
.genus-label {
    font-size: 0.88rem;
    font-weight: 600;
    color: #8b949e !important;
    font-style: italic;
}

/* ── Specimen card ── */
.specimen-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 1.25rem;
    transition: border-color 0.2s ease;
}
.specimen-card:hover { border-color: #30363d; }
.specimen-card-header {
    padding: 0.75rem 1.15rem;
    display: flex;
    align-items: center;
    gap: 0.65rem;
    border-bottom: 1px solid #21262d;
}
.specimen-card-header.ammonia { background: rgba(63,185,80,0.06); }
.specimen-card-header.bolivina { background: rgba(88,166,255,0.06); }
.specimen-card-header.cibicides { background: rgba(248,81,73,0.06); }
.specimen-card-header.elphidium { background: rgba(210,153,34,0.06); }
.specimen-number {
    width: 30px; height: 30px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.8rem;
    color: #ffffff !important; flex-shrink: 0;
}
.specimen-number.ammonia { background: #238636; }
.specimen-number.bolivina { background: #1f6feb; }
.specimen-number.cibicides { background: #da3633; }
.specimen-number.elphidium { background: #9e6a03; }
.specimen-filename {
    font-weight: 600;
    font-size: 0.92rem;
    color: #e6edf3 !important;
}
.specimen-card-body { padding: 1.15rem; }

/* ── Result badge ── */
.result-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.45rem 0.9rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 0.85rem;
}
.result-badge.ammonia { background: rgba(63,185,80,0.1); color: #3fb950 !important; border: 1.5px solid rgba(63,185,80,0.3); }
.result-badge.bolivina { background: rgba(88,166,255,0.1); color: #58a6ff !important; border: 1.5px solid rgba(88,166,255,0.3); }
.result-badge.cibicides { background: rgba(248,81,73,0.1); color: #f85149 !important; border: 1.5px solid rgba(248,81,73,0.3); }
.result-badge.elphidium { background: rgba(210,153,34,0.1); color: #d29922 !important; border: 1.5px solid rgba(210,153,34,0.3); }
.result-confidence {
    font-size: 0.82rem;
    font-weight: 500;
    opacity: 0.75;
}

/* ── Probability bars ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.45rem;
    font-size: 0.85rem;
}
.prob-label {
    min-width: 78px;
    font-weight: 500;
    color: #c9d1d9 !important;
    font-style: italic;
}
.prob-bar-bg {
    flex: 1;
    height: 6px;
    background: #21262d;
    border-radius: 10px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 10px;
    transition: width 0.5s ease;
}
.prob-bar-fill.ammonia { background: #3fb950; }
.prob-bar-fill.bolivina { background: #58a6ff; }
.prob-bar-fill.cibicides { background: #f85149; }
.prob-bar-fill.elphidium { background: #d29922; }
.prob-value {
    min-width: 48px;
    text-align: right;
    font-weight: 600;
    color: #8b949e !important;
    font-size: 0.82rem;
}

/* ── Metric cards ── */
.metric-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.15rem 0.85rem;
    text-align: center;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #58a6ff !important;
    line-height: 1.2;
}
.metric-label {
    font-size: 0.78rem;
    color: #8b949e !important;
    margin-top: 0.25rem;
    font-weight: 500;
}

/* ── Diversity cards ── */
.diversity-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.15rem;
    text-align: center;
}
.diversity-name {
    font-size: 0.82rem;
    font-weight: 600;
    color: #8b949e !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.4rem;
}
.diversity-value {
    font-size: 1.85rem;
    font-weight: 800;
    color: #58a6ff !important;
    line-height: 1;
    margin-bottom: 0.4rem;
}
.diversity-desc {
    font-size: 0.75rem;
    color: #6e7681 !important;
    line-height: 1.4;
}

/* ── Highlighted specimens ── */
.highlight-card {
    border-radius: 12px;
    padding: 0.9rem 1.15rem;
    display: flex;
    align-items: center;
    gap: 0.65rem;
}
.highlight-card.best {
    background: rgba(63,185,80,0.06);
    border: 1px solid rgba(63,185,80,0.2);
}
.highlight-card.worst {
    background: rgba(210,153,34,0.06);
    border: 1px solid rgba(210,153,34,0.2);
}
.highlight-icon { font-size: 1.35rem; flex-shrink: 0; }
.highlight-text {
    font-size: 0.88rem;
    line-height: 1.5;
    color: #c9d1d9 !important;
}
.highlight-text strong { font-weight: 700; color: #e6edf3 !important; }
.highlight-text em { color: #c9d1d9 !important; }

/* ── Download section ── */
.download-section {
    background: #161b22;
    border-radius: 14px;
    padding: 1.75rem;
    text-align: center;
    border: 1px solid #21262d;
    margin-top: 0.5rem;
}
.download-title {
    color: #e6edf3 !important;
    font-size: 1.1rem;
    font-weight: 700;
    margin: 0 0 0.35rem;
}
.download-subtitle {
    color: #8b949e !important;
    font-size: 0.85rem;
    margin: 0 0 0.75rem;
}

/* ── Upload empty state ── */
.empty-upload {
    text-align: center;
    padding: 3rem 1rem;
    background: #161b22;
    border: 2px dashed #30363d;
    border-radius: 14px;
    margin-top: 0.5rem;
}
.empty-upload-icon { font-size: 2.75rem; margin-bottom: 0.65rem; opacity: 0.4; }
.empty-upload-text { font-size: 1rem; font-weight: 500; color: #8b949e !important; margin: 0; }

/* ── Streamlit overrides ── */
.stDataFrame { border-radius: 12px !important; overflow: hidden; }
div[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 0.75rem 0.85rem;
}
div[data-testid="stMetric"] label { color: #8b949e !important; font-weight: 600 !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #58a6ff !important; font-weight: 700 !important; }

.stDownloadButton > button {
    background: #238636 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    padding: 0.65rem 1.75rem !important;
    font-size: 0.95rem !important;
    border: 1px solid rgba(240,246,252,0.1) !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
    background: #2ea043 !important;
    transform: translateY(-1px) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid #21262d;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    font-weight: 600;
    color: #8b949e !important;
    padding: 0.5rem 1rem;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #e6edf3 !important;
    border-bottom: 2px solid #58a6ff;
}

/* ── Footer ── */
.app-footer {
    margin-top: 3rem;
    padding: 1.25rem 0;
    border-top: 1px solid #21262d;
    text-align: center;
    color: #6e7681 !important;
    font-size: 0.78rem;
}

/* ── General visibility ── */
h1, h2, h3, h4, h5, h6 { color: #e6edf3 !important; }
p, span, li, label, div { color: inherit; }
.stAlert p { color: inherit !important; }
</style>
"""

# ──────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────
MODEL_PATH = "forams_model.pth"
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

GENUS_COLORS = {
    "Ammonia": "#3fb950",
    "Bolivina": "#58a6ff",
    "Cibicides": "#f85149",
    "Elphidium": "#d29922",
}

GENUS_CSS_CLASS = {
    "Ammonia": "ammonia",
    "Bolivina": "bolivina",
    "Cibicides": "cibicides",
    "Elphidium": "elphidium",
}


# ──────────────────────────────────────────────
# i18n helper
# ──────────────────────────────────────────────
def t(key: str, lang: str = "es", **kwargs) -> str:
    text = TRANSLATIONS.get(key, {}).get(lang, TRANSLATIONS.get(key, {}).get("es", key))
    if kwargs:
        text = text.format(**kwargs)
    return text


def genus_info(genus: str, lang: str) -> dict:
    return GENUS_TRANSLATIONS.get(genus, {}).get(lang, GENUS_TRANSLATIONS.get(genus, {}).get("es", {}))


# ──────────────────────────────────────────────
# Carga del modelo (caché)
# ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    model = models.resnet18()
    model.fc = torch.nn.Linear(512, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, classes


# ──────────────────────────────────────────────
# Preprocesado e inferencia
# ──────────────────────────────────────────────
def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def predict(image: Image.Image, model, classes):
    transform = get_transform()
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    probs = probabilities.numpy()
    sorted_idx = np.argsort(probs)[::-1]
    return [{"clase": classes[idx], "probabilidad": float(probs[idx])} for idx in sorted_idx]


def predict_batch(images, model, classes):
    transform = get_transform()
    tensors = torch.stack([transform(img) for img in images])
    with torch.no_grad():
        outputs = model(tensors)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    all_results = []
    for i in range(len(images)):
        probs = probabilities[i].numpy()
        sorted_idx = np.argsort(probs)[::-1]
        all_results.append([{"clase": classes[idx], "probabilidad": float(probs[idx])} for idx in sorted_idx])
    return all_results


# ──────────────────────────────────────────────
# Estadísticos
# ──────────────────────────────────────────────
def compute_statistics(all_results, filenames, classes):
    n = len(all_results)
    top_classes = [r[0]["clase"] for r in all_results]
    top_confidences = [r[0]["probabilidad"] for r in all_results]

    genus_count = {}
    genus_confidences = {}
    for cls_name, conf in zip(top_classes, top_confidences):
        genus_count[cls_name] = genus_count.get(cls_name, 0) + 1
        genus_confidences.setdefault(cls_name, []).append(conf)

    confs = np.array(top_confidences)
    global_stats = {
        "total": n,
        "media_confianza": float(np.mean(confs)),
        "std_confianza": float(np.std(confs, ddof=1)) if n > 1 else 0.0,
        "min_confianza": float(np.min(confs)),
        "max_confianza": float(np.max(confs)),
        "mediana_confianza": float(np.median(confs)),
    }

    idx_max = int(np.argmax(confs))
    idx_min = int(np.argmin(confs))
    global_stats["max_specimen"] = {"idx": idx_max, "archivo": filenames[idx_max], "clase": top_classes[idx_max], "confianza": top_confidences[idx_max]}
    global_stats["min_specimen"] = {"idx": idx_min, "archivo": filenames[idx_min], "clase": top_classes[idx_min], "confianza": top_confidences[idx_min]}
    global_stats["genero_dominante"] = max(genus_count, key=genus_count.get)
    global_stats["generos_detectados"] = len(genus_count)

    per_genus = {}
    for genus in sorted(genus_count.keys()):
        gc = np.array(genus_confidences[genus])
        per_genus[genus] = {
            "n": genus_count[genus],
            "porcentaje": genus_count[genus] / n * 100,
            "media_confianza": float(np.mean(gc)),
            "std_confianza": float(np.std(gc, ddof=1)) if len(gc) > 1 else 0.0,
            "min_confianza": float(np.min(gc)),
            "max_confianza": float(np.max(gc)),
            "mediana_confianza": float(np.median(gc)),
        }

    proportions = np.array([genus_count.get(c, 0) / n for c in genus_count])
    proportions = proportions[proportions > 0]
    shannon = float(-np.sum(proportions * np.log(proportions))) if len(proportions) > 1 else 0.0
    simpson = float(1 - np.sum(proportions ** 2)) if len(proportions) > 1 else 0.0
    s = len(proportions)
    pielou = float(shannon / np.log(s)) if s > 1 else 0.0

    return global_stats, per_genus, {"shannon": shannon, "simpson": simpson, "pielou": pielou}


# ──────────────────────────────────────────────
# Generación de PDF
# ──────────────────────────────────────────────
class ReportPDF(FPDF):
    def __init__(self, lang="es", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lang = lang

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, t("pdf_title", self._lang), align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 5, f"{t('pdf_generated', self._lang)}: {datetime.now().strftime('%d/%m/%Y %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.line(10, self.get_y() + 2, 200, self.get_y() + 2)
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"{t('pdf_page', self._lang)} {self.page_no()}/{{nb}}", align="C")


def generate_pdf(specimens, global_stats, per_genus, diversity, lang="es"):
    pdf = ReportPDF(lang=lang)
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Resumen general
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, t("pdf_general_summary", lang), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"{t('pdf_total_specimens', lang)}: {len(specimens)}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"{t('pdf_genera_detected', lang)}: {global_stats['generos_detectados']} de 4", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"{t('pdf_dominant_genus', lang)}: {global_stats['genero_dominante']}", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(90, 7, t("col_genus", lang), border=1, align="C")
    pdf.cell(45, 7, t("pdf_quantity", lang), border=1, align="C")
    pdf.cell(45, 7, t("pdf_percentage", lang), border=1, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for genus_name, stats in per_genus.items():
        pdf.cell(90, 7, genus_name, border=1, align="C")
        pdf.cell(45, 7, str(stats["n"]), border=1, align="C")
        pdf.cell(45, 7, f"{stats['porcentaje']:.1f}%", border=1, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # Estadísticos de confianza
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, t("pdf_confidence_stats", lang), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, t("pdf_global_confidence", lang), new_x="LMARGIN", new_y="NEXT")

    col_w = [50, 30, 30, 30, 30, 30]
    headers = [t("pdf_metric", lang), t("stat_mean", lang), t("stat_median", lang), t("stat_std", lang), t("stat_min", lang), t("stat_max", lang)]
    pdf.set_font("Helvetica", "B", 9)
    for w, h in zip(col_w, headers):
        pdf.cell(w, 7, h, border=1, align="C")
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(col_w[0], 7, t("pdf_global", lang), border=1, align="C")
    pdf.cell(col_w[1], 7, f"{global_stats['media_confianza']:.1%}", border=1, align="C")
    pdf.cell(col_w[2], 7, f"{global_stats['mediana_confianza']:.1%}", border=1, align="C")
    pdf.cell(col_w[3], 7, f"{global_stats['std_confianza']:.1%}", border=1, align="C")
    pdf.cell(col_w[4], 7, f"{global_stats['min_confianza']:.1%}", border=1, align="C")
    pdf.cell(col_w[5], 7, f"{global_stats['max_confianza']:.1%}", border=1, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, t("pdf_confidence_per_genus", lang), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 9)
    for w, h in zip(col_w, headers):
        pdf.cell(w, 7, h, border=1, align="C")
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for genus_name, stats in per_genus.items():
        pdf.cell(col_w[0], 7, genus_name, border=1, align="C")
        pdf.cell(col_w[1], 7, f"{stats['media_confianza']:.1%}", border=1, align="C")
        pdf.cell(col_w[2], 7, f"{stats['mediana_confianza']:.1%}", border=1, align="C")
        pdf.cell(col_w[3], 7, f"{stats['std_confianza']:.1%}", border=1, align="C")
        pdf.cell(col_w[4], 7, f"{stats['min_confianza']:.1%}", border=1, align="C")
        pdf.cell(col_w[5], 7, f"{stats['max_confianza']:.1%}", border=1, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Especímenes destacados
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, t("pdf_notable_specimens", lang), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    mx = global_stats["max_specimen"]
    mn = global_stats["min_specimen"]
    pdf.cell(0, 6, f"{t('pdf_highest_confidence', lang)}: {mx['archivo']} - {mx['clase']} ({mx['confianza']:.1%})", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"{t('pdf_lowest_confidence', lang)}: {mn['archivo']} - {mn['clase']} ({mn['confianza']:.1%})", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Índices de diversidad
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, t("pdf_diversity_title", lang), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(70, 7, t("pdf_index", lang), border=1, align="C")
    pdf.cell(40, 7, t("pdf_value", lang), border=1, align="C")
    pdf.cell(80, 7, t("pdf_interpretation", lang), border=1, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)

    sh = diversity["shannon"]
    sh_interp = t("pdf_diversity_low", lang) if sh < 0.5 else (t("pdf_diversity_moderate", lang) if sh < 1.0 else t("pdf_diversity_high", lang))
    pdf.cell(70, 7, "Shannon (H')", border=1, align="C")
    pdf.cell(40, 7, f"{sh:.4f}", border=1, align="C")
    pdf.cell(80, 7, sh_interp, border=1, align="C", new_x="LMARGIN", new_y="NEXT")

    si = diversity["simpson"]
    si_interp = t("pdf_diversity_low", lang) if si < 0.3 else (t("pdf_diversity_moderate", lang) if si < 0.6 else t("pdf_diversity_high", lang))
    pdf.cell(70, 7, "Simpson (1-D)", border=1, align="C")
    pdf.cell(40, 7, f"{si:.4f}", border=1, align="C")
    pdf.cell(80, 7, si_interp, border=1, align="C", new_x="LMARGIN", new_y="NEXT")

    pi = diversity["pielou"]
    pi_interp = t("pdf_evenness_low", lang) if pi < 0.4 else (t("pdf_evenness_moderate", lang) if pi < 0.7 else t("pdf_evenness_high", lang))
    pielou_label = "Equitatividad Pielou (J)" if lang == "es" else ("Pielou Evenness (J)" if lang == "en" else "Équitabilité Pielou (J)")
    pdf.cell(70, 7, pielou_label, border=1, align="C")
    pdf.cell(40, 7, f"{pi:.4f}", border=1, align="C")
    pdf.cell(80, 7, pi_interp, border=1, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # Tabla resumen
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, t("pdf_summary_table", lang), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    col_widths = [15, 75, 50, 40]
    headers_tbl = ["#", t("col_file", lang), t("col_genus", lang), t("col_confidence", lang)]
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(15, 76, 129)
    pdf.set_text_color(255, 255, 255)
    for w, h in zip(col_widths, headers_tbl):
        pdf.cell(w, 7, h, border=1, align="C", fill=True)
    pdf.ln()
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 9)
    for i, sp in enumerate(specimens):
        top = sp["results"][0]
        # alternate row color
        if i % 2 == 0:
            pdf.set_fill_color(248, 250, 252)
        else:
            pdf.set_fill_color(255, 255, 255)
        pdf.cell(col_widths[0], 7, str(i + 1), border=1, align="C", fill=True)
        pdf.cell(col_widths[1], 7, sp["filename"][:40], border=1, align="L", fill=True)
        pdf.cell(col_widths[2], 7, top["clase"], border=1, align="C", fill=True)
        pdf.cell(col_widths[3], 7, f"{top['probabilidad']:.1%}", border=1, align="C", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # Detalle por espécimen
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, t("pdf_specimen_detail", lang), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    for i, sp in enumerate(specimens):
        if pdf.get_y() > 220:
            pdf.add_page()

        pdf.set_font("Helvetica", "B", 11)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 8, f"  {t('specimen', lang)} {i + 1}: {sp['filename']}", fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            thumb = sp["image"].copy()
            thumb.thumbnail((150, 150))
            thumb.save(tmp, format="JPEG", quality=85)
            tmp_path = tmp.name
        try:
            img_y = pdf.get_y()
            pdf.image(tmp_path, x=12, y=img_y, w=35)
        finally:
            os.unlink(tmp_path)

        table_x = 55
        pdf.set_xy(table_x, pdf.get_y())
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(60, 6, t("col_genus", lang), border=1, align="C")
        pdf.cell(40, 6, t("pdf_probability", lang), border=1, align="C", new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "", 9)
        for r in sp["results"]:
            pdf.set_x(table_x)
            is_top = r == sp["results"][0]
            if is_top:
                pdf.set_font("Helvetica", "B", 9)
            pdf.cell(60, 6, r["clase"], border=1, align="C")
            pdf.cell(40, 6, f"{r['probabilidad']:.1%}", border=1, align="C", new_x="LMARGIN", new_y="NEXT")
            if is_top:
                pdf.set_font("Helvetica", "", 9)

        top = sp["results"][0]
        pdf.set_x(table_x)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(100, 7, f"{t('pdf_classification', lang)}: {top['clase']} ({top['probabilidad']:.1%})", new_x="LMARGIN", new_y="NEXT")

        min_y = img_y + 38
        if pdf.get_y() < min_y:
            pdf.set_y(min_y)
        pdf.ln(4)

    return pdf.output()


# ──────────────────────────────────────────────
# Componentes UI
# ──────────────────────────────────────────────
def render_section_header(icon, title, color_class="blue"):
    st.markdown(
        f"""
        <div class="section-header">
            <div class="section-icon {color_class}">{icon}</div>
            <div><p class="section-title">{title}</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_genus_cards(genus_count):
    cols = st.columns(4)
    for i, (genus_name, _) in enumerate(GENUS_COLORS.items()):
        count = genus_count.get(genus_name, 0)
        css = GENUS_CSS_CLASS[genus_name]
        with cols[i]:
            st.markdown(
                f"""
                <div class="genus-card {css}">
                    <div class="genus-count {css}">{count}</div>
                    <div class="genus-label">{genus_name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_specimen_card(idx, filename, image, results, lang="es"):
    top = results[0]
    css = GENUS_CSS_CLASS.get(top["clase"], "ammonia")

    st.markdown(
        f"""
        <div class="specimen-card">
            <div class="specimen-card-header {css}">
                <div class="specimen-number {css}">{idx + 1}</div>
                <span class="specimen-filename">{filename}</span>
            </div>
            <div class="specimen-card-body">
        """,
        unsafe_allow_html=True,
    )

    col_img, col_res = st.columns([1, 1.2])

    with col_img:
        st.image(image, use_container_width=True)

    with col_res:
        pct = top["probabilidad"] * 100
        st.markdown(
            f"""
            <div class="result-badge {css}">
                <span>{top['clase']}</span>
                <span class="result-confidence">&middot; {pct:.1f}% {t('confidence', lang)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        bars_html = ""
        for r in results:
            r_css = GENUS_CSS_CLASS.get(r["clase"], "ammonia")
            r_pct = r["probabilidad"] * 100
            bars_html += f"""
            <div class="prob-row">
                <span class="prob-label">{r['clase']}</span>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill {r_css}" style="width:{r_pct:.1f}%"></div>
                </div>
                <span class="prob-value">{r_pct:.1f}%</span>
            </div>
            """
        st.markdown(bars_html, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)


def render_metric_cards(labels, values):
    cols = st.columns(len(labels))
    for col, label, val in zip(cols, labels, values):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{val:.1%}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_diversity_cards(diversity, lang):
    div_data = [
        ("Shannon (H')", diversity["shannon"], t("shannon_desc", lang)),
        ("Simpson (1-D)", diversity["simpson"], t("simpson_desc", lang)),
        ("Pielou (J)", diversity["pielou"], t("pielou_desc", lang)),
    ]
    cols = st.columns(3)
    for col, (name, val, desc) in zip(cols, div_data):
        with col:
            st.markdown(
                f"""
                <div class="diversity-card">
                    <div class="diversity-name">{name}</div>
                    <div class="diversity-value">{val:.4f}</div>
                    <div class="diversity-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_highlighted_specimens(global_stats, lang):
    mx = global_stats["max_specimen"]
    mn = global_stats["min_specimen"]
    cols = st.columns(2)
    with cols[0]:
        st.markdown(
            f"""
            <div class="highlight-card best">
                <div class="highlight-icon">🏆</div>
                <div class="highlight-text">
                    <strong>{t('stats_highest', lang)}</strong><br>
                    {mx['archivo']} &mdash; <em>{mx['clase']}</em> ({mx['confianza']:.1%})
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f"""
            <div class="highlight-card worst">
                <div class="highlight-icon">⚠️</div>
                <div class="highlight-text">
                    <strong>{t('stats_lowest', lang)}</strong><br>
                    {mn['archivo']} &mdash; <em>{mn['clase']}</em> ({mn['confianza']:.1%})
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if "lang" not in st.session_state:
        st.session_state.lang = "es"

    with st.sidebar:
        lang_label = list(LANGUAGES.keys())[list(LANGUAGES.values()).index(st.session_state.lang)]
        selected_lang_label = st.selectbox(
            "🌐 Idioma / Language / Langue",
            options=list(LANGUAGES.keys()),
            index=list(LANGUAGES.keys()).index(lang_label),
        )
        st.session_state.lang = LANGUAGES[selected_lang_label]
        st.markdown("---")

    lang = st.session_state.lang

    # Hero
    desc_text = t("app_description", lang)
    # Convert markdown bold/italic to HTML for the hero
    import re
    desc_html = desc_text
    desc_html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', desc_html)
    desc_html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', desc_html)
    desc_html = desc_html.replace("  \n", "<br>")

    title_text = t("app_title", lang).replace("🔬 ", "")
    badge_text = t("sidebar_genera", lang).upper() if lang != "en" else "MICROPALEONTOLOGY"
    st.markdown(
        f"""
        <div class="hero-container">
            <div class="hero-badge"><span class="hero-badge-dot"></span> {badge_text}</div>
            <h1 class="hero-title">{title_text.upper()}</h1>
            <div class="hero-line"></div>
            <p class="hero-subtitle">{desc_html}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tip
    tip_text = t("tip_crop", lang)
    tip_html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', tip_text)
    st.markdown(f'<div class="tip-banner">{tip_html}</div>', unsafe_allow_html=True)

    # Modelo
    try:
        model, classes = load_model()
    except Exception as e:
        st.error(f"{t('model_error', lang)}: {e}")
        return

    # Sidebar géneros
    with st.sidebar:
        st.header(t("sidebar_genera", lang))
        for genus_name in GENUS_COLORS:
            gi = genus_info(genus_name, lang)
            with st.expander(f"🦠 {genus_name}"):
                st.markdown(gi.get("descripcion", ""))
                st.caption(f"{t('habitat_label', lang)}: {gi.get('habitat', 'N/A')}")

    # Upload
    uploaded_files = st.file_uploader(
        t("uploader_label", lang),
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        accept_multiple_files=True,
        help=t("uploader_help", lang),
    )

    if not uploaded_files:
        st.markdown(
            f"""
            <div style="text-align:center; padding:3rem 1rem; background:#161b22;
                        border:2px dashed #30363d; border-radius:14px; margin-top:0.5rem;">
                <div style="font-size:3.5rem; margin-bottom:0.75rem; opacity:0.4;">📷</div>
                <p style="font-size:1.05rem; font-weight:500; margin:0; color:#8b949e !important;">
                    {t('upload_prompt', lang).replace('👆 ', '')}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Footer even when empty
        st.markdown(f'<div class="app-footer">{t("footer_text", lang)}</div>', unsafe_allow_html=True)
        return

    images = []
    filenames = []
    for f in uploaded_files:
        img = Image.open(f).convert("RGB")
        images.append(img)
        filenames.append(f.name)

    st.success(t("specimens_loaded", lang, n=len(images)))

    with st.spinner(t("classifying", lang, n=len(images))):
        all_results = predict_batch(images, model, classes)

    # ── Resumen ──
    render_section_header("📊", t("summary_header", lang).replace("📊 ", ""), "blue")

    genus_count = {}
    for res in all_results:
        genus_count[res[0]["clase"]] = genus_count.get(res[0]["clase"], 0) + 1

    render_genus_cards(genus_count)

    # ── Detalle ──
    render_section_header("🔍", t("detail_header", lang).replace("🔍 ", ""), "purple")

    for idx in range(len(images)):
        render_specimen_card(idx, filenames[idx], images[idx], all_results[idx], lang)

    # ── Tabla resumen ──
    render_section_header("📋", t("summary_table_header", lang).replace("📋 ", ""), "green")

    table_data = []
    for idx in range(len(images)):
        top = all_results[idx][0]
        table_data.append({
            "#": idx + 1,
            t("col_file", lang): filenames[idx],
            t("col_genus", lang): top["clase"],
            t("col_confidence", lang): f"{top['probabilidad']:.1%}",
        })
    st.dataframe(table_data, use_container_width=True, hide_index=True)

    # ── Estadísticos ──
    global_stats, per_genus, diversity = compute_statistics(all_results, filenames, classes)

    render_section_header("📈", t("stats_header", lang).replace("📈 ", ""), "orange")

    tab_labels = [
        t("global_confidence", lang).replace("**", ""),
        t("confidence_per_genus", lang).replace("**", ""),
        t("highlighted_specimens", lang).replace("**", ""),
        t("diversity_header", lang).replace("**", ""),
    ]
    tab_global, tab_genus, tab_highlighted, tab_diversity = st.tabs(tab_labels)

    with tab_global:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        stat_labels = [t("stat_mean", lang), t("stat_median", lang), t("stat_std", lang), t("stat_min", lang), t("stat_max", lang)]
        stat_values = [
            global_stats["media_confianza"],
            global_stats["mediana_confianza"],
            global_stats["std_confianza"],
            global_stats["min_confianza"],
            global_stats["max_confianza"],
        ]
        render_metric_cards(stat_labels, stat_values)

    with tab_genus:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        genus_table = []
        for genus, stats in per_genus.items():
            genus_table.append({
                t("col_genus", lang): genus,
                "N": stats["n"],
                "%": f"{stats['porcentaje']:.1f}%",
                t("stat_mean", lang): f"{stats['media_confianza']:.1%}",
                t("stat_median", lang): f"{stats['mediana_confianza']:.1%}",
                t("stat_std", lang): f"{stats['std_confianza']:.1%}",
                t("stat_min_short", lang): f"{stats['min_confianza']:.1%}",
                t("stat_max_short", lang): f"{stats['max_confianza']:.1%}",
            })
        st.dataframe(genus_table, use_container_width=True, hide_index=True)

    with tab_highlighted:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        render_highlighted_specimens(global_stats, lang)

    with tab_diversity:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        render_diversity_cards(diversity, lang)

    # ── PDF ──
    render_section_header("📄", t("pdf_header", lang).replace("📄 ", ""), "red")

    st.markdown(
        f"""
        <div class="download-section">
            <p class="download-title">📄 {t('pdf_header', lang).replace('📄 ', '')}</p>
            <p class="download-subtitle">{t('pdf_download_subtitle', lang)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    specimens_for_pdf = []
    for idx in range(len(images)):
        specimens_for_pdf.append({
            "filename": filenames[idx],
            "image": images[idx],
            "results": all_results[idx],
        })

    pdf_bytes = bytes(generate_pdf(specimens_for_pdf, global_stats, per_genus, diversity, lang=lang))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    col_dl = st.columns([1, 2, 1])
    with col_dl[1]:
        st.download_button(
            label=t("pdf_button", lang),
            data=pdf_bytes,
            file_name=f"informe_foraminiferos_{timestamp}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )

    # Footer
    st.markdown(f'<div class="app-footer">{t("footer_text", lang)}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

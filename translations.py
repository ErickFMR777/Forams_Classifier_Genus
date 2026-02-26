# ──────────────────────────────────────────────
# Traducciones – ES / EN / FR
# ──────────────────────────────────────────────

LANGUAGES = {
    "Español": "es",
    "English": "en",
    "Français": "fr",
}

TRANSLATIONS = {
    # ── Títulos y encabezados ──
    "page_title": {
        "es": "Clasificador de Foraminíferos",
        "en": "Foraminifera Classifier",
        "fr": "Classificateur de Foraminifères",
    },
    "app_title": {
        "es": "🔬 Clasificador de Foraminíferos Bentónicos",
        "en": "🔬 Benthic Foraminifera Classifier",
        "fr": "🔬 Classificateur de Foraminifères Benthiques",
    },
    "app_description": {
        "es": (
            "Clasifica fotografías de microscopía óptica y electrónica de barrido (SEM) de "
            "foraminíferos bentónicos en **4 géneros**: *Ammonia*, *Bolivina*, *Cibicides* y "
            "*Elphidium*.  \n"
            "Puedes subir **múltiples especímenes** a la vez en formato **JPG, PNG, BMP, TIFF o WebP** "
            "y descargar el informe de clasificación en PDF."
        ),
        "en": (
            "Classifies optical microscopy and scanning electron microscopy (SEM) photographs of "
            "benthic foraminifera into **4 genera**: *Ammonia*, *Bolivina*, *Cibicides* and "
            "*Elphidium*.  \n"
            "You can upload **multiple specimens** at once in **JPG, PNG, BMP, TIFF or WebP** format "
            "and download the classification report as PDF."
        ),
        "fr": (
            "Classifie les photographies de microscopie optique et électronique à balayage (MEB) de "
            "foraminifères benthiques en **4 genres** : *Ammonia*, *Bolivina*, *Cibicides* et "
            "*Elphidium*.  \n"
            "Vous pouvez télécharger **plusieurs spécimens** à la fois au format **JPG, PNG, BMP, "
            "TIFF ou WebP** et exporter le rapport de classification en PDF."
        ),
    },
    "tip_crop": {
        "es": (
            "💡 **Recomendación:** Para obtener mejores resultados, recorta cada imagen de modo que el "
            "foraminífero ocupe la mayor parte del encuadre. Imágenes con mucho fondo vacío o con el "
            "espécimen muy pequeño pueden reducir la confianza y precisión de la clasificación."
        ),
        "en": (
            "💡 **Tip:** For best results, crop each image so the foraminifera fills most of the frame. "
            "Images with too much empty background or a very small specimen may reduce classification "
            "confidence and accuracy."
        ),
        "fr": (
            "💡 **Conseil :** Pour de meilleurs résultats, recadrez chaque image de sorte que le "
            "foraminifère occupe la majeure partie du cadre. Les images avec trop de fond vide ou un "
            "spécimen trop petit peuvent réduire la confiance et la précision de la classification."
        ),
    },

    # ── Sidebar ──
    "language": {
        "es": "🌐 Idioma",
        "en": "🌐 Language",
        "fr": "🌐 Langue",
    },
    "sidebar_genera": {
        "es": "📚 Géneros",
        "en": "📚 Genera",
        "fr": "📚 Genres",
    },
    "habitat_label": {
        "es": "📍 Hábitat",
        "en": "📍 Habitat",
        "fr": "📍 Habitat",
    },

    # ── Uploader ──
    "uploader_label": {
        "es": "Sube una o más imágenes de foraminíferos",
        "en": "Upload one or more foraminifera images",
        "fr": "Téléchargez une ou plusieurs images de foraminifères",
    },
    "uploader_help": {
        "es": "Puedes seleccionar varias imágenes a la vez. Formatos: JPG, PNG, BMP, TIFF, WebP",
        "en": "You can select multiple images at once. Formats: JPG, PNG, BMP, TIFF, WebP",
        "fr": "Vous pouvez sélectionner plusieurs images à la fois. Formats : JPG, PNG, BMP, TIFF, WebP",
    },
    "upload_prompt": {
        "es": "👆 Sube una o más imágenes para comenzar la clasificación.",
        "en": "👆 Upload one or more images to start classification.",
        "fr": "👆 Téléchargez une ou plusieurs images pour commencer la classification.",
    },
    "specimens_loaded": {
        "es": "📷 {n} espécimen(es) cargado(s)",
        "en": "📷 {n} specimen(s) loaded",
        "fr": "📷 {n} spécimen(s) chargé(s)",
    },
    "classifying": {
        "es": "Clasificando {n} espécimen(es)...",
        "en": "Classifying {n} specimen(s)...",
        "fr": "Classification de {n} spécimen(s)...",
    },

    # ── Resultados ──
    "summary_header": {
        "es": "📊 Resumen de clasificación",
        "en": "📊 Classification Summary",
        "fr": "📊 Résumé de la classification",
    },
    "detail_header": {
        "es": "🔍 Detalle por espécimen",
        "en": "🔍 Specimen Detail",
        "fr": "🔍 Détail par spécimen",
    },
    "specimen": {
        "es": "Espécimen",
        "en": "Specimen",
        "fr": "Spécimen",
    },
    "genus_identified": {
        "es": "Género identificado",
        "en": "Identified genus",
        "fr": "Genre identifié",
    },
    "confidence": {
        "es": "confianza",
        "en": "confidence",
        "fr": "confiance",
    },

    # ── Tabla resumen ──
    "summary_table_header": {
        "es": "📋 Tabla resumen",
        "en": "📋 Summary Table",
        "fr": "📋 Tableau récapitulatif",
    },
    "pdf_summary_table": {
        "es": "Tabla resumen",
        "en": "Summary Table",
        "fr": "Tableau récapitulatif",
    },
    "col_file": {
        "es": "Archivo",
        "en": "File",
        "fr": "Fichier",
    },
    "col_genus": {
        "es": "Género",
        "en": "Genus",
        "fr": "Genre",
    },
    "col_confidence": {
        "es": "Confianza",
        "en": "Confidence",
        "fr": "Confiance",
    },

    # ── Estadísticos ──
    "stats_header": {
        "es": "📈 Estadísticos",
        "en": "📈 Statistics",
        "fr": "📈 Statistiques",
    },
    "global_confidence": {
        "es": "**Confianza global (predicción principal)**",
        "en": "**Global confidence (top prediction)**",
        "fr": "**Confiance globale (prédiction principale)**",
    },
    "stat_mean": {
        "es": "Media",
        "en": "Mean",
        "fr": "Moyenne",
    },
    "stat_median": {
        "es": "Mediana",
        "en": "Median",
        "fr": "Médiane",
    },
    "stat_std": {
        "es": "Desv. Est.",
        "en": "Std. Dev.",
        "fr": "Éc. type",
    },
    "stat_min": {
        "es": "Mínimo",
        "en": "Minimum",
        "fr": "Minimum",
    },
    "stat_max": {
        "es": "Máximo",
        "en": "Maximum",
        "fr": "Maximum",
    },
    "stat_min_short": {
        "es": "Mín.",
        "en": "Min.",
        "fr": "Min.",
    },
    "stat_max_short": {
        "es": "Máx.",
        "en": "Max.",
        "fr": "Max.",
    },
    "confidence_per_genus": {
        "es": "**Confianza por género**",
        "en": "**Confidence per genus**",
        "fr": "**Confiance par genre**",
    },
    "col_n": {
        "es": "N",
        "en": "N",
        "fr": "N",
    },
    "col_pct": {
        "es": "%",
        "en": "%",
        "fr": "%",
    },
    "highlighted_specimens": {
        "es": "**Especímenes destacados**",
        "en": "**Notable specimens**",
        "fr": "**Spécimens remarquables**",
    },
    "highest_confidence": {
        "es": "🔝 Mayor confianza: **{file}** — {cls} ({conf})",
        "en": "🔝 Highest confidence: **{file}** — {cls} ({conf})",
        "fr": "🔝 Confiance la plus élevée : **{file}** — {cls} ({conf})",
    },
    "lowest_confidence": {
        "es": "⚠️ Menor confianza: **{file}** — {cls} ({conf})",
        "en": "⚠️ Lowest confidence: **{file}** — {cls} ({conf})",
        "fr": "⚠️ Confiance la plus faible : **{file}** — {cls} ({conf})",
    },
    "diversity_header": {
        "es": "**Índices de diversidad ecológica**",
        "en": "**Ecological diversity indices**",
        "fr": "**Indices de diversité écologique**",
    },
    "shannon_desc": {
        "es": "Mide la incertidumbre; 0 = homogéneo",
        "en": "Measures uncertainty; 0 = homogeneous",
        "fr": "Mesure l'incertitude ; 0 = homogène",
    },
    "simpson_desc": {
        "es": "Prob. de que 2 individuos sean de distinto género",
        "en": "Prob. that 2 individuals belong to different genera",
        "fr": "Prob. que 2 individus soient de genres différents",
    },
    "pielou_desc": {
        "es": "Equitatividad; 1 = distribución uniforme",
        "en": "Evenness; 1 = uniform distribution",
        "fr": "Équitabilité ; 1 = distribution uniforme",
    },

    # ── PDF ──
    "pdf_header": {
        "es": "📄 Descargar informe PDF",
        "en": "📄 Download PDF Report",
        "fr": "📄 Télécharger le rapport PDF",
    },
    "pdf_button": {
        "es": "⬇️ Descargar informe en PDF",
        "en": "⬇️ Download PDF report",
        "fr": "⬇️ Télécharger le rapport PDF",
    },
    "pdf_title": {
        "es": "Informe de Clasificación de Foraminíferos",
        "en": "Foraminifera Classification Report",
        "fr": "Rapport de Classification des Foraminifères",
    },
    "pdf_generated": {
        "es": "Generado",
        "en": "Generated",
        "fr": "Généré",
    },
    "pdf_page": {
        "es": "Página",
        "en": "Page",
        "fr": "Page",
    },
    "pdf_general_summary": {
        "es": "Resumen General",
        "en": "General Summary",
        "fr": "Résumé Général",
    },
    "pdf_total_specimens": {
        "es": "Total de especímenes analizados",
        "en": "Total specimens analyzed",
        "fr": "Total de spécimens analysés",
    },
    "pdf_genera_detected": {
        "es": "Géneros detectados",
        "en": "Genera detected",
        "fr": "Genres détectés",
    },
    "pdf_dominant_genus": {
        "es": "Género dominante",
        "en": "Dominant genus",
        "fr": "Genre dominant",
    },
    "pdf_quantity": {
        "es": "Cantidad",
        "en": "Quantity",
        "fr": "Quantité",
    },
    "pdf_percentage": {
        "es": "Porcentaje",
        "en": "Percentage",
        "fr": "Pourcentage",
    },
    "pdf_confidence_stats": {
        "es": "Estadísticos de Confianza",
        "en": "Confidence Statistics",
        "fr": "Statistiques de Confiance",
    },
    "pdf_global_confidence": {
        "es": "Confianza global (predicción principal)",
        "en": "Global confidence (top prediction)",
        "fr": "Confiance globale (prédiction principale)",
    },
    "pdf_metric": {
        "es": "Métrica",
        "en": "Metric",
        "fr": "Métrique",
    },
    "pdf_global": {
        "es": "Global",
        "en": "Global",
        "fr": "Global",
    },
    "pdf_confidence_per_genus": {
        "es": "Confianza por género",
        "en": "Confidence per genus",
        "fr": "Confiance par genre",
    },
    "pdf_notable_specimens": {
        "es": "Especímenes destacados",
        "en": "Notable specimens",
        "fr": "Spécimens remarquables",
    },
    "pdf_highest_confidence": {
        "es": "Mayor confianza",
        "en": "Highest confidence",
        "fr": "Confiance la plus élevée",
    },
    "pdf_lowest_confidence": {
        "es": "Menor confianza",
        "en": "Lowest confidence",
        "fr": "Confiance la plus faible",
    },
    "pdf_diversity_title": {
        "es": "Índices de Diversidad",
        "en": "Diversity Indices",
        "fr": "Indices de Diversité",
    },
    "pdf_index": {
        "es": "Índice",
        "en": "Index",
        "fr": "Indice",
    },
    "pdf_value": {
        "es": "Valor",
        "en": "Value",
        "fr": "Valeur",
    },
    "pdf_interpretation": {
        "es": "Interpretación",
        "en": "Interpretation",
        "fr": "Interprétation",
    },
    "pdf_diversity_low": {
        "es": "Diversidad baja",
        "en": "Low diversity",
        "fr": "Diversité faible",
    },
    "pdf_diversity_moderate": {
        "es": "Diversidad moderada",
        "en": "Moderate diversity",
        "fr": "Diversité modérée",
    },
    "pdf_diversity_high": {
        "es": "Diversidad alta",
        "en": "High diversity",
        "fr": "Diversité élevée",
    },
    "pdf_evenness_low": {
        "es": "Equitatividad baja",
        "en": "Low evenness",
        "fr": "Équitabilité faible",
    },
    "pdf_evenness_moderate": {
        "es": "Equitatividad moderada",
        "en": "Moderate evenness",
        "fr": "Équitabilité modérée",
    },
    "pdf_evenness_high": {
        "es": "Equitatividad alta",
        "en": "High evenness",
        "fr": "Équitabilité élevée",
    },
    "pdf_specimen_detail": {
        "es": "Detalle por Espécimen",
        "en": "Specimen Detail",
        "fr": "Détail par Spécimen",
    },
    "pdf_probability": {
        "es": "Probabilidad",
        "en": "Probability",
        "fr": "Probabilité",
    },
    "pdf_classification": {
        "es": "Clasificación",
        "en": "Classification",
        "fr": "Classification",
    },

    # ── Error ──
    "model_error": {
        "es": "Error al cargar el modelo",
        "en": "Error loading model",
        "fr": "Erreur lors du chargement du modèle",
    },

    # ── UI adicional ──
    "stats_highest": {
        "es": "Mayor confianza",
        "en": "Highest confidence",
        "fr": "Confiance la plus élevée",
    },
    "stats_lowest": {
        "es": "Menor confianza",
        "en": "Lowest confidence",
        "fr": "Confiance la plus faible",
    },
    "pdf_download_subtitle": {
        "es": "Descarga un informe completo con los resultados de clasificación, estadísticos y gráficos.",
        "en": "Download a complete report with classification results, statistics and charts.",
        "fr": "Téléchargez un rapport complet avec les résultats de classification, statistiques et graphiques.",
    },
    "footer_text": {
        "es": "Clasificador de Foraminíferos Bentónicos · 4 géneros",
        "en": "Benthic Foraminifera Classifier · 4 genera",
        "fr": "Classificateur de Foraminifères Benthiques · 4 genres",
    },
}

# ── Información de géneros por idioma ──
GENUS_TRANSLATIONS = {
    "Ammonia": {
        "es": {
            "descripcion": "Género de foraminífero bentónico muy común en ambientes costeros y estuarinos. "
                           "Presenta una concha trocoespiral con cámaras infladas y suturas deprimidas.",
            "habitat": "Aguas someras costeras, estuarios, lagunas",
        },
        "en": {
            "descripcion": "Very common benthic foraminifera genus in coastal and estuarine environments. "
                           "It has a trochospiral shell with inflated chambers and depressed sutures.",
            "habitat": "Shallow coastal waters, estuaries, lagoons",
        },
        "fr": {
            "descripcion": "Genre de foraminifère benthique très commun dans les environnements côtiers et estuariens. "
                           "Il présente une coquille trochospiralée avec des chambres gonflées et des sutures déprimées.",
            "habitat": "Eaux côtières peu profondes, estuaires, lagunes",
        },
    },
    "Bolivina": {
        "es": {
            "descripcion": "Foraminífero bentónico con concha biserial comprimida lateralmente. "
                           "Común en ambientes marinos de plataforma continental.",
            "habitat": "Plataforma continental, ambientes de baja oxigenación",
        },
        "en": {
            "descripcion": "Benthic foraminifera with a laterally compressed biserial shell. "
                           "Common in continental shelf marine environments.",
            "habitat": "Continental shelf, low-oxygen environments",
        },
        "fr": {
            "descripcion": "Foraminifère benthique avec une coquille bisériale comprimée latéralement. "
                           "Commun dans les environnements marins du plateau continental.",
            "habitat": "Plateau continental, environnements à faible oxygénation",
        },
    },
    "Cibicides": {
        "es": {
            "descripcion": "Foraminífero bentónico con concha trocoespiral, generalmente planoconvexo, "
                           "que vive adherido a sustratos duros. Indicador paleoceanográfico importante.",
            "habitat": "Plataforma continental, adherido a sustratos",
        },
        "en": {
            "descripcion": "Benthic foraminifera with a trochospiral shell, generally planoconvex, "
                           "that lives attached to hard substrates. Important paleoceanographic indicator.",
            "habitat": "Continental shelf, attached to substrates",
        },
        "fr": {
            "descripcion": "Foraminifère benthique avec une coquille trochospiralée, généralement planoconvexe, "
                           "qui vit fixé aux substrats durs. Indicateur paléocéanographique important.",
            "habitat": "Plateau continental, fixé aux substrats",
        },
    },
    "Elphidium": {
        "es": {
            "descripcion": "Foraminífero bentónico con concha planispiral involuta, ornamentada con "
                           "puentes suturales (procesos retrales). Habita ambientes someros.",
            "habitat": "Aguas someras, plataforma interna",
        },
        "en": {
            "descripcion": "Benthic foraminifera with an involute planispiral shell, ornamented with "
                           "sutural bridges (retral processes). Inhabits shallow environments.",
            "habitat": "Shallow waters, inner shelf",
        },
        "fr": {
            "descripcion": "Foraminifère benthique avec une coquille planispiralée involute, ornée de "
                           "ponts suturaux (processus rétraux). Habite les environnements peu profonds.",
            "habitat": "Eaux peu profondes, plateau interne",
        },
    },
}

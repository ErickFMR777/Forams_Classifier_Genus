# 🔬 Forams Classifier – Genus

Aplicación web para la clasificación automatizada de **4 géneros de foraminíferos bentónicos** (*Ammonia*, *Bolivina*, *Cibicides*, *Elphidium*) mediante deep learning.

## Características

- **Clasificación por imagen** usando un modelo ResNet-18 fine-tuned (~11.2M parámetros)
- **Carga múltiple** de especímenes (JPG, PNG, BMP, TIFF, WebP)
- **Estadísticos** de confianza globales y por género
- **Índices de diversidad**: Shannon (H'), Simpson (1-D), Pielou (J)
- **Exportación a PDF** con tabla resumen, estadísticos y detalle por espécimen
- **Multiidioma**: Español, English, Français
- **Interfaz dark** profesional con Streamlit

## Estructura del proyecto

```
├── app.py                  # Aplicación Streamlit principal
├── translations.py         # Traducciones ES/EN/FR
├── forams_model.pth        # Modelo PyTorch (ResNet-18)
├── requirements.txt        # Dependencias Python
├── .streamlit/
│   └── config.toml         # Configuración del tema
├── .gitignore
└── README.md
```

## Instalación y ejecución

```bash
# Clonar el repositorio
git clone https://github.com/ErickFMR777/Forams_Classifier_Genus.git
cd Forams_Classifier_Genus

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar
streamlit run app.py
```

La aplicación se abrirá en `http://localhost:8501`.

## Requisitos

- Python ≥ 3.10
- PyTorch ≥ 2.0
- Streamlit ≥ 1.30

## Modelo

ResNet-18 pre-entrenado en ImageNet y fine-tuned para clasificar imágenes de foraminíferos bentónicos obtenidas por microscopía óptica y electrónica de barrido (SEM). Las imágenes se redimensionan a 224×224 px y se normalizan con los parámetros estándar de ImageNet.

## Licencia

MIT

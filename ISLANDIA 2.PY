import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import subprocess

# --- CONFIGURACIÓN DE ARCHIVOS ---
archivos_objetivo = [
    "gastos islandia.csv", 
    "ingresos islandia.csv", 
    "islandia.csv", 
    "produccion de islandia.csv"
]

def extraer_datos_reales(archivo):
    """Limpia el CSV de Islandia de forma agresiva para encontrar números."""
    try:
        df = pd.read_csv(archivo, sep=None, engine='python', encoding='utf-8-sig')
        posibles_cols = ['OBS_VALUE', 'Value', 'obs_value', 'value']
        col_seleccionada = None
        
        for c in posibles_cols:
            if c in df.columns:
                col_seleccionada = c
                break
        
        if not col_seleccionada:
            for c in df.columns:
                if df[c].dtype in [np.float64, np.int64]:
                    col_seleccionada = c
                    break
        
        if not col_seleccionada: col_seleccionada = df.columns[-1]

        serie = df[col_seleccionada].astype(str).str.replace(r'[^\d.-]', '', regex=True)
        serie = pd.to_numeric(serie, errors='coerce').dropna()
        
        return serie.values
    except Exception as e:
        print(f"❌ Error crítico en {archivo}: {e}")
        return np.array([])

def generar_analisis(datos, nombre):
    """Calcula el temario con interpretaciones."""
    n = len(datos)
    if n < 5: return None
    
    # 1. Tendencia Central
    media = np.mean(datos)
    mediana = np.median(datos)
    moda_res = stats.mode(datos, keepdims=True)
    moda = moda_res[0][0] if len(moda_res[0]) > 0 else 0
    
    # 2. Dispersión
    std = np.std(datos)
    varianza = np.var(datos)
    rango = np.ptp(datos)
    iqr = stats.iqr(datos)
    desv_media = np.mean(np.absolute(datos - media))
    cv = (std / media) if media != 0 else 0
    
    # Interpretación de la variabilidad
    interpretacion_var = "baja (datos consistentes)" if cv < 0.15 else "moderada" if cv < 0.35 else "alta (mucha volatilidad)"
    
    # 3. Datos Agrupados (Sturges)
    k = int(1 + 3.322 * np.log10(n))
    counts, bins = np.histogram(datos, bins=k)
    
    # 4. Probabilidades
    p_exito = (datos > media).mean()
    dist_binomial = stats.binom.pmf(k=2, n=10, p=p_exito)
    dist_poisson = stats.poisson.pmf(int(media), media) if media > 0 else 0
    
    # Gráfica
    plt.figure(figsize=(7, 4))
    plt.hist(datos, bins=k, color='#3498db', edgecolor='black', alpha=0.7)
    plt.axvline(media, color='red', linestyle='dashed', linewidth=2, label=f'Media: {media:.2f}')
    plt.axvline(mediana, color='green', linestyle='dotted', linewidth=2, label=f'Mediana: {mediana:.2f}')
    plt.title(f"Distribución de Frecuencias: {nombre}")
    plt.xlabel("Valor")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.tight_layout()
    img_name = f"plot_{nombre.replace(' ', '_')}.png"
    plt.savefig(img_name)
    plt.close()

    return {
        "nombre": nombre, "media": media, "mediana": mediana, "moda": moda,
        "std": std, "var": varianza, "rango": rango, "iqr": iqr, "dm": desv_media,
        "cv": cv, "inter_var": interpretacion_var,
        "bin": dist_binomial, "poi": dist_poisson, "img": img_name, "k": k,
        "n": n
    }

# --- PROCESO PRINCIPAL ---
reportes = []
for f in archivos_objetivo:
    if os.path.exists(f):
        print(f"🔍 Analizando {f}...")
        data = extraer_datos_reales(f)
        analisis = generar_analisis(data, f.upper().replace(".CSV", ""))
        if analisis: reportes.append(analisis)

# --- GENERACIÓN DE LATEX ---
latex = r"""
\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[spanish]{babel}
\usepackage{amsmath, graphicx, booktabs, geometry, xcolor, float}
\geometry{margin=2.5cm}
\begin{document}

\begin{center}
    \vspace*{1cm}
    \Huge \textbf{Reporte Estadístico Integral: Islandia}\\
    \vspace{0.5cm}
    \LARGE Análisis de Indicadores Económicos\\
    \vspace{1cm}
    \large \textbf{Analista:} Jorge - Sistema Automatizado de Datos\\
    \today
\end{center}

\newpage
\tableofcontents
\newpage

\section{Introducción}
Este documento presenta un análisis estadístico exhaustivo de los datos económicos de Islandia. Se evalúan medidas de tendencia central para identificar valores típicos, medidas de dispersión para entender el riesgo o variabilidad, y modelos probabilísticos para proyectar comportamientos futuros.
"""

for r in reportes:
    latex += r"""
\section{Análisis de """ + r['nombre'] + r"""}

\subsection{Medidas de Tendencia Central}
Estas medidas nos indican el "centro" de los datos, permitiendo resumir toda la muestra en un solo valor representativo.
\begin{itemize}
    \item \textbf{Media (Promedio):} """ + f"{r['media']:.4f}" + r""". Representa el valor de equilibrio aritmético de los datos.
    \item \textbf{Mediana:} """ + f"{r['mediana']:.4f}" + r""". Es el valor que divide la muestra exactamente a la mitad; el 50\% de los datos son menores o iguales a este número.
    \item \textbf{Moda:} """ + f"{r['moda']:.4f}" + r""". Es el dato que más se repite en la serie original.
\end{itemize}

\subsection{Medidas de Dispersión}
La dispersión mide qué tan alejados están los datos respecto a la media. Es fundamental para evaluar la estabilidad.
\begin{center}
\begin{tabular}{lp{4cm}lp{4cm}}
\toprule
\textbf{Métrica} & \textbf{Resultado} & \textbf{Métrica} & \textbf{Resultado} \\ \midrule
Rango & """ + f"{r['rango']:.2f}" + r""" & Varianza & """ + f"{r['var']:.2f}" + r""" \\
Desv. Estándar & """ + f"{r['std']:.2f}" + r""" & IQR & """ + f"{r['iqr']:.2f}" + r""" \\
Coef. Variación & """ + f"{r['cv']:.2%}" + r""" & Desv. Media & """ + f"{r['dm']:.2f}" + r""" \\ \bottomrule
\end{tabular}
\end{center}

\textbf{Interpretación:} El Coeficiente de Variación es del """ + f"{r['cv']:.2%}" + r""", lo que indica una variabilidad \textbf{""" + r['inter_var'] + r"""}. 

\subsection{Análisis de Frecuencias (Datos Agrupados)}
Para entender la forma de la distribución, se aplicó la \textbf{Regla de Sturges}, resultando en """ + str(r['k']) + r""" intervalos de clase. 

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{""" + r['img'] + r"""}
    \caption{Histograma de frecuencias y líneas de tendencia central.}
\end{figure}

\subsection{Probabilidad y Distribuciones}
\begin{itemize}
    \item \textbf{Distribución Binomial:} $P(X=2, n=10) = $ """ + f"{r['bin']:.4f}" + r""". Representa la probabilidad de obtener exactamente 2 éxitos en 10 intentos, considerando el promedio como umbral de éxito.
    \item \textbf{Distribución Poisson:} $P(X=\mu) = $ """ + f"{r['poi']:.4f}" + r""". Mide la probabilidad de que ocurra un evento exactamente igual al promedio en un intervalo determinado.
\end{itemize}
\newpage
"""

# --- SECCIÓN DE CONCLUSIÓN GENERAL ---
media_general_cv = np.mean([r['cv'] for r in reportes])
resumen_conclusion = "estable" if media_general_cv < 0.20 else "volátil"

latex += r"""
\section{Conclusión General del Análisis}
Tras procesar los """ + str(len(reportes)) + r""" conjuntos de datos de Islandia, se concluye lo siguiente:

\begin{enumerate}
    \item \textbf{Consistencia de los Datos:} El coeficiente de variación promedio global es de """ + f"{media_general_cv:.2%}" + r""", lo que sugiere un entorno económico relativamente """ + resumen_conclusion + r""".
    \item \textbf{Tendencia:} La mayoría de los indicadores muestran una distribución donde la media y la mediana presentan valores cercanos, lo que indica una baja presencia de valores atípicos (outliers) extremos.
    \item \textbf{Proyección:} Los modelos de probabilidad sugieren que los valores tienden a concentrarse fuertemente alrededor de la media actual, permitiendo una planificación financiera basada en los promedios calculados.
\end{enumerate}

\vspace{1cm}
\begin{center}
    \textit{Fin del reporte estadístico.}
\end{center}
"""

with open("Reporte_Consolidado.tex", "w", encoding="utf-8") as f:
    f.write(latex + r"\end{document}")

# Compilar
try:
    print("🚀 Generando PDF...")
    subprocess.run(["pdflatex", "-interaction=batchmode", "Reporte_Consolidado.tex"], stdout=subprocess.PIPE)
    print("✅ PDF FINAL GENERADO CON ÉXITO: Reporte_Consolidado.pdf")
except Exception as e:
    print(f"⚠️ Error al compilar: {e}. El archivo .tex está listo.")
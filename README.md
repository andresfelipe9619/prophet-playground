# Baloto Analytics

Herramientas de análisis y forecasting sobre resultados históricos del Baloto (Colombia): 5 balotas principales (1-43) + superbalota (1-16), sorteos cada miércoles y sábado.

## Antes de usar esto: un descargo honesto

Los sorteos de Baloto son, por diseño, **independientes y uniformes**: no hay tendencia, estacionalidad ni "memoria" real entre sorteos que un modelo pueda aprender. Eso significa:

- Ningún modelo de este repo (Prophet, AutoARIMA/AutoETS/AutoTheta, XGBoost) puede superar de forma sostenida la probabilidad teórica. Si en un backtest puntual un modelo "gana", lo más probable es que sea varianza, no señal — por eso `backtest.py` reporta siempre el resultado junto a lo que el azar puro esperaría, con su p-valor.
- El heurístico de "número atrasado" (overdue) es la falacia del jugador: el tiempo desde la última vez que salió un número no cambia la probabilidad de que salga en el próximo sorteo. Se incluye en el dashboard porque es una vista popular, no porque funcione.
- Antes de mirar cualquier predicción, revisa la pestaña **Aleatoriedad** del dashboard: si tus datos pasan las pruebas de aleatoriedad (que es lo esperable), cualquier "patrón" que un modelo muestre es ruido sobreajustado.

Dicho esto, el proyecto sigue siendo útil como: (1) ejercicio serio de forecasting/backtesting con librerías modernas, y (2) panel de estadística descriptiva (frecuencias, gaps, hot/cold) para explorar tus propios datos.

## Qué cambió en esta actualización

- **Prophet** actualizado a 1.4.x, sin las estacionalidades semanales/quincenales/anuales inventadas de la versión anterior (no tenían sentido sobre datos que solo existen los miércoles y sábados).
- **StatsForecast.py** (Nixtla `statsforecast`) reemplaza al viejo `ARIMA.py`: en vez de un SARIMAX con orden fijo a mano, ajusta AutoARIMA + AutoETS + AutoTheta con búsqueda automática de orden, para las 6 posiciones a la vez.
- **XGBoost.py** corregido: la versión anterior usaba `train_test_split` con `shuffle=True` sobre una serie de tiempo, lo cual mezclaba sorteos futuros en el entrenamiento (data leakage) e inflaba la precisión reportada. Ahora el split es cronológico y se agregan features de lags/frecuencia móvil.
- **`analysis/randomness.py`**: frecuencias, gaps, hot/cold, y pruebas estadísticas reales (chi-cuadrado de uniformidad — por posición y agrupada, runs test, ACF/Ljung-Box) para saber si hay algo que modelar antes de modelarlo.
- **`backtest.py`**: backtest walk-forward que compara cada modelo contra la expectativa exacta de aciertos por azar (distribución hipergeométrica), con test de significancia.
- **`dashboard/app.py`**: dashboard interactivo en Streamlit con las piezas anteriores, pensado para decisión informada, no para "el número ganador".

## Estructura

```
Prophet.py            # forecasting por posición con Prophet
StatsForecast.py       # AutoARIMA / AutoETS / AutoTheta (Nixtla statsforecast)
XGBoost.py             # XGBoost con split cronológico
backtest.py            # backtest walk-forward vs. azar (CLI)
summary_charts.py      # gráficas estáticas originales (matplotlib/seaborn)
contants.py             # festivos de Colombia (para Prophet, opcional) y utilidades de fecha
analysis/
  randomness.py        # frecuencias, gaps, hot/cold, pruebas de aleatoriedad
models/
  common.py             # rangos de balotas, helpers compartidos
  baseline.py            # baseline de frecuencia + expectativa por azar (hipergeométrica)
  statsforecast_model.py # wrapper de Nixtla statsforecast
  xgboost_model.py        # features + entrenamiento XGBoost
dashboard/
  app.py                # dashboard Streamlit
utils/
  processor.py           # carga/preprocesamiento de CSVs y comparación de predicciones
  sample_data.py          # generador de datos sintéticos de demo
  scraper.py               # scraper de resultados (loterias.com)
  csv_merger.py            # combina CSVs anuales exportados
```

## Datos

Los scripts esperan `exported_data/final-final.csv` con columnas `Date` (dd/mm/yyyy) y `Ball` (6 números separados por guion: 5 principales + superbalota, ej. `3-12-19-27-41-8`). Esa carpeta está en `.gitignore` — no viene en el repo.

Si no tienes ese archivo, `utils/sample_data.py` genera datos sintéticos (sorteos uniformes independientes reales, no una simulación de "patrón") para que puedas explorar el dashboard y los scripts sin tus datos privados. El dashboard cae a este modo demo automáticamente y lo avisa en pantalla.

## Instalación

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Uso

**Dashboard (recomendado):**
```bash
streamlit run dashboard/app.py
```
Sube tu CSV en la barra lateral, o déjalo vacío para explorar con datos demo.

**Scripts individuales:**
```bash
python Prophet.py
python StatsForecast.py AutoARIMA   # o AutoETS / AutoTheta
python XGBoost.py
```

**Backtest vs. azar (línea de comandos):**
```bash
python backtest.py --n-windows 20 --min-train 100          # sin Prophet (más rápido)
python backtest.py --n-windows 20 --include-prophet          # con Prophet (más lento)
```
Guarda un resumen en `backtest_summary.csv` con, por modelo: aciertos promedio, aciertos esperados por azar, y el p-valor de si la diferencia es real.

## Métricas

Ver `README_METRICS.md` para el detalle de las métricas de cross-validation de Prophet (MAE, RMSE, MAPE, coverage, etc.). La pregunta que de verdad importa para decidir si vale la pena confiar en un modelo — ¿le gana al azar? — la responde `backtest.py`, no esas métricas por sí solas.

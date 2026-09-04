# Sobre "mejorar la precisión" de estos modelos

Este documento antes listaba tácticas genéricas de tuning de Prophet (ajustar `changepoint_prior_scale`, agregar estacionalidades, transformar la variable, etc.). Se quitaron porque, aplicadas a sorteos de lotería (independientes y uniformes por diseño), lo único que logran es sobreajustar ruido histórico — cualquier mejora en las métricas de cross-validation sobre datos pasados no se traduce en mejor predicción futura, porque no hay señal real que capturar.

Si quieres saber si un modelo realmente aporta algo, la pregunta correcta no es "¿mejoré el MAPE?" sino "¿le gana al azar de forma estadísticamente significativa?". Eso es exactamente lo que responde `backtest.py` (ver README.md), comparando los aciertos del modelo contra la expectativa exacta por azar (distribución hipergeométrica) con un p-valor.

Para diagnosticar si hay *algo* que modelar antes de intentar afinar cualquier modelo, usa `analysis/randomness.py` (o la pestaña "Aleatoriedad" del dashboard): chi-cuadrado de uniformidad, runs test y Ljung-Box. Si esas pruebas no muestran evidencia de no-aleatoriedad — que es lo esperable — ajustar hiperparámetros de Prophet no va a producir un modelo mejor, solo uno más confiado en su propio ruido.

"""Shared UI vocabulary for every dashboard page.

The dashboard spans three domains now, and this module is what keeps them
looking and reading like one product: the explanatory copy, the two helpers that
render a section and a chart, and the verdict badge.

**All explanatory copy lives in the dicts below**, and nowhere else. That was
already this project's rule when the dashboard was Baloto-only, and splitting
the pages into modules is exactly when it would have been easiest to lose: three
files with their own inline strings cannot be reviewed as a whole, and the rule
this project actually cares about — no chart or table appears without saying what
it does *not* mean — is only checkable when the texts sit together.

There are four such dicts, all keyed the same way so one key carries every layer
of explanation a surface needs:

- `HELP` — the hover ⓘ. There when you want it, out of the way when you don't.
- `PLAIN` — the always-visible guided box: what you are looking at, what you may
  conclude, and what it does **not** mean. Written for a reader who is competent
  but has never met a p-value. This is the layer that stops the dashboard being
  readable only by someone who already knows the answer.
- `READ` — a one-line "how to read it" printed under a chart's title.
- `GLOSSARY` — the jargon, defined once, with a concrete example each.

`section()` and `chart()` pull `PLAIN` and `READ` in automatically off the key
they already take, so a page gets the guided layer without a single new argument
at the call site, and copy can never drift to a page module.
"""

import streamlit as st

from dashboard import mobile

# Every explanatory tooltip in the UI, in one place.
#
# Streamlit renders `help=` as a small ⓘ next to the element and shows the text
# on hover, which is the right home for "what am I looking at?" — it is there
# when you want it and out of the way when you don't. Keeping the copy in one
# dict rather than inline at each call site is what makes it reviewable as a
# whole: this project's rule is that no chart or table appears without saying
# what it does *not* mean, and that is only checkable if the texts sit together.
#
# Spanish, like the rest of the UI. Docstrings and comments stay English.
HELP = {
    # -- Resumen
    "tab_resumen": "Estado general de los datos cargados y el veredicto sobre si tus sorteos se comportan "
                   "como un sorteo justo. Empieza por aquí: si los datos vienen mal, todo lo demás sobra.",
    "n_draws": "Cuántos sorteos hay cargados después de filtrar. Más sorteos = pruebas estadísticas más "
               "sensibles. Por debajo de ~200 casi nada es concluyente.",
    "date_from": "Fecha del sorteo más antiguo del archivo cargado.",
    "date_to": "Fecha del sorteo más reciente. Si está muy atrás, actualiza con "
               "`python -m lottery.utils.scraper --years <año>`.",
    "sorted_flag": "Muchas fuentes publican las 5 balotas ordenadas de menor a mayor. Si es así, cada columna "
                   "deja de ser una balota al azar y pasa a ser un estadístico de orden (el mínimo, el 2do "
                   "menor...), lo que hace que las pruebas por posición marquen patrones falsos. La prueba "
                   "agrupada de abajo es inmune a esto.",
    "pooled_test": "Prueba chi-cuadrado agrupada: junta todas las posiciones y solo pregunta si cada número "
                   "sale aproximadamente la misma cantidad de veces. Al no mirar en qué columna cayó cada "
                   "balota, no la engaña que los datos vengan ordenados.",
    "pooled_main_p": "p-valor de la prueba agrupada sobre las 5 balotas principales. Alto (>0.05) = sin "
                     "evidencia contra la uniformidad, que es el resultado sano. Bajo sería raro de verdad, "
                     "y más probablemente indica un problema de datos que una lotería vencible.",
    "pooled_super_p": "Lo mismo para la superbalota, evaluada aparte porque su rango es 1-16 y no 1-43.",

    # -- Probabilidades
    "tab_probabilidades": "La única pestaña con respuestas exactas y sin modelos: son combinatoria pura, no "
                          "dependen de tu histórico ni de ninguna predicción.",
    "jackpot_odds": "Probabilidad de acertar las 5 principales más la superbalota, calculada exactamente. "
                    "No cambia con la estrategia, la suerte ni la fecha.",
    "total_combos": "Todas las jugadas posibles: C(43,5) × 16. Todas son igual de probables.",
    "prize_table": "Los montos son de ejemplo y debes reemplazarlos por la tabla oficial vigente: varias "
                   "categorías son variables y el premio mayor se acumula. Las probabilidades de la izquierda "
                   "son exactas y no cambian con lo que escribas.",
    "ticket_price": "Precio de un tiquete. Se usa para calcular el valor esperado y el RTP de abajo.",
    "ev_section": "Cuánto vale jugar, dados los premios que escribiste arriba. Es aritmética exacta sobre las "
                  "probabilidades reales, no una simulación ni una estimación.",
    "expected_return": "Promedio que devuelve un tiquete a largo plazo, antes de restar lo que costó.",
    "expected_value": "Retorno esperado menos el precio del tiquete. Negativo significa que cada jugada pierde "
                      "esa cantidad en promedio. Ninguna forma de elegir números lo cambia.",
    "rtp": "Qué porcentaje de lo apostado devuelve el juego a largo plazo. Una máquina tragamonedas ronda el "
           "90%; una lotería suele estar muy por debajo.",
    "any_prize": "Probabilidad de llevarte *algo*, aunque sea la categoría más baja que pague.",
    "breakeven": "Cuánto tendría que acumularse el premio mayor para que el valor esperado llegue a cero. Ojo: "
                 "aun superándolo, el acumulado se reparte entre todos los ganadores y hay retención.",
    "category_chart": "Probabilidad exacta de cada categoría, en escala logarítmica porque abarcan varios "
                      "órdenes de magnitud. Las barras más bajas son las que pagan más.",

    # -- Frecuencia y gaps
    "tab_frecuencia": "Cuántas veces ha salido cada número y cuánto lleva sin salir. Vista descriptiva: "
                      "describe el pasado, no anticipa el futuro.",
    "freq_chart": "Veces que salió cada número en esta posición, contra la línea punteada de lo que se "
                  "esperaría si todo fuera uniforme. Ojo con el caso ordenado: si el Resumen dice que tus "
                  "balotas vienen guardadas de menor a mayor, esta columna no es una balota al azar sino un "
                  "estadístico de orden, y verás una escalera marcada (la 'Balota 1' es siempre el mínimo, "
                  "así que los números bajos dominan). Eso es el orden, no un patrón del sorteo. Sin ese "
                  "efecto, las diferencias que quedan son ruido de muestreo normal: con unos cientos de "
                  "sorteos y 43 números ninguna barra cae exactamente en la línea. El veredicto real está "
                  "en la prueba agrupada del Resumen.",
    "gaps_section": "Cada cuánto suele aparecer un número y cuánto lleva ausente. El *overdue score* es el "
                    "heurístico de 'ya se demoró, le toca'. Para sorteos independientes no tiene poder "
                    "predictivo: es la falacia del jugador. Está aquí porque mucha gente lo busca, no porque "
                    "sirva.",

    # -- Hot / cold
    "tab_hotcold": "Qué números vienen saliendo más (o menos) que su promedio histórico en la ventana "
                   "reciente que elijas.",
    "hotcold_chart": "Diferencia entre el % de apariciones en los últimos sorteos y el % de todo el "
                     "histórico. Rojo = por encima, azul = por debajo. Con ventanas cortas hay muy pocas "
                     "observaciones por número, así que estas barras se mueven mucho aunque el sorteo no "
                     "haya cambiado en nada.",
    "hotcold_window": "Cuántos sorteos recientes componen la ventana 'caliente'. Más corta = más ruido.",

    # -- Aleatoriedad
    "tab_aleatoriedad": "Las pruebas formales: ¿hay alguna estructura explotable en estos sorteos? Lo "
                        "esperable, y lo sano, es que la respuesta sea no.",
    "verdict_table": "Tres pruebas por posición. chi-cuadrado: ¿salen todos los números con la misma "
                     "frecuencia? runs test: ¿hay rachas por encima o por debajo de la mediana? Ljung-Box: "
                     "¿un sorteo dice algo del siguiente? Recuerda que al correr 6 posiciones a la vez, ~1 de "
                     "cada 20 pruebas marca 'No' por puro azar.",
    "acf_section": "Autocorrelación: cuánto se parece la serie a sí misma desplazada N sorteos. Es la prueba "
                   "directa de si un modelo de series de tiempo (ARIMA, Prophet) tiene algo que aprender aquí.",
    "acf_chart": "Cada barra es la correlación con el sorteo N posiciones atrás. Las líneas punteadas son la "
                 "banda de significancia: barras dentro de la banda son indistinguibles de cero. Un ACF "
                 "totalmente dentro de la banda es exactamente lo que produce un proceso sin memoria.",
    "ljung_box": "Prueba conjunta sobre todos los lags a la vez. p alto = no hay autocorrelación detectable, "
                 "es decir, no hay 'memoria' que un modelo pueda explotar.",

    # -- Estructura de la combinación
    "structure_section": "Resume cada sorteo en un solo número que no depende del orden en que estén "
                         "guardadas las balotas — su suma, cuántas son impares, cuántas caben en una "
                         "fecha — y lo compara contra la distribución exacta. La prueba agrupada mira "
                         "cada balota por separado; ésta mira las cinco juntas, así que ve cosas que "
                         "aquella no puede ver.",
    "sum_chart": "Distribución de la suma de las 5 balotas. La curva es la forma exacta que produce la "
                 "combinatoria, no un ajuste a tus datos. Las sumas del centro son más frecuentes "
                 "porque hay muchísimas más combinaciones que suman 110 que combinaciones que suman "
                 "15 — no porque una combinación concreta sea más probable que otra.",
    "parity_chart": "Cuántas de las 5 balotas son impares, contra lo exacto. El pool 1-43 tiene 22 "
                    "impares y 21 pares, así que 2 o 3 impares es lo normal y 0 o 5 es raro.",
    "calendar_chart": "Cuántas de las 5 balotas son 31 o menos, contra lo exacto. Los números 1-31 "
                      "caben en una fecha y se juegan muchísimo más, aunque no sean más probables. "
                      "Este reparto es lo que decide entre cuánta gente repartirías el premio.",
    "structure_verdict": "Las tres pruebas juntas. Son inmunes al problema de los datos ordenados, y por "
                         "una razón más fuerte que la prueba agrupada: ordenar un sorteo no cambia su "
                         "suma ni cuántas impares tiene, porque son propiedades del conjunto.",
    "sum_percentile": "Qué porcentaje de todas las combinaciones posibles suma igual o menos que ésta. "
                      "Cerca de 50% es una suma del montón, que es donde juega casi todo el mundo. No "
                      "cambia tu probabilidad de ganar; cambia con cuánta gente repartes.",

    # -- Forecast
    "tab_forecast": "Corre un modelo sobre tu histórico y pídele una sugerencia para el próximo sorteo. Es un "
                    "ejercicio de forecasting, no una predicción confiable.",
    "model_choice": "FrequencyBaseline juega el número más frecuente de cada posición (la referencia a "
                    "vencer). AutoARIMA/AutoETS/AutoTheta son modelos clásicos de series de tiempo. Prophet "
                    "descompone tendencia y estacionalidad. XGBoost usa lags y medias móviles. Ninguno "
                    "supera al azar en el backtest — para eso está esa pestaña.",

    # -- Jugadas
    "tab_jugadas": "Genera jugadas, verifícalas contra sorteos reales y mide si tu forma de elegirlas le gana "
                   "al azar. Generar números es válido; lo que ninguna estrategia logra es hacer una jugada "
                   "más probable que otra.",
    "n_tickets": "Cuántas jugadas generar de una vez.",
    "distinct_numbers": "Cuántos números distintos cubre el conjunto de jugadas entre todas.",
    "pool_coverage": "Qué porcentaje de los 43 números toca tu portafolio. Cubrir más reparte los resultados "
                     "sobre el conjunto; no mejora ninguna jugada individual.",
    "portfolio_odds": "Probabilidad del premio mayor con todas estas jugadas juntas. Comprar N jugadas divide "
                      "la probabilidad entre N — es aritmética, y cuesta N veces más.",
    "check_input": "Escribe una jugada real (la tuya, por ejemplo) y mira cómo le habría ido en cada sorteo "
                   "de tu histórico.",
    "history_chart": "En cuántos sorteos de tu histórico esa jugada habría caído en cada categoría. Compáralo "
                     "con las probabilidades exactas de la pestaña Probabilidades: cualquier otra jugada da "
                     "una distribución estadísticamente equivalente.",
    "experiment_intro": "El experimento honesto: para cada sorteo se generan jugadas usando solo los sorteos "
                        "anteriores, y se comparan los aciertos contra la expectativa exacta del azar.",
    "draws_back": "Cuántos sorteos históricos evaluar. Más sorteos = más poder estadístico.",
    "per_draw": "Cuántas jugadas generar para cada sorteo evaluado.",
    "strategy_chart": "Aciertos promedio por jugada de cada estrategia, contra lo que da el azar puro "
                      "(0.58 de 5). Barras casi iguales es el resultado esperado y correcto.",
    "strategy_table": "La columna de veredicto usa umbral corregido por comparaciones múltiples (Bonferroni): "
                      "al probar varias estrategias a la vez, alguna parece ganadora por azar mucho más "
                      "seguido de lo que sugiere un 0.05 suelto.",
    "stability_section": "Una sola corrida es un sorteo de un proceso ruidoso: con α = 0.05, una estrategia "
                         "sin ninguna ventaja parece ganadora ~1 de cada 20 veces. Esto repite el "
                         "experimento con varias semillas y cuenta cuántas veces marcó ganador.",
    "stability_table": "`random` no puede tener ventaja: su tasa de marcado es tu piso de falsos positivos "
                       "medido. Una estrategia que no marque claramente más seguido que ella no ha "
                       "demostrado nada.",

    # -- Backtest
    "tab_backtest": "El veredicto del proyecto: cada modelo se entrena solo con datos anteriores al sorteo "
                    "que intenta predecir, y se compara contra lo que realmente salió. Lo que importa no es "
                    "cuántos aciertos, sino cuántos más que el azar puro.",
    "experiment_choice": "Últimos N sorteos da un promedio sobre la cola del histórico. Corte por fecha "
                         "responde la pregunta concreta: entreno con todo hasta julio, ¿qué habría predicho "
                         "para agosto y septiembre, que ya sabemos cómo salieron?",
    "n_windows": "Cuántos sorteos recientes dejar fuera del entrenamiento y evaluar. Con menos de ~30, una "
                 "sola corrida es una anécdota.",
    "min_train": "Cuántos sorteos como mínimo debe tener el modelo para entrenar antes de la primera ventana.",
    # Measured, not assumed: Prophet refits per position per window, which sounds
    # expensive and is not — 20 windows over 1035 draws cost 44.9s without it and
    # 49.8s with. It now ships on, because a backtest missing a model compares
    # five things while the correction beside it says six.
    "include_prophet": "Prophet reajusta un modelo por posición y por ventana. Suena caro y no lo es: "
                       "medido sobre 20 ventanas y 1035 sorteos, añade alrededor del 11% al tiempo "
                       "total. Viene encendido; apágalo solo si tienes prisa.",
    # Shown in place of `include_prophet` when the package is missing. It should
    # not be: prophet is in requirements.txt and every model the selector offers
    # is meant to be here. This is the safety net for an incomplete install, and
    # it says so — an option greyed out without a reason reads as a broken app,
    # which is exactly what it would be.
    "prophet_missing": "Prophet no está instalado en esta copia, y debería estarlo: está en las "
                       "dependencias del proyecto. Los demás modelos funcionan. Reinstala con "
                       "`pip install -r requirements.txt`.",
    "cutoff_date": "El modelo se entrena con todos los sorteos hasta esta fecha (inclusive) y predice los "
                   "posteriores, que ya sabemos cómo salieron.",
    "holdout_mode": "Reentrenar en cada sorteo es lo que harías jugando de verdad: antes de cada sorteo "
                    "reajustas el modelo con todo lo conocido hasta ahí. Entrenar una vez es la prueba "
                    "literal 'ajusto en julio y proyecto agosto y septiembre a ciegas' — más dura, porque el "
                    "modelo no ve nada nuevo.",
    "holdout_detail": "Sorteo por sorteo: los números que salieron de verdad y cuántos acertó cada modelo. "
                      "Una fila con 3 aciertos no es señal: acertar 3 o más de 5 en 43 pasa cerca del 1% de "
                      "las veces por azar, así que con varios modelos y una docena de sorteos es esperable "
                      "ver alguna. Lo que decide es el promedio.",
    "holdout_chart": "Aciertos de cada modelo en cada sorteo del periodo de prueba. La línea punteada es lo "
                     "que da el azar (0.58 de 5). Los picos por encima y por debajo son varianza normal.",
    "summary_chart": "Aciertos promedio de cada modelo contra el promedio del azar puro. Si las barras se ven "
                     "casi iguales, ese es el resultado esperado y correcto para una lotería justa.",
    "effect_ci": "La ventaja observada con su intervalo de confianza del 95%. El p-valor dice si "
                 "descartas el azar; el intervalo dice **con cuánta precisión** mediste. Un intervalo "
                 "que cruza el cero y es ancho no significa 'no hay ventaja': significa que esta "
                 "corrida no tuvo resolución para saberlo. Con pocas ventanas el intervalo es enorme.",
    "summary_table": "El p-valor es de una cola: mide si el modelo es *mejor* que el azar, no solo distinto. "
                     "Lee la columna corregida: al probar varios modelos contra los mismos sorteos, alguno "
                     "pasa el 5% por suerte mucho más seguido de lo que ese 5% sugiere. La corrección de "
                     "Bonferroni baja el umbral a 0.05 dividido entre el número de modelos.",

    # -- Potencia y sensibilidad
    "tab_power": "Las dos preguntas que van antes de cualquier resultado: ¿qué tan grande tendría que "
                 "ser una ventaja para que estos datos la vieran, y estas pruebas son capaces de ver "
                 "una ventaja real cuando existe?",
    "mde_section": "El efecto mínimo detectable (MDE): la ventaja más pequeña que este número de "
                   "sorteos podría distinguir del azar de forma confiable. Todo lo que esté por debajo "
                   "es invisible para tu backtest — no porque no exista, sino porque no hay datos "
                   "suficientes. Es lo que le pone resolución a un 'no encontré nada'.",
    "mde_draws": "Cuántos sorteos tendría la evaluación. Súbelo para ver cuánto mejora la resolución: "
                 "el MDE cae con la raíz de N, así que cuadruplicar los datos solo lo reduce a la mitad.",
    "mde_metric": "La ventaja más pequeña detectable, como porcentaje sobre la media del azar. Si tu "
                  "backtest no encontró nada, lo que demostraste es 'no hay ventaja mayor que esto'.",
    "mde_target": "El promedio de aciertos que un modelo tendría que alcanzar para que la prueba lo "
                  "marcara, contra la media del azar de 0.58.",
    "power_chart": "Probabilidad de detectar una ventaja según su tamaño, con la cantidad de sorteos "
                   "elegida. La línea punteada es el 80%, el umbral convencional. A la izquierda de "
                   "donde la curva la cruza, tu prueba es prácticamente ciega.",
    "required_table": "Cuánta historia haría falta para detectar cada tamaño de ventaja. La columna de "
                      "años es la que importa: varias filas superan la edad del juego, que es la "
                      "respuesta honesta a por qué nadie ha demostrado nunca que un sistema de lotería "
                      "funcione.",
    "super_mde": "Lo mismo para la superbalota, que es un ensayo de Bernoulli (1 de 16) y no "
                 "hipergeométrico, así que su varianza y su resolución son distintas.",
    "sensitivity_section": "El espejo del panel de arriba. Ya sabes que estas pruebas dicen "
                           "'aleatorio' sobre datos aleatorios; esto verifica que digan 'no aleatorio' "
                           "sobre datos con un patrón plantado a propósito. Sin esta comprobación, un "
                           "resultado nulo también podría significar que las pruebas están ciegas.",
    "sensitivity_strengths": "Fuerza del sesgo inyectado: 3 números reciben peso (1 + fuerza) frente a "
                             "1 de los demás. Fuerza 0 reproduce sorteos uniformes y es el control — "
                             "ahí las tres tasas deben quedar cerca de α.",
    "sensitivity_seeds": "Cada semilla regenera los datos y los tiquetes, así que son experimentos "
                         "independientes, no re-tiradas del mismo dataset.",
    "sensitivity_table": "Empieza por las filas de fuerza 0. Si algún detector dispara mucho más "
                         "seguido que α ahí, está roto y el resto de la tabla no significa nada. "
                         "`random` debe quedarse en el piso incluso con sesgo fuerte: un tiquete "
                         "uniforme tiene los mismos aciertos esperados sin importar cómo estén "
                         "pesadas las balotas, así que es el control correcto. Solo una estrategia "
                         "que *aprenda* cuáles números salen más puede convertir el sesgo en aciertos.",
    # -- Reparto de premios
    "tab_split": "La única palanca real de este juego. Elegir números impopulares **no** te hace "
                 "ganar más seguido — eso es imposible. Cambia la otra mitad del valor esperado: "
                 "cuánta gente comparte contigo si ganas.",
    "popularity_score": "Qué tan comúnmente se juega esa combinación, de 0 (muy rara) a 1 (muy "
                        "jugada). Es **ordinal**: 0.6 no significa 60% de nada, solo que se juega "
                        "más que una de 0.4. Sirve para comparar combinaciones entre sí.",
    "tickets_sold": "Tiquetes vendidos para ese sorteo. No hay valor por defecto que sea "
                    "autoritativo — los operadores lo publican de forma inconsistente, y "
                    "inventarlo convertiría un orden documentado en un número falso. Ponlo tú.",
    "multiplier": "Cuántas veces más se juega la combinación más popular frente a la menos "
                  "popular. Es el input menos defendible de todo el modelo, por eso el resultado "
                  "viene con banda en vez de con un solo número.",
    "split_table": "Ordenada por lo que vale el premio mayor para cada jugada *si gana*. Todas "
                   "ganan con la misma probabilidad: 1 en 15,401,568. Lo que cambia es entre "
                   "cuántos repartes. Si la banda baja/alta es tan ancha que te cambiaría la "
                   "decisión, este modelo no puede tomarla por ti.",
    "bias_components": "Los sesgos individuales antes de ponderar, cada uno de 0 a 1. `calendar` "
                       "es el que domina: los números 1-31 caben en una fecha y se juegan mucho "
                       "más que el 32-43, que no son menos probables.",

    # -- Registro
    "tab_registry": "Lo único que no se puede ajustar después: una predicción escrita **antes** de "
                    "que el sorteo existiera. Todo lo demás en este panel mira hacia atrás, y mirar "
                    "hacia atrás siempre se puede afinar sin querer.",
    "registry_record": "Registra una jugada contra un sorteo futuro. Queda con marca de tiempo y no "
                       "se puede editar ni borrar desde aquí. Un sorteo que ya pasó es rechazado — "
                       "no advertido, rechazado: una sola fila retroactiva vuelve inútil el archivo "
                       "entero.",
    "registry_label": "De dónde salió la jugada: el nombre del modelo, la estrategia, o 'yo'. Es lo "
                      "que después permite comparar fuentes entre sí. Si te arrepientes, registra "
                      "otra con etiqueta distinta — ambas quedan en el registro.",
    "registry_pending": "Predicciones cuyo sorteo todavía no ocurre, o que aún no se han puntuado. "
                        "Estas son las que valen: ya están escritas y todavía no sabes el resultado.",
    "registry_summary": "Resultado de las predicciones ya puntuadas contra la línea base del azar. "
                        "Lee primero el efecto mínimo detectable: con pocas predicciones, 'no le gana "
                        "al azar' es una afirmación sobre el tamaño de la muestra, no sobre las "
                        "predicciones.",
    "registry_mde": "La ventaja más pequeña que este número de predicciones puntuadas podría "
                    "revelar. A 3 sorteos por semana, un año son 156 observaciones, que dan para "
                    "detectar cerca de +23% y nada más fino.",
    "sensitivity_chart": "Tasa de detección contra fuerza del sesgo. Donde una curva cruza el 80% está "
                         "el umbral de sensibilidad de ese detector: por debajo de eso, un resultado "
                         "nulo suyo no descarta nada.",
    # -- Fútbol
    "fb_seasons": "Archivos de temporada de la carpeta indicada. Puedes elegir varios, pero solo se "
                  "cargan juntos si resuelven a la misma fuente de cuotas: mezclar cuotas de apertura "
                  "con cuotas de cierre pondría dos mercados distintos en una sola columna, y todo "
                  "modelo evaluado ahí se mediría contra dos barras a la vez.",
    "fb_closing_only": "Rechaza los archivos cuyas mejores cuotas son de apertura, en lugar de "
                       "cargarlos y producir una línea base que no aguanta la conclusión que alguien "
                       "va a sacar de ella. Las cuotas de cierre empiezan en la temporada 2019/20.",
    "fb_method": "Cómo se quita el margen de la casa para convertir cuotas en probabilidades. Los tres "
                 "métodos discrepan más en los no favoritos. Ninguno es correcto: se elige uno, se "
                 "dice cuál, y se comprueba que la conclusión no cambie al cambiarlo.",
    "fb_tab_datos": "Qué trae el archivo cargado y, sobre todo, de dónde salen sus cuotas — el dato "
                    "que decide si estos partidos pueden servir de línea base o solo de entrenamiento.",
    "fb_n_matches": "Partidos con resultado final. Los partidos sin marcador (el calendario de una "
                    "temporada en curso) se descartan al cargar: no son resultados.",
    "fb_odds_source": "La única fuente de cuotas que se usa para todo el archivo. Se prefiere cierre "
                      "sobre apertura, y un promedio de mercado sobre una sola casa. No hay respaldo "
                      "por fila: si a un partido le falta esa fuente, queda vacío.",
    "fb_closing": "Las cuotas de cierre son el precio después de que se movió el dinero, y son la "
                  "barra real. Las de apertura son blandas: ganarles suele ser ganarle a la primera "
                  "estimación de la casa, no al mercado.",
    "fb_coverage": "Qué proporción de los partidos tiene un trío de precios utilizable. Un trío "
                   "incompleto se borra entero, porque dos de tres precios no se pueden normalizar.",
    "fb_table": "Los partidos tal como quedaron tras el contrato: fecha, equipos, goles, resultado "
                "y el trío de cuotas de la fuente resuelta.",
    "fb_tab_mercado": "El sobrerredondeo es cuánto exceden de 1 las tres probabilidades implícitas: "
                      "el margen de la casa. Hay que quitarlo antes de que los precios signifiquen "
                      "algo como pronóstico.",
    "fb_overround": "Media del margen en estos partidos. Un mercado típico de primera división está "
                    "entre el 2% y el 8%; mucho más alto sugiere una casa poco competitiva o un "
                    "archivo de otra época.",
    "fb_overround_chart": "Cuántos partidos hay en cada nivel de margen. Una cola larga hacia la "
                          "derecha son partidos con menos liquidez, donde la casa se protege más.",
    "fb_tab_calibracion": "Si el mercado estuviera mal calibrado, habría un sesgo que explotar sin "
                          "modelo alguno. Esta curva comprueba que no lo hay, que es justamente lo "
                          "que hace difícil la barra.",
    "fb_calibration": "Cada punto agrupa los partidos por la probabilidad que el mercado le dio al "
                      "local y compara esa media con la frecuencia real de victoria local. Cerca de "
                      "la diagonal = calibrado. Con una temporada, cada punto tiene pocos partidos: "
                      "la dispersión es ruido.",
    "fb_tab_metodos": "El mismo partido bajo las tres normalizaciones. Sirve para ver de qué tamaño "
                      "es la discrepancia antes de apoyar una conclusión en una de ellas.",
    "fb_match_pick": "Elige un partido para ver sus tres probabilidades bajo cada método.",
    "fb_tab_resultados": "Descriptivo, no predictivo: cómo terminaron estos partidos. Nada de esta "
                         "pestaña compara un modelo contra nada.",
    "fb_outcome_share": "Frecuencia observada del resultado en estos partidos. La referencia de la "
                        "primera división inglesa a largo plazo ronda 45% local, 25% empate, 30% "
                        "visitante.",
    "fb_outcomes": "Reparto de los tres resultados. La ventaja de local es real y grande, y es la "
                   "razón por la que «siempre local» parece un modelo hasta que se lo compara con "
                   "el precio.",
    "fb_goals": "Goles por partido de cada lado. La referencia a largo plazo ronda 1,54 del local "
                "y 1,19 del visitante.",
    "fb_observed_vs_market": "La media de las probabilidades del mercado frente a la frecuencia "
                             "observada. Que coincidan no es un resultado: el mercado acierta el "
                             "agregado sin esfuerzo. El veredicto partido a partido, con regla de "
                             "puntuación y modelo, está en «¿Le gana este modelo al mercado?».",

    "fb_source_toggle": (
        "Europa usa los archivos de liga de football-data.co.uk (cuotas de cierre desde "
        "2019/20). Colombia usa el archivo «extra» new/COL.csv: mismo deporte, pero solo "
        "cuotas de apertura — la línea base es blanda y ninguna ventaja medida ahí está probada."
    ),
    "fb_forecast_tab": (
        "Un pronóstico para un partido concreto. Las probabilidades del modelo se muestran "
        "siempre junto a las del mercado; el modelo por sí solo no dice si acierta — eso lo "
        "responde la pestaña Resultados, sobre muchos partidos y con corrección."
    ),
    "fb_h2h": (
        "Historial y forma reciente. Es descriptivo: no es una predicción, y una racha corta "
        "es en su mayor parte ruido."
    ),
    "fb_form": "Últimos partidos de cada equipo antes de esta fecha. V/E/D desde la óptica del equipo.",
    "fb_model_1x2": (
        "Probabilidad de local / empate / visitante según Dixon-Coles ajustado a las temporadas "
        "cargadas. No incorpora lesiones, alineaciones ni el mercado."
    ),
    "fb_scoreline_grid": (
        "Probabilidad de cada marcador exacto. El más probable rara vez pasa del 10-12%: sirve "
        "para ver la forma de la distribución, no para apostar a un resultado exacto."
    ),
    "fb_most_likely_scores": "Los marcadores con más probabilidad. Suman una fracción pequeña del total.",
    "fb_over_under": "Probabilidad de más/menos de 2.5 goles, derivada de la misma matriz de marcadores.",
    "fb_btts": "Probabilidad de que ambos equipos marquen.",
    "fb_your_odds": (
        "Cuotas decimales actuales de una casa de apuestas para este partido. Si las pones, se "
        "les quita el margen y se comparan con el modelo. Es un partido y una comparación sin "
        "corregir: no es un veredicto."
    ),
    "fb_model_vs_market": (
        "Modelo contra mercado para este partido. Una diferencia a favor del modelo en un "
        "partido no significa nada — hace falta la evaluación de la pestaña Resultados."
    ),
    "fb_half_life": (
        "Vida media en días del peso temporal: un partido de hace tantos días pesa la mitad. "
        "Más bajo = más peso a la forma reciente. 0 = todos los partidos pesan igual."
    ),
    "fb_eval_tab": (
        "Puntuación fuera de muestra del modelo contra la cuota de cierre, con el veredicto "
        "naive y el corregido. Es el único número de esta página que dice si el modelo vale algo."
    ),
    "fb_skill_score": (
        "1 − score(modelo)/score(mercado). Positivo = el modelo puntuó mejor. En una sola "
        "temporada un valor positivo pequeño está dentro del ruido."
    ),
    "fb_beats_market": (
        "Veredicto de una prueba pareada de una cola. Mira siempre la columna corregida: con "
        "varios métodos de de-margen probados a la vez, la naive se supera por azar."
    ),
    "fb_model_calibration": (
        "Cuando el modelo dice 60% de victoria local, ¿gana el local ~60% de las veces? "
        "Con una temporada cada punto tiene pocos partidos, así que la dispersión es ruido."
    ),
    "fb_model_calibration_curve": (
        "Cada punto agrupa los partidos por la probabilidad que el modelo le dio al local y "
        "compara esa media con la frecuencia real. Ajustado dentro de muestra: favorece al modelo."
    ),

    # -- Fútbol · varios modelos, mezcla y valor
    "fb_models_table": "Cada modelo puntuado contra la misma cuota de cierre y sobre los mismos "
                       "partidos. Un partido que algún modelo no puede predecir se salta para "
                       "todos: dos modelos medidos sobre partidos distintos no son comparables, y "
                       "eso no se ve en la forma de la tabla.",
    "fb_models_pick": "Cuáles medir. El umbral corregido se divide entre cuántos elijas, así que "
                      "añadir un modelo endurece la prueba para todos — que es justo lo correcto.",
    "fb_elo": "Elo: una sola nota de fuerza por equipo, que sube o baja tras cada partido según el "
              "resultado y contra quién. Es la línea base barata que un modelo serio debe superar. "
              "Solo produce local/empate/visitante: una nota única no puede saber cómo se reparten "
              "los goles.",
    "fb_elo_ranking": "Fuerza estimada de cada equipo al final de los partidos cargados. Los puntos "
                      "no tienen unidad interpretable por sí solos; lo que significa algo es la "
                      "diferencia entre dos equipos.",
    "fb_blend": "Mezcla del modelo con el mercado. Con peso 0 la mezcla **es** el mercado y puntúa "
                "exactamente igual; si subir el peso mejora la puntuación, el modelo aporta algo "
                "que el precio no tenía. Es una pregunta más útil que «¿le gana al mercado?», que "
                "casi nada responde que sí.",
    "fb_blend_weight": "Cuánto peso lleva el modelo frente al mercado. 0 = solo mercado, 1 = solo "
                       "modelo.",
    "fb_pool": "Lineal promedia las probabilidades y siempre queda entre las dos fuentes. "
               "Logarítmica promedia en escala log: se apoya más en lo que ambas favorecen y "
               "castiga duro lo que una de las dos casi descartó. Si la conclusión cambia al "
               "cambiar de regla, el hallazgo es sobre la regla, no sobre el modelo.",
    "fb_tab_valor": "La parte que puede hacer daño. Todo lo de esta pestaña vale **solo si el "
                    "modelo es bueno**, y lo único que dice algo sobre eso está en Resultados.",
    "fb_two_bars": "Las dos barras son distintas y confundirlas es toda la trampa. Para juzgar un "
                   "**modelo** se compara contra el precio sin margen. Para juzgar una **apuesta** "
                   "se compara contra 1/cuota, porque el margen lo pagas tú. Entre las dos hay un "
                   "hueco donde el modelo discrepa del mercado y la discrepancia no alcanza a "
                   "pagar la comisión.",
    "fb_value_table": "Por resultado: lo que dice el modelo, lo que dice el mercado, lo que "
                      "tendrías que superar para ganar dinero, y el estado que sale de comparar "
                      "los tres.",
    "fb_kelly": "Kelly es la apuesta que maximiza el crecimiento a largo plazo **dada una ventaja "
                "real**. Sobre una ventaja imaginaria no es que no ayude: sube la apuesta justo "
                "cuando el modelo está más seguro y más equivocado. Por eso aquí es un cuarto de "
                "Kelly y por eso no se muestra sin el veredicto al lado.",
    "fb_margin_cost": "Cuánta probabilidad se come el margen, por resultado. Es el ancho del hueco "
                      "de «solo discrepancia», y verlo como número es lo que impide que «mi modelo "
                      "dice 38% y el mercado 35%» se lea como una apuesta.",

    # -- Ciclismo · línea base, modelo y evaluación
    "cy_tab_pronostico": "Quién debería ganar, según el ranking previo y según el modelo. Las dos "
                         "cosas se muestran juntas: una probabilidad de un modelo sin su línea "
                         "base al lado no dice nada.",
    "cy_baseline_pick": "Sobre qué se construye la línea base. El ranking previo se arma con los "
                        "resultados **anteriores** a la carrera elegida, nunca con la carrera "
                        "misma.",
    "cy_race_pick": "Qué carrera pronosticar. Todo lo que se muestre usa solo lo que se sabía "
                    "antes de su fecha.",
    "cy_uniform_warning": "Un sorteo uniforme sobre la lista de salida da 0,55% a cada uno de ~180 "
                          "ciclistas. Cualquier pronóstico le gana sabiendo un solo nombre, así "
                          "que **no es una línea base**: está aquí para que el error tenga nombre.",
    "cy_win_probs": "Probabilidad de ganar de cada ciclista, bajo Plackett-Luce: la fuerza de cada "
                    "uno dividida entre la fuerza total. Quitas al ganador y repites para el "
                    "segundo puesto — así una sola cifra por ciclista genera un orden completo.",
    "cy_top_n": "Probabilidad de acabar entre los primeros N, estimada por simulación. No hay "
                "fórmula barata más allá del primer puesto. Con pocas simulaciones el error es de "
                "un punto largo, así que no partas pelos con diferencias pequeñas.",
    "cy_model_vs_baseline": "El modelo ajustado frente al ranking previo, ciclista a ciclista. "
                            "Donde discrepan está lo que el modelo cree ver; si vale algo o no lo "
                            "dice la pestaña Evaluación, no esta.",
    "cy_tab_evaluacion": "El veredicto: se recorre carrera por carrera, cada pronóstico se "
                         "construye **solo** con lo anterior, y se puntúa contra el ranking previo.",
    "cy_metric": "Plackett-Luce puntúa el orden de llegada completo y es el veredicto. Las otras "
                 "dos solo miran quién ganó. Las correlaciones de rango se leen fácil y premian "
                 "acertar el centro del pelotón, que es la parte que no le importa a nadie — son "
                 "diagnóstico, no veredicto.",
    "cy_eval_table": "Menor es mejor en todas las métricas. Lee la columna corregida: el umbral ya "
                     "está dividido entre cuántos pronosticadores compiten.",
    "cy_eval_chart": "Puntuación por carrera de cada pronosticador contra el ranking. Por debajo "
                     "de la línea del ranking es mejor que él ese día.",
    "cy_sample_warning": "Una gran vuelta son 21 carreras puntuadas y una temporada de clásicas son "
                         "unas pocas decenas. Con veintitantas observaciones solo se distingue una "
                         "diferencia grande, así que aquí un «no le gana» habla del tamaño de la "
                         "muestra todavía más que en las otras pestañas.",

    # -- Ciclismo
    "cy_files": "Archivos de resultados de la carpeta indicada. Un archivo contiene un solo tipo de "
                "resultado, y no se pueden cargar juntos tipos distintos: un puesto en una etapa y "
                "un puesto en la general son cantidades diferentes en la misma columna `rank`.",
    "cy_finishers_only": "Oculta en la tabla a quienes no clasificaron. El filtro se aplica después "
                         "de las comprobaciones, para que la advertencia de «no hay ni un abandono» "
                         "siga saltando sobre el archivo tal como se publicó.",
    "cy_tab_datos": "Qué trae el archivo y las tres cosas que el contrato protege y que no se ven "
                    "en la forma de la tabla: el tipo de resultado, los no clasificados y si los "
                    "tiempos son coherentes.",
    "cy_n_rows": "Una fila por ciclista y por resultado, incluidos los que no clasificaron.",
    "cy_kind": "Resultado de etapa, clásica de un día o clasificación general. El `rank` significa "
               "algo distinto en cada uno, así que un archivo solo puede tener uno.",
    "cy_non_finishers": "Ciclistas sin puesto: abandonos, no salidos, descalificados o fuera de "
                        "control. Se conservan a propósito.",
    "cy_missing_times": "Filas sin tiempo. Las diferencias que no se pudieron anclar al tiempo del "
                        "ganador quedan vacías en lugar de inventarse: un hueco visible se recupera, "
                        "un tiempo inventado no.",
    "cy_violations": "Ciclistas cronometrados más rápido que alguien que quedó por delante. No puede "
                     "pasar en una clasificación real, así que es la huella de diferencias al ganador "
                     "guardadas en una columna que significa tiempo total.",
    "cy_table": "Las filas tal como quedaron tras el contrato, con el tiempo total formateado.",
    "cy_tab_abandonos": "Quién no llega al final, y por qué importa: los abandonos no son aleatorios.",
    "cy_status_counts": "Cuántos ciclistas hay en cada estado. «Fuera de control» llegó a meta pero "
                        "quedó eliminado por tiempo, así que no cuenta como clasificado.",
    "cy_attrition": "Cuántos ciclistas siguen clasificando en cada etapa. La caída es la carrera "
                    "desgastando al pelotón, y es información sobre la carrera, no ruido a limpiar.",
    "cy_abandons_per_stage": "Abandonos por etapa. En una carrera real se concentran en las etapas "
                             "duras y en las caídas — los abandonos están correlacionados entre "
                             "ciclistas, algo que los datos sintéticos de este repositorio no simulan.",
    "cy_tab_tiempos": "Cómo se abre la carrera por detrás del primero. Sirve para ver de un golpe si "
                      "los tiempos son totales coherentes o diferencias mal guardadas.",
    "cy_stage_pick": "Elige la etapa cuyas diferencias quieres ver.",
    "cy_gaps": "Segundos de cada puesto respecto al primero. Un tramo plano al principio es un grupo "
               "con el mismo tiempo; un salto es donde se partió la carrera. Si la curva baja en "
               "algún punto, los tiempos están mal.",
    "cy_leader_time": "Tiempo total del primer clasificado, que es el ancla con la que el scraper "
                      "convierte cada diferencia publicada en un tiempo total.",
    "cy_same_time": "Cuántos ciclistas comparten el tiempo del primero. En una llegada en pelotón "
                    "son la mayoría del grupo, y es lo que significa el marcador «,,» en la página.",
}


# The always-visible guided box, keyed exactly like HELP.
#
# Three fields, and the third is the one this project exists for. "ojo" is not a
# disclaimer bolted on at the end — it is the sentence that stops a reader
# walking away with the conclusion the surface *looks* like it supports. A
# frequency chart that says "el 7 salió más" and nothing else has misinformed
# someone; the same chart with "eso no lo hace más probable" has not.
#
# Written for a reader who is technically competent and has never met a p-value.
# Concrete numbers over abstractions, second person, no term used before it is
# explained. A key with no entry here simply renders no box, so partial coverage
# is fine: the tab-level and conceptually hard keys are the ones that need it.
PLAIN = {
    # -- Baloto · Resumen
    "tab_resumen": {
        "veo": "Cuántos sorteos cargaste, de qué fechas, y una prueba que revisa si tus números "
               "salen repartidos de forma pareja.",
        "concluyo": "Si la prueba dice que todo se ve parejo, tus datos son sanos y puedes seguir "
                    "al resto del panel. Es el chequeo previo, como mirar que la báscula esté en "
                    "cero antes de pesar algo.",
        "ojo": "Que los datos sean sanos **no** significa que se pueda predecir nada. Significa lo "
               "contrario: que el sorteo se comporta como debe, o sea impredecible.",
    },
    "pooled_test": {
        "veo": "Una sola pregunta: ¿cada número del 1 al 43 sale más o menos la misma cantidad de "
               "veces? El resultado es un número entre 0 y 1 llamado *p-valor*.",
        "concluyo": "**p-valor alto (por encima de 0.05) = todo normal.** Es el resultado que "
                    "esperas. Un p-valor bajo sería rarísimo y lo más probable es que apunte a un "
                    "problema en el archivo de datos, no a una lotería que se pueda vencer.",
        "ojo": "El p-valor **no** es «la probabilidad de que el sorteo sea justo». Es «qué tan "
               "raros serían estos datos si el sorteo fuera justo». Un 0.40 quiere decir «nada "
               "raro», no «40% de probabilidad de ser justo».",
    },

    # -- Baloto · Probabilidades
    "tab_probabilidades": {
        "veo": "Las probabilidades exactas de cada categoría de premio y cuánto vale jugar, "
               "dados los premios que tú escribas.",
        "concluyo": "Esta es la única pestaña con respuestas **definitivas**. No hay modelo, no "
                    "hay estimación y no depende de tu histórico: es la misma aritmética que usa "
                    "el operador para fijar los premios.",
        "ojo": "Nada de lo que hagas mueve estos números. No hay forma de elegir 5 números que "
               "sea mejor que otra: las 15.401.568 combinaciones son exactamente igual de "
               "probables.",
    },
    "ev_section": {
        "veo": "Cuánto devuelve en promedio un tiquete a la larga, comparado con lo que cuesta.",
        "concluyo": "Si el «valor esperado» es negativo, cada tiquete pierde esa plata en "
                    "promedio. Jugando muchas veces, eso es lo que efectivamente pasa.",
        "ojo": "«En promedio» no quiere decir que pierdas esa cantidad cada vez. Casi siempre "
               "pierdes todo el tiquete, y muy de vez en cuando ganas algo. El promedio es lo "
               "que queda al sumar las dos cosas.",
    },

    # -- Baloto · Frecuencia
    "tab_frecuencia": {
        "veo": "Cuántas veces salió cada número y cuánto lleva sin salir.",
        "concluyo": "Sirve para conocer tu histórico. Es una foto del pasado, nada más.",
        "ojo": "Que un número haya salido más **no** lo hace más probable, y que lleve mucho sin "
               "salir **no** lo hace «estar atrasado». Las balotas no recuerdan. Esta es la "
               "confusión de la que viven casi todos los «sistemas» de lotería.",
    },
    "tab_hotcold": {
        "veo": "Qué números vienen saliendo por encima o por debajo de su promedio histórico en "
               "los últimos sorteos.",
        "concluyo": "Que unas barras estén arriba y otras abajo es exactamente lo que produce el "
                    "azar puro. Si tiras una moneda 20 veces casi nunca salen 10 y 10.",
        "ojo": "«Caliente» y «frío» no son propiedades del número, son ruido de la ventana que "
               "elegiste. Muévela unos sorteos y la lista cambia. La pestaña **Jugadas → Medir "
               "estrategias** mide si jugar los calientes sirve de algo: no sirve.",
    },

    # -- Baloto · Aleatoriedad
    "tab_aleatoriedad": {
        "veo": "Las pruebas formales que buscan cualquier estructura aprovechable: repeticiones, "
               "rachas y memoria entre un sorteo y el siguiente.",
        "concluyo": "Lo sano es que todas digan «no hay nada». Si un sorteo tuviera memoria, un "
                    "modelo de series de tiempo tendría algo que aprender; estas pruebas son las "
                    "que responden si la tiene.",
        "ojo": "Al correr seis pruebas a la vez, es normal que ~1 de cada 20 marque «No» por puro "
               "azar. Una sola casilla en rojo no es un hallazgo.",
    },
    "acf_section": {
        "veo": "Si el resultado de un sorteo dice algo sobre el siguiente, el de dos atrás, y así.",
        "concluyo": "Barras dentro de la banda punteada = sin memoria detectable. Es la prueba "
                    "directa de si ARIMA, Prophet o cualquier modelo de series de tiempo tiene "
                    "materia prima aquí.",
        "ojo": "Un ACF plano **no** es un fallo del panel: es el retrato exacto de un proceso sin "
               "memoria, que es lo que un sorteo justo debe ser.",
    },

    "structure_section": {
        "veo": "Cada sorteo reducido a un solo número — la suma de las 5 balotas, cuántas son "
               "impares, cuántas caben en una fecha — comparado contra la forma exacta que produce "
               "la combinatoria.",
        "concluyo": "Si tus barras siguen la curva exacta, el sorteo no solo saca cada número las "
                    "veces que toca: también los **combina** como debe. Es una pregunta distinta "
                    "de la del Resumen, y una máquina puede pasar aquella y fallar ésta.",
        "ojo": "Que la suma 110 sea más frecuente que la 15 **no** significa que convenga jugar "
               "sumas del centro. Hay 14.090 combinaciones que suman 110 y **una sola** que suma "
               "15; cada una de las 14.091 es igual de probable. Lo único que cambia es con cuánta "
               "gente repartirías.",
    },

    # -- Baloto · Forecast
    "tab_forecast": {
        "veo": "Un modelo entrenado sobre tu histórico proponiendo números para el próximo sorteo.",
        "concluyo": "Sirve para ver qué produce cada modelo y cuánto se demora. Como ejercicio de "
                    "forecasting es legítimo.",
        "ojo": "Estos números **no** son mejores que cinco elegidos al azar, y la pestaña "
               "**Backtest** lo demuestra sobre tus propios datos. Úsalos si te divierte, no "
               "porque el modelo sepa algo.",
    },

    # -- Baloto · Jugadas
    "tab_jugadas": {
        "veo": "Tres cosas: generar jugadas, ver cómo le habría ido a una jugada tuya en el "
               "pasado, y medir si una forma de elegir números le gana al azar.",
        "concluyo": "Generar números está perfecto. Lo que ninguna estrategia logra es hacer una "
                    "jugada más probable que otra, y aquí eso se mide en vez de afirmarse.",
        "ojo": "Si una estrategia sale ganadora en una corrida, míra la sección de estabilidad "
               "antes de creerle: una estrategia sin ninguna ventaja parece ganadora ~1 de cada "
               "20 corridas.",
    },
    "experiment_intro": {
        "veo": "Para cada sorteo del pasado se generan jugadas usando **solo** lo que se sabía "
               "antes de ese sorteo, y se cuenta cuántos números acertaron.",
        "concluyo": "Es la comparación honesta: si una estrategia tuviera ventaja real, aquí se "
                    "vería. La referencia es 0.58 aciertos de 5, que es lo que da el azar.",
        "ojo": "Lee la columna **corregida**, no la suelta. Al probar varias estrategias a la vez, "
               "que alguna pase el filtro por suerte deja de ser improbable.",
    },
    "stability_section": {
        "veo": "El mismo experimento repetido muchas veces con datos distintos, contando cuántas "
               "veces cada estrategia salió «ganadora».",
        "concluyo": "`random` no puede tener ventaja, así que su tasa es tu piso medido de falsas "
                    "alarmas. Una estrategia que no marque claramente más seguido que ella no ha "
                    "demostrado nada.",
        "ojo": "Esto es exactamente cómo se convence la gente de que su sistema funciona: corren "
               "el experimento una vez, les da bien, y no lo vuelven a correr.",
    },
    "tab_split": {
        "veo": "Cuánto valdría el premio mayor para cada jugada **si gana**, según cuánta gente "
               "suele jugar esos mismos números.",
        "concluyo": "Esta es la **única palanca real** de todo el panel. No mejora tu "
                    "probabilidad de ganar; mejora cuánto te llevas cuando ganas, porque el "
                    "acumulado se reparte entre todos los que acertaron.",
        "ojo": "Los pesos de este modelo no se pueden calibrar — haría falta saber qué tiquetes "
               "compró la gente, y ningún operador lo publica. La **dirección** es sólida (las "
               "fechas se sobrejuegan muchísimo); cualquier cifra concreta es aproximada, por eso "
               "viene con banda.",
    },

    # -- Baloto · Backtest
    "tab_backtest": {
        "veo": "El veredicto del proyecto. Cada modelo se entrena **solo** con sorteos anteriores "
               "al que intenta adivinar, y se cuenta cuántos números acierta de verdad.",
        "concluyo": "Lo que importa no es «cuántos aciertos» sino **cuántos más que el azar**. El "
                    "azar puro da 0.58 aciertos de 5 en promedio; un modelo con ventaja real "
                    "daría consistentemente más.",
        "ojo": "«Ningún modelo le gana al azar» significa **ninguna ventaja mayor que la que estos "
               "datos alcanzan a ver**, no «ninguna ventaja». La pestaña **Potencia** dice "
               "exactamente de qué tamaño es esa frontera.",
    },

    # -- Baloto · Potencia
    "tab_power": {
        "veo": "Dos preguntas que van **antes** de cualquier resultado: ¿qué tan grande tendría "
               "que ser una ventaja para que estos datos la vieran? y ¿estas pruebas sirven para "
               "ver una ventaja cuando sí existe?",
        "concluyo": "Sin esto, un «no encontré nada» no vale. Un detector de metales apagado "
                    "tampoco encuentra nada.",
        "ojo": "La resolución mejora con la **raíz** de la cantidad de sorteos: cuadruplicar tu "
               "histórico solo reduce a la mitad la ventaja más pequeña que podrías detectar.",
    },
    "mde_section": {
        "veo": "La ventaja más pequeña que tu cantidad de sorteos podría distinguir del azar de "
               "forma confiable.",
        "concluyo": "Es el número que le pone resolución a un resultado nulo. Con 200 sorteos "
                    "evaluados solo verías ventajas grandes; todo lo más fino es invisible.",
        "ojo": "«Invisible para esta prueba» no es lo mismo que «no existe». Son dos frases muy "
               "distintas y la mayoría de los análisis las confunde.",
    },
    "sensitivity_section": {
        "veo": "Se generan sorteos con un sesgo **plantado a propósito** y se mide cuántas veces "
               "lo detecta cada prueba.",
        "concluyo": "Es el control de calidad de todo lo demás. Empieza siempre por las filas de "
                    "fuerza 0: ahí no hay sesgo, así que las tres tasas deben quedar bajas. Si "
                    "alguna no lo hace, esa prueba está rota y el resto de la tabla no dice nada.",
        "ojo": "`random` debe quedarse abajo **incluso con sesgo fuerte**. No es un fallo: un "
               "tiquete uniforme no sabe qué números están favorecidos, así que no puede "
               "aprovecharlos. Es el control correcto.",
    },

    # -- Baloto · Registro
    "tab_registry": {
        "veo": "Predicciones escritas **antes** de que el sorteo existiera, con fecha y sin "
               "posibilidad de editarlas.",
        "concluyo": "Todo lo demás en este panel mira hacia atrás, y mirar hacia atrás siempre se "
                    "puede afinar sin querer. Esta pestaña es la única evidencia que no admite "
                    "ajustes posteriores.",
        "ojo": "Un registro con pocas predicciones no demuestra nada en ninguna dirección. A 3 "
               "sorteos por semana, un año son 156 filas, que apenas alcanzan para detectar una "
               "ventaja de +23% o mayor.",
    },

    # -- Fútbol
    "fb_tab_datos": {
        "veo": "Qué partidos trae el archivo y, sobre todo, **de dónde salen sus cuotas**.",
        "concluyo": "Ese dato decide todo lo demás. Las cuotas de **cierre** (justo antes del "
                    "partido, después de que se movió el dinero) son la barra real. Las de "
                    "**apertura** son la primera estimación de la casa y son mucho más blandas.",
        "ojo": "Aquí la barra **no** es el azar: es el mercado. Un modelo que le gana a «siempre "
               "local» no ha encontrado nada — la ventaja de local ya está en el precio.",
    },
    "fb_tab_mercado": {
        "veo": "El margen de la casa de apuestas, y si sus precios están bien calibrados.",
        "concluyo": "Las tres probabilidades implícitas de un partido suman más de 1 — ese exceso "
                    "(2% a 8% típico) es la comisión de la casa. Hay que quitarla antes de que "
                    "los precios signifiquen algo como pronóstico.",
        "ojo": "Comparar un modelo contra `1/cuota` sin quitar el margen lo mide contra una barra "
               "deliberadamente equivocada a favor de la casa. Es el error más común del área.",
    },
    "fb_forecast_tab": {
        "veo": "El pronóstico del modelo para un partido concreto, siempre al lado del mercado.",
        "concluyo": "Sirve para ver en qué se parecen y en qué discrepan. Cuando discrepan mucho, "
                    "lo interesante es entender por qué.",
        "ojo": "Un partido no prueba nada, gane quien gane. El único número de esta página que "
               "dice si el modelo vale algo está en **Resultados**, medido sobre muchos partidos "
               "y con corrección.",
    },
    "fb_eval_tab": {
        "veo": "El modelo puntuado contra la cuota de cierre en partidos que **no** vio al "
               "entrenar.",
        "concluyo": "Éste es el veredicto. Mira siempre la fila **corregida**: con varios modelos "
                    "y varios métodos probados a la vez, que alguno pase el filtro por suerte es "
                    "esperable.",
        "ojo": "Un skill score positivo pequeño con intervalo que cruza el cero **no** es una "
               "ventaja: es ruido que salió con el signo favorable.",
    },
    "fb_tab_resultados": {
        "veo": "Cómo terminaron estos partidos, sin más.",
        "concluyo": "Puramente descriptivo. Sirve para ubicarte: la referencia de primera división "
                    "inglesa ronda 45% local, 25% empate, 30% visitante.",
        "ojo": "Que la media del mercado coincida con la frecuencia observada **no** es un "
               "resultado: acertar el agregado no le cuesta nada al mercado. Lo difícil es "
               "acertar partido a partido.",
    },

    "fb_tab_valor": {
        "veo": "Qué pasaría si apostaras según el modelo a las cuotas que escribas: cuánta "
               "ventaja habría y cuánto tocaría apostar.",
        "concluyo": "**Nada de esto vale si el modelo no le gana al mercado**, y eso solo lo dice "
                    "la pestaña Resultados. Lo que sí es cierto pase lo que pase es el hueco del "
                    "margen: para ganar dinero no basta con discrepar del mercado, hay que "
                    "discrepar lo suficiente como para pagar la comisión de la casa.",
        "ojo": "Kelly **no** es una protección. Es la apuesta óptima suponiendo que tu ventaja es "
               "real; si no lo es, sube la apuesta justo cuando el modelo está más seguro y más "
               "equivocado, y convierte una pérdida lenta en una rápida. Por eso aquí es un cuarto "
               "de Kelly y por eso no aparece sin el veredicto al lado.",
    },

    # -- Ciclismo
    "cy_tab_pronostico": {
        "veo": "Quién debería ganar y quién debería entrar en el top 10, según el ranking previo "
               "y según el modelo ajustado.",
        "concluyo": "Aquí el objetivo no es un resultado de tres vías sino un **orden de llegada** "
                    "de ~180 ciclistas. Una sola cifra de fuerza por ciclista genera el orden "
                    "completo: el más fuerte gana con probabilidad proporcional a su fuerza, se le "
                    "quita de la lista, y se repite para el segundo puesto.",
        "ojo": "Que el modelo dé 12% al favorito no es poco ni mucho por sí solo. Con 180 "
               "corredores, un sorteo uniforme daría 0,55% — por eso ese sorteo **no** es la línea "
               "base contra la que hay que medirse, sino el ranking previo.",
    },
    "cy_tab_evaluacion": {
        "veo": "Cada carrera puntuada dos veces: con el pronóstico del modelo y con el ranking "
               "previo, usando solo lo que se sabía antes de esa carrera.",
        "concluyo": "Menor es mejor. La diferencia media entre las dos puntuaciones, con su "
                    "intervalo, es lo único que dice si el modelo aporta algo sobre el ranking.",
        "ojo": "Son veintitantas carreras. Con esa cantidad solo se distingue una diferencia "
               "grande, así que un «no le gana» aquí habla sobre todo del tamaño de la muestra. Y "
               "ojo con los sprints: en una llegada masiva el orden es casi ruido, y ningún "
               "pronóstico puede ni debe ganarle al ranking ahí.",
    },
    "cy_tab_datos": {
        "veo": "Qué trae el archivo y las tres trampas que el contrato de datos protege y que no "
               "se ven mirando la tabla.",
        "concluyo": "Las tres son invisibles en la forma de los datos y arruinan cualquier "
                    "análisis posterior: mezclar puestos de etapa con puestos de la general, "
                    "perder a los que abandonaron, y guardar diferencias al ganador en una "
                    "columna que significa tiempo total.",
        "ojo": "Aquí el objetivo no es un resultado de tres vías sino un **orden de llegada** de "
               "~180 ciclistas. Una probabilidad uniforme sobre la lista de salida no es una "
               "línea base: es una forma de hacer ver brillante a cualquier modelo.",
    },
    "cy_tab_abandonos": {
        "veo": "Quién no llega al final y en qué etapas se cae la gente.",
        "concluyo": "Un quinto de la lista de salida de una gran vuelta puede no terminarla, y los "
                    "abandonos se concentran en los ciclistas de peor forma.",
        "ojo": "Filtrarlos convierte «predecir el orden de llegada» en «predecir el orden entre "
               "los que llegaron», que es un problema bastante más fácil y sobre el que nadie "
               "puede apostar.",
    },
    "cy_tab_tiempos": {
        "veo": "Cómo se abre la carrera por detrás del primero, puesto por puesto.",
        "concluyo": "Un tramo plano al principio es un grupo que llegó junto; un salto es donde se "
                    "partió la carrera. En una llegada masiva, la mayoría comparte el tiempo del "
                    "ganador y eso es correcto.",
        "ojo": "Si la curva **baja** en algún punto, alguien está cronometrado más rápido que "
               "quien quedó por delante. Eso es imposible en una clasificación real y es la "
               "huella de diferencias guardadas como si fueran tiempos totales.",
    },
}

# A one-line "how to read it" printed under a chart's title, before the figure.
#
# Keyed like HELP, and pulled in by `chart()` automatically, so a chart gets it
# without a new argument at the call site. Different job from HELP: the ⓘ says
# what the chart means and does not mean, this says where to point your eyes.
READ = {
    "category_chart": "La escala vertical es logarítmica — cada marca vale 10× la anterior. Las "
                      "barras más bajas son las que más pagan, y por eso están tan abajo.",
    "freq_chart": "Compara cada barra con la línea punteada, que es lo que saldría si todo fuera "
                  "perfectamente parejo. Ninguna cae justo encima: esa diferencia es ruido normal.",
    "hotcold_chart": "Rojo = salió más que su promedio en la ventana reciente; azul = menos. "
                     "Mueve el deslizador y verás que la lista cambia sola.",
    "acf_chart": "Si todas las barras caen dentro de las dos líneas punteadas, no hay memoria "
                 "entre sorteos. Es lo que esperas ver.",
    "sum_chart": "Las barras son tus sorteos; la línea es la forma exacta. Deberían seguirse. La "
                 "campana no la produce ningún ajuste: sale de contar combinaciones.",
    "parity_chart": "Barras contra puntos. 2 o 3 impares domina porque hay muchas más formas de "
                    "repartir 5 balotas así que de sacarlas todas impares.",
    "calendar_chart": "Barras contra puntos. Fíjate en cuánto pesa la parte de la izquierda: esos "
                      "son los sorteos con pocos números «de fecha», los que menos gente juega.",
    "history_chart": "La barra de «sin premio» domina siempre. Compárala con las probabilidades "
                     "exactas de la pestaña Probabilidades: cuadran.",
    "strategy_chart": "Las dos barras de cada estrategia deberían quedar casi iguales. Si una se "
                      "despega, mira primero el intervalo de confianza en la tabla de abajo.",
    "summary_chart": "Barra del modelo contra barra del azar, por modelo. Casi iguales es el "
                     "resultado correcto para una lotería justa.",
    "holdout_chart": "Cada línea es un modelo. Los picos por encima y por debajo de la línea "
                     "punteada son varianza normal, no rachas.",
    "power_chart": "Busca dónde la curva cruza la línea del 80%. A la izquierda de ese punto tu "
                   "prueba está prácticamente ciega.",
    "sensitivity_chart": "Donde una curva cruza el 80% está el umbral de ese detector. La curva "
                         "de `random` debe quedarse abajo del todo: ése es el control.",
    "split_table": "Las barras llevan bigotes porque el modelo da una **banda**, no una cifra. Si "
                   "la banda es tan ancha que te cambiaría la decisión, este modelo no puede "
                   "tomarla por ti.",
    "fb_overround_chart": "El eje horizontal es la comisión de la casa. La cola hacia la derecha "
                          "son partidos con menos dinero apostado, donde la casa se protege más.",
    "fb_calibration": "Los puntos deberían caer sobre la diagonal. Que lo hagan es justamente lo "
                      "que vuelve difícil esta barra: no hay un sesgo obvio que explotar.",
    "fb_model_calibration_curve": "Misma lectura que la curva del mercado, pero ajustada sobre los "
                                  "mismos partidos que muestra — eso favorece al modelo.",
    "fb_outcomes": "La barra de local siempre gana. Esa ventaja es real y grande, y ya está "
                   "metida en el precio: no es una oportunidad.",
    "fb_observed_vs_market": "Dos barras casi iguales es lo esperado y no prueba nada.",
    "fb_scoreline_grid": "Fila = goles del local, columna = goles del visitante. Lo más oscuro es "
                         "lo más probable, y rara vez pasa del 10-12%.",
    "fb_model_vs_market": "Donde las dos barras discrepan está lo que el modelo cree que el "
                          "mercado no ve. En un partido suelto, casi siempre es el modelo el que "
                          "está equivocado.",
    "cy_attrition": "La línea solo puede bajar. La caída es la carrera desgastando al pelotón — "
                    "información sobre la carrera, no ruido que haya que limpiar.",
    "cy_abandons_per_stage": "Los picos son las etapas duras y las caídas. Los abandonos van "
                             "correlacionados entre ciclistas.",
    "cy_gaps": "Solo puede subir. Un tramo plano es un grupo con el mismo tiempo; un escalón es "
               "donde se rompió la carrera. Si baja, los tiempos están mal.",
    "fb_elo_ranking": "Lo que importa no es el número sino la distancia entre dos equipos: 100 "
                      "puntos de diferencia son aproximadamente 64% de cuota esperada.",
    "fb_margin_cost": "Cuanto más alta la barra, más tiene que discrepar el modelo en ese "
                      "resultado para que apostarlo tenga sentido.",
    "cy_win_probs": "Compara la altura de las barras con la línea del sorteo uniforme: lo que "
                    "sobresalga de ella es lo que el pronóstico cree saber. En una carrera real "
                    "unos pocos nombres se llevan casi toda la probabilidad; sobre los datos "
                    "sintéticos de este panel la caída es mucho más suave, porque el generador "
                    "no tiene estrellas.",
    "cy_model_vs_baseline": "Cada punto es un ciclista. Sobre la diagonal, el modelo lo ve mejor "
                            "que el ranking; debajo, peor.",
    "cy_eval_chart": "Más abajo es mejor. Compara cada línea con la del ranking, no con cero.",
}

# The jargon, defined once. Rendered as an expander in the sidebar of every page.
#
# Each entry is one plain sentence plus a concrete example, because the example
# is what makes an abstraction stick. Ordered roughly by how early a reader
# meets the term, not alphabetically — a glossary you read top to bottom teaches
# more than one you only look things up in.
GLOSSARY = [
    ("Línea base",
     "Con qué se compara un pronóstico para saber si vale algo. En Baloto es el azar puro; en "
     "fútbol es la cuota de cierre; en ciclismo es el ranking previo. Sin línea base fijada de "
     "antemano, cualquier modelo parece bueno."),
    ("p-valor",
     "Qué tan raros serían tus datos **si no hubiera ningún efecto**. Bajo (<0.05) = raro, algo "
     "pasa. Alto = nada fuera de lo normal. Ejemplo: p = 0.40 significa «esto pasaría 4 de cada "
     "10 veces por pura casualidad», o sea nada llamativo. **No** es la probabilidad de que tu "
     "hipótesis sea cierta."),
    ("Intervalo de confianza (IC 95%)",
     "El rango dentro del cual está la respuesta de verdad, con la precisión que te dieron tus "
     "datos. Ejemplo: una ventaja de +0.02 [−0.15, +0.19] quiere decir que ni siquiera sabes el "
     "signo. El p-valor dice si descartas el azar; el intervalo dice **con cuánta precisión "
     "mediste**."),
    ("Corrección de Bonferroni",
     "Si pruebas 6 modelos a la vez, tienes 6 oportunidades de que alguno pase el filtro por "
     "suerte — con 6 pruebas eso pasa ~26% de las veces. La corrección baja el umbral (0.05 "
     "dividido entre 6) para compensar. **Siempre lee la columna corregida.**"),
    ("Walk-forward / fuera de muestra",
     "Entrenar el modelo solo con lo que se sabía antes del evento que intenta predecir, y "
     "avanzar. Es la única forma honesta de probar un pronóstico: si el modelo vio el resultado "
     "al entrenar, «acertarlo» no significa nada."),
    ("Potencia y efecto mínimo detectable (MDE)",
     "La ventaja más pequeña que tus datos alcanzarían a ver. Con 200 sorteos solo detectas "
     "ventajas grandes. Es lo que convierte un «no encontré nada» en una frase con contenido: "
     "«no hay ventaja mayor que tanto»."),
    ("Distribución hipergeométrica",
     "La fórmula exacta para «saqué 5 bolas de 43 sin reponer, ¿cuántas coinciden con las tuyas?». "
     "Da 0.58 aciertos de 5 en promedio. Es exacta, no una simulación — por eso es la referencia "
     "correcta del backtest."),
    ("Chi-cuadrado",
     "Una prueba que compara «cuántas veces salió cada cosa» contra «cuántas veces debería haber "
     "salido». Responde: ¿este desbalance es más de lo que produce el azar?"),
    ("Autocorrelación / Ljung-Box",
     "Si el resultado de hoy dice algo sobre el de mañana. Si no hay autocorrelación, no hay "
     "«memoria» y los modelos de series de tiempo (ARIMA, Prophet) no tienen nada que aprender."),
    ("Estadístico de orden",
     "Cuando las balotas se publican ordenadas de menor a mayor, la primera columna ya no es «una "
     "balota al azar»: es siempre el mínimo de las cinco. Analizarla por separado produce patrones "
     "falsos garantizados. La prueba agrupada es inmune."),
    ("Cuota decimal",
     "Cuánto te pagan por cada peso apostado, premio incluido. Cuota 2.50 = si apuestas 1.000 y "
     "aciertas, recibes 2.500. La probabilidad implícita es 1 dividido entre la cuota."),
    ("Margen / sobrerredondeo",
     "Las tres probabilidades implícitas de un partido suman más de 1 (típico 1.02-1.08). Ese "
     "exceso es la comisión de la casa. Hay que quitarlo antes de comparar un modelo contra el "
     "precio, o lo estás midiendo contra una barra torcida a favor de la casa."),
    ("Cuota de apertura vs. de cierre",
     "La apertura es la primera estimación de la casa. El cierre es el precio justo antes del "
     "partido, después de que todo el dinero se movió, y es muchísimo más afilado. Ganarle a la "
     "apertura no prueba casi nada."),
    ("Calibración",
     "Cuando dices «60%», ¿pasa el 60% de las veces? Un pronóstico calibrado acierta la "
     "frecuencia a largo plazo. Puede estar perfectamente calibrado y ser inútil (decir siempre "
     "«33%») — por eso calibración no basta."),
    ("RPS (ranked probability score)",
     "La nota de un pronóstico de fútbol. Más bajo es mejor. Cobra la **distancia** del error: "
     "apostar por el local cuando gana el visitante es más grave que haber apostado al empate, "
     "porque local-empate-visitante están ordenados."),
    ("Dixon-Coles",
     "El modelo de fútbol de este panel. Estima una fuerza de ataque y una de defensa por equipo, "
     "más la ventaja de local, y corrige la frecuencia de los marcadores bajos (0-0, 1-0, 1-1), "
     "que un Poisson simple subestima."),
    ("Elo",
     "Una sola nota de fuerza por equipo, que sube o baja después de cada partido según el "
     "resultado y contra quién. Viene del ajedrez. Es una línea base útil y fácil, pero solo "
     "produce local/empate/visitante, nunca un marcador."),
    ("Criterio de Kelly",
     "Cuánto apostar cuando crees tener ventaja, para maximizar el crecimiento a largo plazo sin "
     "arruinarte. Solo tiene sentido si la ventaja es **real**: aplicado sobre una ventaja "
     "imaginaria, Kelly acelera la quiebra en vez de evitarla."),
    ("Plackett-Luce",
     "La forma estándar de convertir «una fuerza por ciclista» en la probabilidad de un orden de "
     "llegada completo: el más fuerte gana con probabilidad proporcional a su fuerza, se lo quita "
     "de la lista, y se repite para el segundo puesto."),
]


def section(title, help_key):
    """A subheader with the ⓘ, plus the guided box when the key has one."""
    st.subheader(title, help=HELP[help_key])
    plain(help_key)


def plain(help_key):
    """The always-visible 'what am I looking at' box for `help_key`.

    Rendered by `section()` automatically; called directly only where a surface
    needs the box without opening a new subheader.
    """
    entry = PLAIN.get(help_key)
    if entry is None:
        return
    with st.container(border=True):
        st.markdown(f"**Qué estás viendo.** {entry['veo']}")
        st.markdown(f"**Qué puedes concluir.** {entry['concluyo']}")
        # Last and never optional: this is the line that stops a reader leaving
        # with the conclusion the surface merely looks like it supports.
        st.markdown(f"**Lo que NO significa.** {entry['ojo']}")


def chart(fig, title, help_key):
    """Render a Plotly figure under a titled line carrying its own ⓘ.

    The title moves out of the figure and into Streamlit so every chart in the
    dashboard gets the same typography and the same explain-on-hover affordance;
    Plotly's own title has nowhere to hang a help icon. A `READ` entry for the
    same key prints between the title and the figure — you want to know where to
    look before you look, not after.
    """
    st.markdown(f"**{title}**", help=HELP[help_key])
    if help_key in READ:
        st.caption(f"Cómo leerlo: {READ[help_key]}")
    # `title=None` leaves Plotly rendering the string "undefined"; an empty text
    # is what actually clears it.
    fig.update_layout(title={"text": ""})
    # Margins, legend placement and touch behaviour come from one place, so a
    # chart cannot be added that is unreadable on a phone: see dashboard/mobile.py.
    st.plotly_chart(mobile.responsive(fig), use_container_width=True,
                    config=mobile.PLOTLY_CONFIG)


def glossary():
    """The jargon expander. Every page puts one in the sidebar."""
    with st.sidebar.expander("Glosario — ¿qué significa esta palabra?"):
        for term, meaning in GLOSSARY:
            st.markdown(f"**{term}.** {meaning}")


def verdict_badge(looks_random, positive_text="Sin evidencia de patrón explotable", negative_text="Posible señal — revisar"):
    if looks_random:
        st.success(positive_text)
    else:
        st.warning(negative_text)


def plain_verdict(passed, headline, detail, good_is_pass=True):
    """A verdict stated as a sentence, with the technical detail underneath.

    A bare `p = 0.412` tells a reader who already knows the answer what they
    already knew and tells everyone else nothing. This states the finding in
    words first and keeps the number where it can still be checked.

    `good_is_pass` flips the colour without touching the wording, because
    "passed" is not always the welcome outcome: a lottery that fails the
    uniformity test is alarming, while a model that fails to beat chance is the
    expected result.
    """
    box = st.success if passed == good_is_pass else st.warning
    box(f"**{headline}**")
    st.caption(detail)



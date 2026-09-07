"""Shared UI vocabulary for every dashboard page.

The dashboard spans three domains now, and this module is what keeps them
looking and reading like one product: the explanatory copy, the two helpers that
render a section and a chart, and the verdict badge.

**All explanatory tooltip copy lives in the one `HELP` dict below.** That was
already this project's rule when the dashboard was Baloto-only, and splitting
the pages into modules is exactly when it would have been easiest to lose: three
files with their own inline strings cannot be reviewed as a whole, and the rule
this project actually cares about — no chart or table appears without saying what
it does *not* mean — is only checkable when the texts sit together.
"""

import streamlit as st

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

    # -- Forecast
    "tab_forecast": "Corre un modelo sobre tu histórico y pídele una sugerencia para el próximo sorteo. Es un "
                    "ejercicio de forecasting, no una predicción confiable.",
    "model_choice": "FrequencyBaseline juega el número más frecuente de cada posición (la referencia a "
                    "vencer). AutoARIMA/AutoETS/AutoTheta son modelos clásicos de series de tiempo. Prophet "
                    "es el más lento. XGBoost usa lags y medias móviles. Ninguno supera al azar en el "
                    "backtest — para eso está esa pestaña.",

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
    "include_prophet": "Prophet reajusta un modelo por posición y por ventana, así que multiplica el tiempo "
                       "de corrida. Déjalo apagado salvo que lo necesites.",
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
                             "agregado sin esfuerzo. La pregunta que importa es partido a partido, "
                             "y para eso hacen falta una regla de puntuación y un modelo.",

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
    "fb_form": "Últimos partidos de cada equipo antes de esta fecha. W/E/D desde la óptica del equipo.",
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


def section(title, help_key):
    """A subheader with the ⓘ that explains the section it opens."""
    st.subheader(title, help=HELP[help_key])


def chart(fig, title, help_key):
    """Render a Plotly figure under a titled line carrying its own ⓘ.

    The title moves out of the figure and into Streamlit so every chart in the
    dashboard gets the same typography and the same explain-on-hover affordance;
    Plotly's own title has nowhere to hang a help icon.
    """
    st.markdown(f"**{title}**", help=HELP[help_key])
    # `title=None` leaves Plotly rendering the string "undefined"; an empty text
    # is what actually clears it.
    fig.update_layout(title={"text": ""}, margin=dict(t=10, b=40))
    st.plotly_chart(fig, use_container_width=True)



def verdict_badge(looks_random, positive_text="Sin evidencia de patrón explotable", negative_text="Posible señal — revisar"):
    if looks_random:
        st.success(positive_text)
    else:
        st.warning(negative_text)


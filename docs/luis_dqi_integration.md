# Contrato del score con qbacktest

`luis_dqi` conserva el scorer DQI v2 y produce claves `date`, `symbol`, `composite_score`, más factores y metadata. La generación del score no necesita macro ni precios de ejecución. Short y rebalanceo no modifican el score.

El módulo `luis_dqi.backtest` conecta esa cinta al **motor externo autorizado**, no a un simulador alternativo. No se distribuyen Broker, Portfolio, Environment ni archivos de qbacktest en este repositorio. Se exige un checkout Git de qbacktest en `a9f0edbc1d0fdc3e318ca9e991027b4131fcf301`, sin modificaciones a `src/qbacktest`. Un motor ya importado desde otra ubicación se rechaza.

## Inputs y calendario

Los tres inputs son obligatorios y se reciben por rutas, sin archivos locales implícitos:

- Scores parquet: `date`, `symbol`, `composite_score`, sin claves duplicadas.
- Precios parquet: `date`, `symbol`, `close_price`, `volume_in_units`; no se fabrica volumen. Las claves pueden estar en el índice original.
- Macro CSV: `BATCH_DATE`, `FIRST_TRADABLE_DATE`, `STRESS_PERCENTILE_0_100`.

Una señal se ejecuta en la primera sesión observada estrictamente posterior a su fecha. El ranking usa precios finitos y positivos de la sesión exacta de corte, sin rellenar precios por ticker para decidir elegibilidad. Empates: símbolo ascendente. No se combinan señales anteriores al estudio en la primera barra.

La macro sólo se usa si `FIRST_TRADABLE_DATE <= execution_date` y `BATCH_DATE < execution_date`. El snapshot archivado **no certifica vintages históricos point-in-time**. En las variantes por eventos se mantienen las canastas del último score mensual; la covarianza y la distribución long/short sí pueden actualizarse a diario con información hasta el cierre anterior.

## Asignación y ejecución

Top N long y bottom N short, Markowitz-LedoitWolf por cada lado. Proxy de retorno: score menos su mínimo más `1e-6`, invirtiendo el score del lado short. Covarianza: últimas 253 filas de precios disponibles, mínimo 60 retornos, columnas con al menos 90% válido, forward-fill de precio limitado a 10 filas y retornos restantes nulos llevados a 0, más ridge `1e-6`. Se resuelve la matriz, se recortan pesos negativos y se redistribuye proporcionalmente respetando cap por nombre `3/N` dentro de cada lado. Los nombres sin covarianza reciben `1/N`; si no se puede estimar, fallback equal weight. N es el tamaño de la canasta candidata: pueden quedar pesos cero.

Es la reconstrucción local documentada de Markowitz v2, **no una afirmación de igualdad con el wrapper original de Clara en Windows**. Sus resultados Markowitz no habían coincidido; las referencias de la presentación fueron recalculadas homogéneamente con estos adaptadores. El scorer de Luis permanece intacto.

Para el punto 3 el short objetivo es `0.30 * clip((stress - 75) / 25, 0, 1)` y long `1 - short`. Target bruto 1 y neto `1 - 2*short`, antes de reservar costos. Ni el estrés original ni los criterios estrictos alteran esta fórmula. Las variantes mensuales también permiten LO, P50/P75/P90, otros tamaños y topes para repetir las comparaciones anteriores.

El broker nativo aplica comisiones 10 pb del nocional y préstamo corto 2% anual ACT/360, con lotes/ticks `1e-8`, caja sin interés y headroom técnico de leverage 1,5. Ese headroom permite rebalancear un bruto que derivó por precios; **no es leverage objetivo**. Se reserva caja por las comisiones previstas con un factor común y buffer numérico `1e-8 NAV`, sin alterar el NAV calculado por el motor.

Si falta una cotización de entrada, su proporción queda en caja: no hay reemplazo ni renormalización oportunista. Las posiciones existentes sin precio negociable se congelan para sizing; el motor mantiene el último precio y fuerza cierre tras 10 sesiones sin datos con recovery 1. El cierre forzado nativo omite comisiones; no se corrige retrospectivamente. No se modelan disponibilidad/locate de préstamo, dividendos, impuestos ni slippage real.

Sharpe: media de retornos diarios / desviación muestral × raíz de 252, RF 0, incluyendo el costo del primer día. CAGR: calendario ACT/365.25 desde la caja inicial fechada al último cierre previo al primer score; la variante de 252 sesiones queda separada.

Ver [políticas, comandos y controles de reproducción](rebalance.md).

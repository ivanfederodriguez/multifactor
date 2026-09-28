# Rebalanceos de Luis: ejecución reproducible

## Mandato fijo del punto 3

Markowitz, Top 30 long / bottom 30 short, P75, tope short 30%, comisión 10 pb, borrow 2% ACT/360 y capital inicial USD 1.000.000. Período completo: 2014-03-03 a 2025-12-31, 2.978 sesiones. Se conservan las 142 fechas del calendario mensual en todas las variantes. Una política por proceso; no usar los adaptadores globales en threads paralelos dentro del mismo intérprete.

`short_target = 0.30 * clip((stress - 75)/25, 0, 1)`.

Se endurecen **los disparadores de rebalanceo**, no el umbral ni el tope del short. Un rebalanceo extra restablece todo el target, no sólo el lado short. El short realmente mantenido puede diferir entre calendarios porque sus posiciones derivan hasta el próximo evento.

## Políticas disponibles

| `--policy` | Rebalanceo adicional al mensual |
| --- | --- |
| `monthly` | Ninguno. |
| `stress_original` | Entrada/salida del régimen `stress >= 75`, o cambio del short target de al menos 5 pp desde el último rebalanceo. |
| `stress_gap5` | Sólo cambio del short target de al menos 5 pp; sin gatillo por cruce P75. |
| `stress_gap10` | Sólo cambio del short target de al menos 10 pp. |
| `stress_gap15` | Sólo cambio del short target de al menos 15 pp. |
| `stress_gap20` | Sólo cambio del short target de al menos 20 pp. |
| `drift_original` | Desvío máximo por acción ≥5 pp **o** distancia global ≥10% NAV. |
| `drift_7p5_15` | Desvío por acción ≥7,5 pp **o** global ≥15% NAV. |
| `drift_10_20` | Desvío por acción ≥10 pp **o** global ≥20% NAV. |
| `drift_15_30` | Desvío por acción ≥15 pp **o** global ≥30% NAV. |

Las comparaciones incluyen igualdad al umbral, con tolerancia numérica `1e-12`. En estrés la referencia es **el último rebalanceo**, no el short target de ayer. En desvío se compara a diario el target causal con las tenencias reales marcadas al cierre anterior. Los pesos son firmados: short negativo. Se considera la unión de acciones actuales y objetivo; entradas/salidas también cuentan. Se excluye la caja. La distancia global es `0.5 * sum(abs(weight_actual - weight_target))`; el criterio es OR, no AND. No es una estimación de la comisión que el broker cobrará.

Ejemplo: de 10% a 9% en AAPL hay 1 pp de diferencia, insuficiente para `drift_original`; de 10% a 1% hay 9 pp y sí se dispara por el nombre. Las variantes estrictas elevan esos umbrales.

**No se regenera un nuevo ranking diario**: el scorer de esta experiencia es mensual. Las canastas quedan fijas hasta el nuevo score; los weights target diarios se recalculan con covarianza hasta el cierre anterior y macro disponible. Implementar un universo/score nuevo diario sería otro experimento, no la reproducción de estos resultados.

## Preparar el entorno

Usar Python 3.11 y un entorno autorizado que ya permita importar las dependencias del checkout privado de qbacktest. Este repositorio no concede acceso ni licencia del motor.

```bash
git clone https://github.com/ivanfederodriguez/multifactor.git
cd multifactor
python -m pip install -e ".[backtest]"
python -m unittest discover -s tests -v
```

No instalar ciegamente el `requirements.txt` histórico del motor sobre este entorno: pide `pandas<3`, mientras la reproducción congelada usó pandas 3.0.3 y el motor importado desde su `src`. Las versiones numéricas de esta reproducción son NumPy 2.4.6, pandas 3.0.3, pyarrow 24.0.0 y scikit-learn 1.9.0. El manifiesto registra además plataforma y Python; compartir el entorno autorizado o comparar numéricamente si difiere. No se promete igualdad binaria universal entre Windows/macOS.

Antes de empezar cada proceso, fijar `PYTHONHASHSEED=0`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` y `MKL_NUM_THREADS=1`, como en las corridas originales.

## Comandos para Clara en PowerShell

Reemplazar las cuatro rutas. Copiar los datos por el canal privado; no están en GitHub. El checkout del motor debe tener el commit requerido, sin cambios en `src/qbacktest`.

```powershell
$env:PYTHONHASHSEED = "0"
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$dqiArgs = @(
  "--qbacktest-root", "C:\ruta\al\qbacktest",
  "--scores-path", "C:\ruta\privada\dqi_v2.parquet",
  "--prices-path", "C:\ruta\privada\prices.parquet",
  "--macro-path", "C:\ruta\privada\macro_tail_score_operational.csv",
  "--reference"
)
python -m luis_dqi.backtest @dqiArgs --policy monthly --output-dir outputs\monthly
python -m luis_dqi.backtest @dqiArgs --policy stress_gap20 --output-dir outputs\stress_gap20
python -m luis_dqi.backtest @dqiArgs --policy drift_10_20 --output-dir outputs\drift_10_20
```

Para repetir todos los criterios, cada corrida completa en una carpeta nueva:

```powershell
$policies = @("monthly", "stress_original", "stress_gap5", "stress_gap10", "stress_gap15", "stress_gap20", "drift_original", "drift_7p5_15", "drift_10_20", "drift_15_30")
foreach ($policy in $policies) {
  python -m luis_dqi.backtest @dqiArgs --policy $policy --output-dir "outputs\grid_$policy"
  if ($LASTEXITCODE -ne 0) { throw "Falló $policy; revisar status.json" }
}
```

En macOS/Linux, usar los mismos argumentos y rutas locales; `python -m luis_dqi.backtest --help` enumera todos. El entrypoint instalado `luis-dqi-backtest` es equivalente. Si se necesita un corte de ejecución, pasar `--deadline-utc` con zona horaria y fecha futura. No hay fecha vencida por defecto; un corte deja estado `aborted`, nunca `completed`.

## Controles y resultados

`--reference` rechaza inputs diferentes, un período recortado, costos/caja diferentes y corridas que no terminan las 2.978 sesiones o no conservan los 142 eventos mensuales. También se rechazan errores de estrategia y órdenes rechazadas. Sin ese flag se pueden estudiar otros datos; los resultados no son la referencia congelada.

Hashes SHA-256 de los inputs congelados:

- Scores: `1837df55bed8eb50b05399a12e81a0a9c7a98308d66d5cd46b259a4722720996`.
- Precios: `4bd8a70cd04ae5e6a984fee0a7f525474222071b0172b92ec256adf622308854`.
- Macro: `7a8d2f2e0c79b8dfcbd4e2626630b8a0c8cfd6e92a536dedda821fb986bb5e42`.
- Motor: commit `a9f0edbc1d0fdc3e318ca9e991027b4131fcf301`.

La carpeta de cada run conserva NAV/retornos diarios, posiciones, trades, target weights, ejecución, eventos, `rebalance_audit.csv`, diagnósticos Markowitz, código del adaptador ejecutado y manifiestos con hashes/costos/versiones. Distinguir eventos de rebalanceo de órdenes por acción. Los resultados deben quedar bajo `outputs/` o fuera del repo: están excluidos de Git.

Verificar desde NAV/trades diarios y, opcionalmente, comparar con una carpeta de referencia completa recibida por el canal privado:

```bash
python -m luis_dqi.backtest.verify --run outputs/monthly
python -m luis_dqi.backtest.verify --run outputs/monthly --against /ruta/privada/referencia_monthly
```

`--against` exige igualdad numérica exacta de las series completas de NAV, trades, weights, target weights y auditoría. El verificador no ejecuta un nuevo backtest ni modifica archivos. No valida por sí solo vintages macro ni la corrección económica de la base.

La verificación también lee los inputs y archivos del motor en las rutas guardadas en el manifiesto. Ejecutarla en la máquina de la corrida, con esas rutas disponibles; una carpeta de referencia copiada de otra máquina puede usarse en `--against` sin trasladar sus rutas de datos.

`luis_dqi/backtest/provenance.json` identifica los tres adaptadores locales originales y los fingerprints AST de 28 definiciones económicas intactas. La función de corrida sólo cambia su bootstrap de modelo/ruta de scores; no se copió el motor ni se cambió contabilidad, sizing, covarianza o reglas de ejecución.

Las pruebas unitarias del motor se habilitan con `LUIS_QBACKTEST_ROOT`. La igualdad de fingerprints prueba conservación de lógica; **no sustituye ejecutar el período completo y comparar NAV y trades**. La reproducción desde otra computadora debe verificar inputs/entorno, errores/rechazos, calendario y series completas, no sólo el Sharpe redondeado.

Validación del empaquetado: [registro del 2026-09-28](validation_rebalance_20260928.json). Pasaron 29 pruebas tanto desde el código como desde el wheel instalado en un venv nuevo, compartiendo las bibliotecas numéricas del entorno autorizado. Se volvieron a ejecutar completos `monthly`, `stress_gap20` y `drift_10_20` en esta máquina; en los tres coincidieron exactamente las seis series comparadas con sus referencias de la presentación. No se afirma que las otras siete políticas se hayan vuelto a simular en esta revisión: sus corridas completas previas se verificaron por lectura y sus reglas originales se conservaron por fingerprint. Falta que Clara confirme la reproducción en su computadora.

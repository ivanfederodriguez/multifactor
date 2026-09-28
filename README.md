# Multifactor: shared Luis DQI v2 model

Este repositorio contiene únicamente el **modelo de Luis para Clara e Iván**: generador de scores, asignación Markowitz y adaptadores de rebalanceo para un motor qbacktest autorizado y separado. No contiene el dashboard, otros modelos, datos privados ni el código del motor.

## Instalar el scorer

Usar Python 3.11. Las versiones de NumPy, pandas y pyarrow se fijan para facilitar la reproducción de la cinta.

```bash
git clone https://github.com/ivanfederodriguez/multifactor.git
cd multifactor
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

## Generar la misma cinta desde Base B

Los datos reales se intercambian por un canal autorizado; **no están publicados en GitHub**. Copiar la Base B corregida a una carpeta local, por ejemplo `private_data/scores_primary_24.parquet`:

```bash
python -m luis_dqi score \
  --source private_data/scores_primary_24.parquet \
  --output outputs/dqi_v2.parquet

python -m luis_dqi verify \
  --scores outputs/dqi_v2.parquet \
  --source private_data/scores_primary_24.parquet \
  --base-b-reference
```

El comando instalado `luis-dqi` es equivalente a `python -m luis_dqi`. No reemplaza un resultado existente sin `--overwrite`. Escribe parquet más `dqi_v2.manifest.json`, con hashes de entrada/salida, código, entorno, cobertura y la transformación de columnas. Ambos deben usar el **mismo commit, la misma entrada y el mismo entorno**; comparar el hash de salida, no sólo el nombre del archivo.

Referencia Base B congelada:

- Entrada SHA-256: `f95ed5f6750e39564bc7c5b186cb68e4a1c249232ecceb0982c4010ef27b61b0`.
- Salida DQI SHA-256: `1837df55bed8eb50b05399a12e81a0a9c7a98308d66d5cd46b259a4722720996`.
- 197.224 filas, 194 fechas de score; 196.304 filas puntuadas.

`--base-b-reference` sólo pasa para esa cinta congelada; no es una validación general de otros datos. Si se cambia NumPy/pandas/pyarrow o arquitectura, un hash parquet distinto requiere comparar claves y valores numéricos antes de concluir que el modelo cambió. No se afirma reproducibilidad binaria universal entre máquinas.

## Qué conserva del modelo de Luis

El pipeline es extracción → inversión de señales donde menor es mejor → winsorización 1/99 → z-score por subindustria/fallback de universo → promedio ponderado por factor → composite con cobertura mínima/cap de renormalización → percentil y rating. Se mantienen seis factores, sus pesos fijos y todas las reglas originales. R&D no existe en Base B: queda nulo y Growth usa la renormalización original, no cero ni una señal inventada.

Los archivos de cálculo y el adaptador coinciden con los previamente utilizados, salvo **dos imports relativos** para convertirlos en paquete. Una prueba reconstruye los bytes originales y verifica sus SHA-256. Las validaciones nuevas del comando rechazan entradas vacías, claves nulas/duplicadas y sobrescrituras accidentales; no alteran los cálculos sobre entradas válidas.

## Backtests y rebalanceo compartidos

El score no cambia al agregar shorts o rebalanceos. El nuevo comando `python -m luis_dqi.backtest` consume la cinta y ejecuta las estrategias públicas sobre **qbacktest sin modificarlo**. El checkout autorizado del motor se pasa con `--qbacktest-root`; se exige el commit `a9f0edbc1d0fdc3e318ca9e991027b4131fcf301` y código del motor sin cambios.

Instalar en el entorno autorizado que ya tiene las dependencias de qbacktest:

```bash
python -m pip install -e ".[backtest]"
python -m luis_dqi.backtest --help
```

Se incluyen las políticas `monthly`, `stress_original`, `drift_original` y los siete umbrales estrictos usados en la presentación. **Las políticas de eventos mantienen Top 30, P75 y tope short 30%; sólo cambia cuándo rebalancear.** Los datos y el motor se suministran por un canal autorizado. No alcanza con clonar este repo para tenerlos.

Ver [cómo reproducir y comparar las políticas](docs/rebalance.md) y [el contrato con qbacktest](docs/luis_dqi_integration.md). Las rutas son argumentos explícitos, funcionan también en Windows y los resultados existentes nunca se sobrescriben.

Hay una plantilla de GitHub Actions en `ci/luis-score.workflow.yml`. No está activa: la credencial utilizada para publicar no permite crear workflows. Para habilitarla, un administrador con los permisos correspondientes puede copiarla a `.github/workflows/luis-score.yml`. Las pruebas locales se ejecutan con el comando de instalación anterior. Para incluir las pruebas unitarias del adaptador nativo, definir `LUIS_QBACKTEST_ROOT` con la ruta al motor autorizado; sin ese motor esas pruebas se omiten y las de lógica, procedencia y scorer siguen disponibles.

## Datos y confidencialidad

Este repositorio es público: nunca agregar entradas Bloomberg/Base B, precios privados, credenciales ni el motor propietario.

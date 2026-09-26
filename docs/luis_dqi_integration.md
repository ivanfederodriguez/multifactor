# Contrato del score con qbacktest

La señal de Luis es independiente del motor de ejecución. `luis_dqi` conserva el scorer DQI v2 y produce una tabla con claves `date`, `symbol` y `composite_score`, más factores, subfactores y metadata. La generación no necesita macro ni precios de ejecución y no cambia pesos en función del estrés.

## Uso con el paquete de Clara

Generar y verificar `outputs/dqi_v2.parquet` siguiendo el README. En el runner compartido del paquete `DQI_qbacktest_Clara_20260924`, pasar su ruta absoluta:

```bash
python analysis_dqi_v2_20260921/build/qbacktest_base_b_20260922/run_case.py \
  --model dqi --scenario p50 \
  --scores-path /ruta/a/multifactor/outputs/dqi_v2.parquet \
  --output-dir /ruta/nueva/resultado_p50 \
  --deadline-utc 2026-10-01T23:00:00+00:00
```

El ejemplo se ejecuta **desde la carpeta del paquete de Clara**, que proporciona el registro mínimo de modelo, precios, macro y enlace a un motor autorizado. Reemplazar fecha de corte por una futura; no confundir este comando con una ejecución autocontenida desde este repo.

Referencia del motor autorizado: commit `a9f0edbc1d0fdc3e318ca9e991027b4131fcf301`. Runner auditado SHA-256: `3932cdb53f00e13c684d502cfed2472201905c59b989bbbf0b2cf4dbd6f0ef75`. No usar simplemente cualquier versión instalada de qbacktest.

## Política externa del backtest

Primera sesión estrictamente posterior al score, top20 long/bottom20 short, equal weight dentro de cada lado, target bruto 1. La parte corta objetivo es `0.20 * clip((stress - P)/(100 - P), 0, 1)` para P=50/75/90; LO usa cero. Se mantiene el scorer fijo: **agregar short no cambia el composite DQI**.

El broker nativo aplica 10 pb de comisión y préstamo corto 2% anual ACT/360, con headroom técnico de leverage 1,5, no leverage objetivo. La macro y los precios del paquete son inputs separados, no datos publicados aquí. El snapshot macro no certifica vintages históricos y no se modelan disponibilidad de préstamo, dividendos, impuestos ni slippage real.

La referencia completa P50, previamente auditada, tiene Sharpe `0.9319991858474546`, CAGR `0.18307401457275918`, drawdown `-0.29696088431976764` y NAV final `7317270.965091507` en 2.978 sesiones. Son referencias de una corrida previa, no una promesa de resultado con datos o motor distintos. Esta integración valida generación del score; no realiza una nueva matriz completa de backtests.

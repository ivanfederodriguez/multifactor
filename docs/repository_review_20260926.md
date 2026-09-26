# Revisión del repositorio — 26/09/2026

Base inspeccionada: `67ab160700c3ef4de376ac97cd6b40600c6e809f`, rama `main`, repositorio público `ivanfederodriguez/multifactor`.

## Hallazgo y decisión

El árbol contiene 1.817 archivos rastreados: `app.py`, `charts.py`, `data_loader.py`, instrucciones de despliegue, dependencias y datos procesados de experimentos. El catálogo ejecutado carga 362 experimentos en ocho familias: OnlineEM (96), EM_Fijo_Reentrenado (64), HRPRolling (48), KalmanEM_base (48), KalmanEM_SB (34), EM_Fijo (24), HRP_Fijo (24) y SB_Fijo (24).

Es útil para navegar parámetros, comparar curvas NAV y visualizar pesos de factores/subfactores. **Se conserva íntegramente**. La condición solicitada de borrar todo si no había nada útil no se cumple.

No hay código de entrenamiento ni generador de score en este árbol: los YAML y resultados son entradas del visor, no un motor reproducible. Las dependencias tenían rangos amplios y no había pruebas ni pipeline CI del scorer. Tampoco se considera a esas métricas una reproducción de DQI Base B con qbacktest.

## Integración añadida

- Paquete `luis_dqi`: config y scorer de Luis v2 más adaptador de columnas PROMETHEUS/Base B.
- CLI único para generar y verificar scores; entrada/salida explícitas, registro de hashes y entorno.
- Dependencias de cálculo fijadas, pruebas sintéticas y comprobación de identidad del código original.
- Plantilla de CI en `ci/luis-score.workflow.yml`. No se activa en Actions porque la credencial de publicación no tiene scope `workflow`.
- Documentación del contrato con el runner externo de qbacktest; no se redistribuye su implementación.
- Exclusión de datos reales y artefactos nuevos en `.gitignore`. Esto no elimina ni reescribe datos ya publicados en el historial anterior.

No se borró ningún archivo previo ni se reescribió historial. Esta revisión no es una auditoría exhaustiva de licencias/confidencialidad de las 362 carpetas ya existentes ni una nueva auditoría económica de sus resultados.

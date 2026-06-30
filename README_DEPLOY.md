# Dashboard de Experimentos Multifactor (Deploy)

Este directorio auto-contenido fue generado para realizar un deploy rápido y gratuito.
Contiene únicamente el código del dashboard y los datos filtrados esenciales de los experimentos.

**Tamaño de datos optimizado:** 222.20 MB (vs ~1.7 GB originales).
**Experimentos incluidos:** 318

## ¿Cómo deployar de forma gratuita?

### Opción 1: Streamlit Community Cloud (Recomendado)
Streamlit ofrece hosting gratuito para dashboards directamente desde GitHub.

1. **Crear un repositorio en GitHub**:
   Crea un repositorio de GitHub nuevo (puede ser público o privado).
2. **Subir este contenido**:
   Inicializa git en esta carpeta `deploy_dist` y súbelo a tu repositorio:
   ```bash
   cd deploy_dist
   git init
   git add .
   git commit -m "Initial commit for deploy"
   git branch -M main
   git remote add origin <URL-DE-TU-REPOSITORIO>
   git push -u origin main
   ```
3. **Deployar en Streamlit**:
   - Entra en [Streamlit Share](https://share.streamlit.io/).
   - Inicia sesión con tu cuenta de GitHub.
   - Haz clic en **"New app"**.
   - Selecciona tu repositorio, la rama (`main`) y escribe `app.py` en "Main file path".
   - Haz clic en **"Deploy!"**. ¡Y listo! Tu app estará online gratis.

### Opción 2: Hugging Face Spaces (Gratis)
Hugging Face te permite hostear apps de Streamlit de forma gratuita con un click.

1. Entra a [Hugging Face Spaces](https://huggingface.co/spaces) y haz clic en **"Create new Space"**.
2. Selecciona **Streamlit** como el SDK.
3. Elige la licencia y si quieres que sea público o privado.
4. Sube todos los archivos de esta carpeta (`deploy_dist`) usando git o la interfaz web de Hugging Face.

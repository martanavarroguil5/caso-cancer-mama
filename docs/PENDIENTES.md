# Pendientes antes de terminar el proyecto

## Importante

- [ ] **Quitar el acceso por Tailscale a la web.** Desde el 27/10/2026 la web escucha
  en `100.120.212.49:8501` solo para verla desde el Mac durante el desarrollo.
  Antes de la entrega:
  1. En `.streamlit/config.toml`, volver a `address = "localhost"` y borrar el
     comentario TEMPORAL.
  2. Borrar la nota «Temporal, quitar antes de la entrega» de la sección Web del
     `README.md`.
  3. Borrar esta entrada (y este archivo si queda vacío).
  4. Reiniciar la web y comprobar que la IP de Tailscale ya no responde.

## Por decidir

- [ ] Umbral de los paquetes (Youden o máximo macro-F1).
- [ ] Evaluar el test público una sola vez, con confirmación expresa.
- [ ] Despliegue en Hugging Face (necesita autorización de push).

# 🔐 Security Policy

## 📦 Supported Versions

| Versión | ¿Recibe actualizaciones de seguridad? |
| ------- | -------------------------------------- |
| 3.5.x   | ✅                                     |
| 3.x     | ❌                                     |
| < 3.0   | ❌                                     |

## 📣 Reporte de vulnerabilidades

Si descubres una vulnerabilidad en este proyecto, por favor **no la ignores**. En su lugar:

1. Abre un issue **privado** o contacta directamente al mantenedor del repositorio.
2. Describe claramente:
   - Qué comportamiento has observado.
   - Cómo reproducirlo (si es posible).
   - Qué impacto potencial tiene.

🔁 El equipo intentará responder en un **plazo máximo de 72 horas**.

Si el problema es crítico o afecta a la confidencialidad, se priorizará su resolución y se te mantendrá informado/a durante el proceso.

---

## 🔒 Archivos prohibidos

Para proteger el entorno de ejecución y la privacidad de datos:

- **Está terminantemente prohibido subir archivos sensibles**, incluyendo pero no limitado a:
  - `.env`
  - `.env.*`
  - `.key`, `.pem`, `.crt`
  - `credentials.json`, `secrets.yml`, etc.
  - Cualquier archivo de configuración que contenga contraseñas o tokens

✅ El `.gitignore` de este proyecto ya filtra estos archivos por defecto.

En caso de detectar un commit que haya subido alguno de estos por error:

- Se procederá a su **eliminación inmediata** mediante `git filter-repo` o herramientas similares.
- Se notificará al autor para evitar futuras filtraciones.

---

## 🧠 Buenas prácticas recomendadas

- Utiliza variables de entorno locales y `dotenv` solo para desarrollo.
- Usa secretos cifrados en producción (por ejemplo, Docker Secrets o GitHub Actions Encrypted Secrets).
- Nunca hagas hardcode de claves en tu código fuente.

---

> 🛡️ Esta política puede ser actualizada con el tiempo para reforzar la seguridad del repositorio.

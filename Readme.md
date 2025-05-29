<h1 align="center">🧠 AI-EMG TFG — Clasificación de Espasticidad con Modelos de IA</h1>

<p align="center">
  <img src="https://img.shields.io/badge/TFG-Ingeniería%20Informática-blue" />
  <img src="https://img.shields.io/badge/Universidad-Antonio%20de%20Nebrija-brightgreen" />
  <img src="https://img.shields.io/badge/Estado-En%20fase%20final-success" />
</p>

---

## 🎯 Objetivo del Proyecto

Este Trabajo de Fin de Grado desarrolla un sistema completo de clasificación de espasticidad neurológica mediante señales EMG, empleando técnicas avanzadas de procesamiento de señales y modelos de inteligencia artificial. El sistema se valida sobre entornos simulados, implementaciones Dockerizadas y código real ejecutable.

---

## 📂 Estructura del Repositorio

```
ModelV1/
ModelV2CNN/
ModelV3Fourier/
ModelV3.5Fourier/
```

Cada carpeta contiene un modelo entrenado independiente con arquitectura distinta y pipeline personalizado.

---

## 🧪 Modelos Desarrollados

| Versión      | Tipo de Modelo         | Descripción                                                                 |
|--------------|------------------------|-----------------------------------------------------------------------------|
| **ModelV1**  | 🔸 Perceptrón multicapa (MLP) | Primer prototipo funcional. Clasificador de EMG con arquitectura densa y básica. |
| **ModelV2CNN** | 🔹 CNN básica             | Primer uso de convoluciones sobre señales EMG preprocesadas. Mejora de precisión. |
| **ModelV3Fourier** | 🔸 CNN + Fourier         | Señales EMG transformadas al dominio frecuencial. CNN con mejor generalización. |
| **ModelV3.5Fourier** | 🔹 CNN + Fourier + optimización GPU | Versión optimizada y preparada para despliegue productivo y ejecución acelerada. |

---

## 🚀 Cómo ejecutar cada modelo


---

## 🐳 Ejecución rápida con DockerHub

Puedes ejecutar los modelos directamente desde las imágenes publicadas en DockerHub sin clonar el repositorio.

### 🔹 ModelV2CNN - CNN básica

```bash
docker run -it --rm -p 8501:8501 josojmf/modelv2cnn-app:latest

```

📦 [Ver en DockerHub](https://hub.docker.com/repository/docker/josojmf/modelv2cnn-app)

---

### 🔸 ModelV3Fourier - CNN + Transformada de Fourier

```bash
docker run -it --rm -p 8501:8501 josojmf/modelv3fourier-app:latest

```

📦 [Ver en DockerHub](https://hub.docker.com/repository/docker/josojmf/modelv3fourier-app)

---

### 🔹 ModelV3.5Fourier - Versión optimizada GPU-ready

```bash
docker run -it --rm -p 8501:8501 josojmf/modelv3.5fourier:23-03-2025

```

📦 [Ver en DockerHub](https://hub.docker.com/repository/docker/josojmf/modelv3.5fourier)

---

> 🧠 Las imágenes incluyen todas las dependencias necesarias, entornos configurados y punto de entrada predefinido.

> ⚠️ Asegúrate de tener `Python 3.9+` o usar los contenedores `Docker` proporcionados.

---
## Ejecución local

### 🔸 1. ModelV1 - Perceptrón multicapa (MLP)

```bash
cd ModelV1/Python_Ai_Signal_Processing
pip install -r requirements.txt
python main.py
```

---

### 🔹 2. ModelV2CNN - CNN básica

```bash
cd ModelV2CNN
pip install -r requirements.txt
python ui.py
```

---

### 🔸 3. ModelV3Fourier - CNN + Transformada de Fourier

```bash
cd ModelV3Fourier
pip install -r requirements.txt
python main.py
```

---

### 🔹 4. ModelV3.5Fourier - Optimizado y GPU-ready

#### 🐳 Opción A: Docker (recomendado)

```bash
cd ModelV3.5Fourier
docker compose up --build
```

#### 🐍 Opción B: Python local

```bash
cd ModelV3.5Fourier
pip install -r requirements.txt
python mainGPU.py
```






## 📊 Resultados esperados

- Precisión final: **87.3%** con ModelV3.5Fourier
- Dataset simulado y normalizado basado en estudios clínicos
- Comparativas entre arquitecturas y filtros de señal

---

## 📦 Entorno y herramientas

| Herramienta | Descripción                             |
|-------------|-----------------------------------------|
| 🐍 Python   | Preprocesamiento y modelos              |
| 🧠 Keras/TensorFlow | Entrenamiento y ejecución de redes   |
| 🐳 Docker   | Contenerización de versiones finales     |
| 🧪 Streamlit | Interfaz visual de clasificación (V3.5) |

---

## 👨‍🎓 Autor

- **José-María Fernández Gómez**
- Grado en Ingeniería Informática - Universidad Nebrija
- 🏛️ Curso 2024–2025

---

## 💬 Contacto

¿Dudas o sugerencias? Puedes abrir un [Issue](https://github.com/Josojmf/AI_EMG_TFG/issues) o escribir directamente al autor.

---

<h3 align="center">Gracias por leer 🙏 ¡Espero que te guste este TFG tanto como a mí desarrollarlo!</h3>

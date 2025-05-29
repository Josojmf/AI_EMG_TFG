# 📑 Modelo V3.5 – Evaluación de Espasticidad con IA

**Objetivo:**  
Aplicación de **inteligencia artificial** para mejorar la evaluación y tratamiento de pacientes con **espasticidad**, derivada de alteraciones neuromusculares tras un **daño cerebral adquirido (DCA).**

---

## ⚙️ Requisitos Previos

1. **Crear un entorno virtual** (se recomienda usar [Anaconda](https://www.anaconda.com/download)).
2. **Clonar la rama del repositorio** correspondiente al modelo que se desea utilizar:

```bash
git clone --branch modelV3 https://github.com/ARIES-UNNE/TFG_spasm_JMFG.git
cd TFG_spasm_JMFG
```

---

## 🚀 Opciones de Ejecución

### ▶️ Opción 1: Usar Docker (recomendado)

> Asegúrate de tener [Docker](https://www.docker.com/) y [Docker Compose](https://docs.docker.com/compose/) instalados.

```bash
docker-compose up --build
```

Esto levantará automáticamente la API del modelo, el frontend y la base de datos en contenedores.

---

### ▶️ Opción 2: Ejecutar Localmente desde Terminal

#### 1. Instalar dependencias

Con `pip`:

```bash
pip install -r requirements.txt
```

O con `conda`:

```bash
conda install --file requirements.txt
```

#### 2. Ejecutar la aplicación

```bash
streamlit run ui.py
```

---

## 📂 Estructura del Proyecto (resumen)

```
TFG_spasm_JMFG/
├── app/                  # Lógica principal de la aplicación y conexiones
├── model/                # Modelos entrenados y scripts relacionados
├── data/                 # Datos de entrada / ejemplo
├── ui.py                 # Interfaz gráfica con Streamlit
├── requirements.txt      # Dependencias del proyecto
├── docker-compose.yml    # Configuración de contenedores
└── README.md
```

---

## 📌 Notas

- Este proyecto utiliza **Streamlit** como frontend para visualización y prueba del modelo.
- El modelo de IA está entrenado con datos de señales musculares .

---

## 🧠 Créditos

Desarrollado por ****José María Fernández Gómez (2025)****
Trabajo Final de Grado – Ingeniería Informática  


# LLM From Scratch

Este repositorio utiliza el corpus [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
para entrenar tres LLMs implementados totalmente desde cero:

- Un LLM basado en la arquitectura de GPT-2
- Un LLM que modifica el anterior añadiendo RMSNorm 
- Un tercer LLM añadiendo RoPE

## Instalación

El repositorio utiliza [uv](https://docs.astral.sh/uv/) para simplificar la
gestión de dependencias y entornos virtuales de Python.

Una vez instalado uv, las dependencias del proyecto su pueden instalar
utilizando el siguiente comando:

```bash
$ uv sync --extra eda
```

## Estructura

El código fuente se encuentra en la carpeta `src`, dividido en distintos
proyectos:

### src/eda

Contiene el Jupyter notebook donde se realiza el análisis exploratorio del
corpus [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).

### src/llm_from_scratch

Contiene el código fuente del LLM.

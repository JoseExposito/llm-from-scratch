# LLM From Scratch

Este repositorio utiliza el corpus [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
para entrenar tres LLMs implementados totalmente desde cero:

- Un LLM basado en la arquitectura de GPT-2
- Un LLM que modifica el anterior añadiendo RMSNorm 
- Un tercer LLM añadiendo RoPE

El código fuente está basado en la implementación llevada a cabo en el libro
"Build a Large Language Model (From Scratch)" de Sebastian Raschka.

## Instalación

El repositorio utiliza [uv](https://docs.astral.sh/uv/) para simplificar la
gestión de dependencias y entornos virtuales de Python.

Una vez instalado uv, las dependencias del proyecto su pueden instalar
utilizando el siguiente comando:

```bash
$ uv sync --extra notebooks
```

## Estructura

El código fuente se encuentra en la carpeta `src`, dividido en distintos
proyectos:

### src/00_eda

Contiene el Jupyter notebook donde se realiza el análisis exploratorio del
corpus [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).

### src/01_dataset_preparation

Contiene el Jupyter notebook donde se muestra como preparar el corpus para ser
utilizado por el LLM. Utiliza [byte-pair encoding](https://en.wikipedia.org/wiki/Byte-pair_encoding)
para tokenizar el texto, las clases [Dataset y DataLoader de PyTorch](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
para proveer los datos al LLM utilizando [sliding window](https://arxiv.org/abs/2502.18845)
y, finalmente, muestra como obtener token embeddings a partir de los token IDs.

### src/02_attention_mechanism

Contiene el Jupyter notebook donde se lleva a cabo la implementación del
mecanismo de atención del LLM. Se parte de una explicación del mecanismo de
atención y se mejora hasta llegar a la implementación un mecanismo de atención
similar al utilizado por GPT-2.

### src/llm_from_scratch

Contiene el código fuente del LLM.

# Bootcamp de Matemáticas para la Maestría en Ciencia de Datos y Políticas Públicas

## Contenido
* [Introducción a optimización](slides/intro_optimizacion.pdf).
    - Maximización o minimización.
    - Formulación matemática.
    - Restricciones y regiones factibles.
    - Soluciones locales y globales.
* [Algoritmo de Descenso de Gradiente](https://milocortes.github.io/bootcamp_matematicas/laboratorios/descenso_gradiente/index.html).
    - Derivadas.
    - Derivadas en múltiples dimensiones.
    - Gradientes.
    - Implementación del algoritmo de Descenso de Gradiente.
* [Aplicación Algoritmos de Optimización de Primer Orden](https://milocortes.github.io/bootcamp_matematicas/laboratorios/aplicaciones_primer_orden/index.html)

## Descarga del repositorio

Para descargar el repositorio utiliza la instrucción:

```
git clone https://github.com/milocortes/bootcamp_matematicas.git
```

## Sincronización del ambiente virtual

Sincronizamos las dependencias en nuestro ambiente virtual con la instrucción:

```bash 
uv sync
```
>Sincronizar (Syncing) es el proceso de instalar las versiones correctas de las dependencias de un lockfile en el ambiente del proyecto.

#### Inicio del servicio de Marimo

Para iniciar Marimo, ejecutamos :

```bash 
uv run marimo edit
```

#### Inicio del servicio de Jupyter Notebook

Para iniciar Jupyter Notebook, ejecutamos :

```bash 
uv run --with jupyter jupyter notebook
```

## Recursos adicionales

* [Algorithms - The Secret Rules of Modern Living - BBC documentary](https://www.youtube.com/watch?v=k2AqGongii0)
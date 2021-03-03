# CAP-PLAB2021

Proyecto de Laboratorio de la asignatura Computación de Altas Prestaciones (CAP) del curso 2020/2021.

## Table of Contents

- [General info](#general-info)
- [Technologies](#technologies)
- [Setup](#setup)
- [Commands](#commands)
- [Status](#status)
- [Future work](#future-work)

## General info

Este proyecto es un estudio sobre la paralelización de un [perceptrón multicapa](https://es.wikipedia.org/wiki/Perceptr%C3%B3n_multicapa) usando diversas técnicas, siendo estas la paralelización mediante threads con OpenMP, el uso de memoria compartida distribuida usando MPI y la aceleración mediante GPU usando CUDA.

## Technologies

- C
- CMake
- SLURM
- OpenMP
- MPI
- CUDA

## Setup

Para adquirir el código basta con clonar el repositorio con el siguiente comando:

```shell
$ git clone git@github.com:JoanRosell/CAP-PLAB2021.git
```

## Commands

### Compilar

```shell
$ ./compile.sh
```

Este script compila el código fuente y genera el ejecutable en un directorio llamado _build_ que está excluido de Git. Esto evita problemas a la hora de hacer los merges y pushes ya que sólo se comparte el código fuente.

### Lanzar sbatch manualmente

```shell
$ sbatch -o myFile.out -e myFile.err job.sh
```

El script _job.sh_ prepara el entorno, compila y ejecuta el proyecto. Este script se lanza al cluster SLURM mediante sbatch especificando dos archivos a los que se redirigen stdout y stderr.

### Lanzar sbatch mediante script

```shell
$ ./submit.sh myFile [epochs, numIn, numHid, numOut]
```
Este script lanza un sbatch con el trabajo _job.sh_ y configura los nombres de los archivos de salida/errores según el primer argumento que recibe. **El resto de argumentos son opcionales** y controlan la ejecución del proyecto. Si no se provee ningún argumento opcional el proyecto se ejecuta con los parámetros por defectos definidos en _commons.h_.

## Status

 - [ ] Preparación del entorno de trabajo (git, scripts, etc.)
 - [ ] Análisis funcional
 - [ ] Análisis computacional (complejidad espacial y temporal)
 - [ ] Paralelización con OpenMP

## Future work

No se prevé trabajo futuro.

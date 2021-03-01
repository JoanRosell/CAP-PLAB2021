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

## Technologies

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

### Lanzar sbatch

```shell
$ sbatch -o myFile.out -e myFile.err job.sh
```

El script 'job.sh' prepara el entorno, compila y ejecuta el proyecto. Este script se lanza al cluster SLURM mediante sbatch especificando dos archivos a los que se redirigen stdout y stderr.

## Status

WIP

## Future work

WIP

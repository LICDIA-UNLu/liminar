# Liminar

Liminar es una familia de modelos de lenguaje abiertos del sistema universitario argentino. Están adaptados a su
normativa, sus instituciones y su vida académica, y al español formal de la Argentina, y se publican con licencia
abierta para que cualquier persona o institución pueda usarlos. La desarrolla el Laboratorio de Investigación en Ciencia
de Datos & Inteligencia Artificial (LICDIA) de la Universidad Nacional de Luján. El nombre remite al Manifiesto Liminar
de la Reforma Universitaria de 1918.

## Dos capas

La **capa nacional** son los modelos publicados: cada uno parte de un modelo abierto y se ajusta con la normativa
universitaria nacional curada y con textos públicos sobre la vida universitaria.

La **capa de cada universidad** es un adaptador LoRA que cada institución entrena con sus propios actos sobre un modelo
Liminar, siguiendo la [receta](receta.md). Esa capa queda en la universidad.

## Contenido

- [Modelos](modelos.md): la familia, las versiones y las licencias.
- [Uso](uso.md): cómo cargar un modelo y cómo usarlo con recuperación de documentos.
- [Receta para universidades](receta.md): cómo entrenar la capa propia.
- [Corpus](corpus.md), [datos de entrenamiento](datos.md), [entrenamiento](entrenamiento.md),
  [evaluación](evaluacion.md) y [cómputo](computo.md): cómo se hizo.

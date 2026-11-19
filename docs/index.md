# Liminar

<div class="liminar-portada" markdown>
![Liminar](assets/liminar-logotipo-claro.png#only-light)
![Liminar](assets/liminar-logotipo-oscuro.png#only-dark)

<div class="lema">Modelos de lenguaje abiertos del sistema universitario argentino</div>
</div>

Liminar es una familia de modelos de lenguaje abiertos del sistema universitario argentino. Están adaptados a su
normativa, sus instituciones y su vida académica, y al español formal de la Argentina, y se publican con licencia
abierta para que cualquier persona o institución pueda usarlos. La desarrolla el Laboratorio de Investigación en Ciencia
de Datos & Inteligencia Artificial (LICDIA) de la Universidad Nacional de Luján. El nombre remite al Manifiesto Liminar
de la Reforma Universitaria de 1918, publicado por la Federación Universitaria de Córdoba. La Reforma impulsó la
autonomía universitaria y el cogobierno, y Liminar retoma la idea de autonomía con modelos abiertos que cada universidad
puede ejecutar en su propia infraestructura y adaptar a sus necesidades.

## Dos capas

La **capa nacional** son los modelos publicados: cada uno parte de un modelo abierto y se ajusta con la normativa
universitaria nacional curada y con textos públicos sobre la vida universitaria.

La **capa de cada universidad** es un adaptador LoRA que cada institución entrena con sus propios actos sobre un modelo
Liminar, siguiendo la [metodología](metodologia.md). Esa capa queda en la universidad.

Los modelos de la capa nacional también se pueden usar tal como se publican, sin volver a entrenarlos. Son modelos de
propósito general con conocimiento adicional del sistema universitario argentino, y se pueden aplicar a consultas con
recuperación de documentos (RAG) sobre la normativa de cada institución, resúmenes de expedientes y resoluciones,
borradores de notas, extracción de datos de documentos o asistentes de atención a estudiantes y docentes. En la
evaluación con documentos obtuvieron entre 4,90 y 5,00 sobre 5, el mismo nivel que sus modelos base.

## Contenido

<div class="grid cards" markdown>

-   :material-cube-outline:{ .lg .middle } __Modelos__

    ---

    La familia, las versiones y las licencias.

    [:octicons-arrow-right-24: Modelos](modelos.md)

-   :material-console:{ .lg .middle } __Uso__

    ---

    Cómo cargar un modelo y cómo usarlo con recuperación de documentos.

    [:octicons-arrow-right-24: Uso](uso.md)

-   :material-school-outline:{ .lg .middle } __Metodología para universidades__

    ---

    Cómo entrenar la capa propia con los actos de la institución.

    [:octicons-arrow-right-24: Metodología](metodologia.md)

-   :material-scale-balance:{ .lg .middle } __Evaluación__

    ---

    Cada modelo contra su base: dominio y funcionalidad.

    [:octicons-arrow-right-24: Evaluación](evaluacion.md)

</div>

Cómo se hizo: [corpus](corpus.md), [datos de entrenamiento](datos.md), [entrenamiento](entrenamiento.md) y
[cómputo](computo.md).

# Metodología para universidades

La capa propia de una universidad es un adaptador LoRA entrenado con sus actos sobre un modelo Liminar. Los pasos son
los mismos que se siguieron para la capa nacional, con el digesto de la universidad como corpus.

1. **Reunir el corpus.** Los actos de la universidad (resoluciones, ordenanzas, disposiciones) con su procedencia: número,
   fecha, órgano emisor y texto completo. Ver [Corpus](corpus.md).
2. **Generar los datos de entrenamiento.** Un modelo maestro escribe preguntas y respuestas sobre cada acto y versiones
   reescritas de cada fragmento. Se aparta una parte de los actos para evaluar. Ver [Datos de entrenamiento](datos.md).
3. **Entrenar el adaptador.** LoRA sobre el modelo Liminar elegido. Ver [Entrenamiento](entrenamiento.md).
4. **Evaluar contra el modelo sin la capa propia**, con preguntas sobre los actos apartados y con preguntas generales,
   para verificar que el modelo aprende lo nuevo sin perder lo que sabía. Ver [Evaluación](evaluacion.md).
5. **Servirlo**, de preferencia con recuperación de documentos sobre el digesto actualizado. Ver [Uso](uso.md).

!!! note "Datos personales"
    Los actos suelen nombrar personas. La capa de cada universidad no se publica: queda en la institución.

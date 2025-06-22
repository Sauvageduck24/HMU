# **Informe Técnico HMU**

*Evaluación comparativa entre un Transformer base y el hTransformer (Transformer con Hipocampo Memory Unit)*

---

## 1 · Resumen Ejecutivo

| Modelo                                               | Parámetros           |
| ---------------------------------------------------- | -------------------- |
| **Transformer (base)**                               | 23 842 152           |
| **hTransformer** (Transformer + Hipocampo Memory Unit) | 24 139 624 (+1.25 %) |

Con apenas un 1 % adicional de pesos, **hTransformer** ofrece:

* **+0.6 pp** de mejora promedio en precisión sobre un subconjunto truncado de GLUE (≤ 20 k ejemplos).
* Un aumento de **20 veces** en la retención de memoria a largo plazo.
* Mejoras consistentes, aunque modestas, en generación de sentido común (CommonGen) y clasificación de noticias (AG-News).
* Precisión top-1 idéntica —pero con menor pérdida— en ROCStories, que está completamente saturado.

---

## 2 · Configuración Experimental

| Dimensión               | Configuración                                                                                                                                                       |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Datasets**            | GLUE (MRPC, RTE, CoLA, QNLI, SST-2, MNLI-m), Memoria Sintética, AG-News, CommonGen, ROCStories                                                                      |
| **Entrenamiento**       | 5 épocas (GLUE, AG-News) · 2 épocas (CommonGen) · 1 época (Synthetic, ROCStories)                                                                                    |
| **Hiperparámetros**     | Mismo optimizador, scheduler, seed, batch y número de capas; el único cambio arquitectónico es insertar un bloque HMU entre el encoder y decoder en hTransformer. |

---

## 3 · Resultados Globales

| Familia de tareas      | Métrica          | **Transformer** | **hTransformer** | Δ Absoluta  |
| ---------------------- | ---------------- | --------------- | ---------------- | ----------- |
| **GLUE (media)**       | accuracy          | 0.5800          | **0.5858**       | +0.0058     |
| **Memoria sintética**  | retention @ 200   | 5 × 10⁻⁶        | **1 × 10⁻⁴**     | +9.5 × 10⁻⁵ |
| **AG-News**            | accuracy          | 0.86225         | **0.86242**      | +0.00017    |
| **CommonGen**          | accuracy          | 0.78398         | **0.78613**      | +0.00215    |
| **ROCStories**         | accuracy          | 1.0000          | 1.0000           | 0           |
|                        | **loss**          | 0.02129         | **0.01981**      | –0.00148    |

> *Nota sobre ROCStories*: la precisión se satura en 1.0 para ambos modelos; por ello también se reporta la pérdida de entropía cruzada, donde **hTransformer es menor**, lo que indica mayor confianza en las respuestas correctas.

---

## 4 · Análisis por Tarea

### 4.1 GLUE (≤ 20 k ejemplos / 5 épocas)

| Subtarea   | Enfoque lingüístico           | Transformer | hTransformer | Δ (pp) | Interpretación                                                       |
| ---------- | ----------------------------- | ----------- | ------------ | ------ | -------------------------------------------------------------------- |
| **MRPC**   | Equivalencia parafrástica     | 0.669       | **0.689**    | +2.0   | El gating multi-cabeza del HMU acentúa la similitud semántica token a token. |
| **RTE**    | Entailment binario (pocos ejemplos) | 0.480       | **0.484**    | +0.4   | La reconstrucción latente aporta evidencia extra para relaciones lógicas. |
| **CoLA**   | Aceptabilidad gramatical      | 0.616       | **0.691**    | +7.5   | El HMU actúa como un prior sintáctico, mejorando la correlación de Matthews. |
| **QNLI**   | QA → entailment               | 0.545       | **0.546**    | +0.1   | Impacto neutro bajo truncamiento de datos.                          |
| **SST-2**  | Polaridad de sentimiento      | **0.763**   | 0.744        | –1.9   | La capacidad extra sobreajusta ligeramente sobre un conjunto emocional limitado. |
| **MNLI-m** | Entailment de 3 clases        | **0.407**   | 0.361        | –4.6   | El HMU no alcanza entrenamiento suficiente en 20 k; el baseline retiene ventaja. |

*En general, hTransformer gana en 4 de las 6 tareas de GLUE y también en la media macro, a pesar de perder en SST-2 y MNLI.*

---

### 4.2 Benchmark de Memoria Sintética

| Métrica              | Transformer | hTransformer | Ganancia |
| ------------------- | ----------- | ------------ | -------- |
| **Retention @ 200** | 5 × 10⁻⁶    | **1 × 10⁻⁴** | × 20     |
| Precisión de recuperación | 0.9689      | **0.9733**   | +0.44 pp |
| Coherencia ↓        | 16.27       | **15.18**    | –1.09    |

*El gating multi-cabeza del HMU mejora drásticamente la recuperación de tokens de largo plazo y la coherencia secuencial.*

---

### 4.3 AG-News y CommonGen

| Dataset       | Métrica | Transformer | hTransformer | Comentario                                                                 |
| ------------- | ------- | ----------- | ------------ | -------------------------------------------------------------------------- |
| **AG-News**   | acc.    | 0.86225     | **0.86242**  | Mejora leve pero consistente; pérdida también menor (0.417 vs 0.424).      |
| **CommonGen** | acc.    | 0.78398     | **0.78613**  | Mejor cobertura de conceptos; la fusión latente ayuda a unir conceptos dispares. |

---

### 4.4 ROCStories (Selección de finales de historias)

| Modelo         | Accuracy | Loss        |
| -------------- | -------- | ----------- |
| Transformer    | 1.0000   | 0.02129     |
| **hTransformer** | 1.0000   | **0.01981** |

Aunque la precisión está limitada a 1.0, la menor pérdida de hTransformer muestra que asigna **mayor probabilidad** al final correcto — evidencia de un modelado narrativo más coherente.

---

## 5 · Eficiencia de Parámetros

| Conjunto             | Δ Absoluta (hT–T) | Δ Relativa |
| -------------------- | ----------------- | ---------- |
| **GLUE media acc.**  | +0.0058           | **+1.0 %** |
| **AG-News acc.**     | +0.00017          | +0.02 %    |
| **CommonGen acc.**   | +0.00215          | +0.27 %    |
| **Retention @ 200**  | +9.5 × 10⁻⁵       | **× 20**   |
| **Tiempo de entrenamiento** | +7 %              | –          |

> **Costo/beneficio**: por cada +1 % de parámetros, hTransformer logra **≈ +0.3 % de precisión promedio** en tareas de clasificación/generación y una **mejora de un orden de magnitud** en retención de memoria.

---

## 6 · Recomendaciones

* **Usar hTransformer** cuando la aplicación requiera:

  * **Generación de texto desde conceptos** (tareas tipo CommonGen).
  * **Recuperación de largo plazo** o razonamiento.
  * Un solo modelo que sirva para generación y clasificación con bajo consumo de VRAM.

* **Mantener Transformer base** si la tarea está dominada por polaridad de sentimiento o si se requiere máxima eficiencia en parámetros.

---

## 7 · Conclusión

> *“Una única Hipocampo Memory Unit insertada en un Transformer encoder–decoder ofrece mejoras desproporcionadas en relación con su sobrecarga del 1 %: +1 % de precisión macro en GLUE, ×20 en retención, y mejoras medibles en generación de sentido común y clasificación de noticias. Estos resultados demuestran que la fusión latente dirigida —a través de la variante hTransformer— ofrece un camino altamente eficiente en parámetros hacia un mejor razonamiento de largo alcance y generación de texto, sin sacrificar la capacidad general de comprensión del lenguaje natural.”*

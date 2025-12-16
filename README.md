# README
## Trabajo Práctico N°3, Procesamiento de Imágenes, de TUIA, FCEIA, UNR.

**Estudiantes**:
- Alejandro Armas, 
- Facundo Ferreira Dacámara, 
- Gabriel Soda

Este repositorio contiene un script desarrollado en Python para la detección automática y análisis de tiradas de dados en secuencias de video utilizando la librería OpenCV, así como un informe con el detalle correspondiente a su desarrollo. 
Enmarcados dentro de la cátedra de **Procesamiento de Imágenes**, de la **Tecnicatura Universitaria en Inteligencia Artificial**, correspondiente a la resolución del Trabajo Práctico N°3.


## Ejecución del script

El script se puede ejecutar enteramente en conjunto con los videos sobre los que hará las detecciones. Asume la existencia de archivos de entrada que cumplan el patrón `tirada_*.mp4` en el directorio de ejecución.

Todos los videos de prueba están suministrados en el repositorio mismo. Pueden intercambiarse por otros pero se debe respetar el patrón de nombres mencionado anteriormente.

En `informe_final_pdi_tp3.pdf` se presenta en detalle cómo fue el desarrollo del trabajo.

## Detección y Análisis de Tiradas de Dados (`problema.py`)

Implementa un sistema completo de detección y lectura automática de dados en video. El algoritmo se divide en tres fases principales:

* **Detección de Frames Estáticos:** Identifica automáticamente los frames donde los dados se encuentran en reposo mediante análisis de diferencias entre máscaras consecutivas. Implementa una máquina de estados con criterio de estabilidad (≤50 píxeles de diferencia durante >10 frames consecutivos) y tolera variaciones por autofoco mediante frames permitidos.

* **Segmentación Robusta:** Utiliza segmentación en espacio HSV para aislar dados rojos del fondo verde. Aplica filtrado Gaussiano (7×7) y operaciones morfológicas (cierre 11×11, apertura 5×5) para generar máscaras sólidas. El procesamiento se realiza a resolución reducida (factor 4×) para eficiencia computacional.

* **Lectura de Valores:** Detecta puntos blancos (pips) mediante segmentación HSV en resolución HD. Utiliza componentes conectadas con kernel elíptico para contar los puntos y determinar el valor de cada dado. Implementa cálculo único por secuencia estática para garantizar consistencia.

## Requisitos

* Python 3.x
* OpenCV (`opencv-python`)
* NumPy


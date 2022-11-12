# Reconocimiento-facial
El proceso que seguí para hacer esta tarea fue el siguiente:

Para cargar correctamente los datos del archivo list_attr_celeba.txt hice el archivo carga_datos.py donde corregí el problema de los separadores y los datos los dejé
tal cual estaban. El archivo corregido lo llamé list_attr_celeba1.txt
En el archivo reconocimiento_facial.py fue dond hice la red neuronal que detectara los atributos de las imágenes. Primero cargué el archivo list_attr_celeba1.txt como
un dataframe. Usando la función train_test_split de sklearn dividí el dataframe en uno de entrenamiento y en otro de testeo, donde el de testing es 20% de los datos
originales y el de train es el 80%. Así mismo, dividí los datos de entrenamiento y de test en tensores que contienen el nombre de las imágenes y en otro los atributos.
Tuve que buscar la forma de convertir los dataframes en tensores pues no podía usar los dos al momento de entrenar la red.
La función para cargar las imágenes y la red son similares a los ejemplos antes vistos, con algunas modificaciones.
Al querer entrenar la red con la primera configuración que hice obtuve resultados muy malos, precisión alrededor de 0.01. Pasé mucho tiempo variando los parámetros,
usando otras funciones de costo y optimizadores pero nada resultaba. Primero pensé que se debía a la enorme cantidad de datos que contenía el dataset de Celeba, así que
busqué una función que me diera solo una parte del dataset original, la que encontré fue df.sample(). Pero aún así el resultado no cambiaba. 
Después me di cuenta que estaba usando 1 y -1 como "las respuestas correctas" de los datos, edité el archivo de carga de datos para convertir los -1 a 0, pero el
resultado no cambió mucho. 
Luego pensé que los datos estaban mal organizados, así que probé con otra forma de crear los datos de entrenamiento y testing pero solo conseguí que al entrenar la red
mi memoria ram y procesador estuvieran al máximo de su capacidad.
Después de varios intentos, probé disminuyendo el número de atributos que estaba usando, porque originalmente estaba usando los 40 atributos. Vi una mejoría en el 
desempeño de la red pero aún no era lo suficientemente bueno. Continué disminuyendo hasta usar solo 1 atributo. Conseguí un 0.85 de precisión usando la activación
'relu' en la neurona de salida, pero estaba usando la función de costo 'categorical_crossentropy', esto causó que en obtuviera loss=nan y no pasara de la primera época.
Tuve que cambiar la función de costo a 'binary_crossentropy' y así fue como conseguí aproximadamente un 0.7 de precisión.
Para sguir el proceso sugerido en la asignación de la tarea de entrenar 3 veces la red usé el código sugerido pero no me funcionó, el error decía que las capas conv2d
no eran iterables. Buscando en internet alguna solución encontré que debía definir cada capa como una variable, agregarla a la red y después deshabilitar el entrenamiento
de dichas capas usando el nombre de la variable asignada a cada una, en internet econtré la función exec() que me ayudó a hacer todo el proceso con un ciclo for. 
Así la entrené tres veces pero no vi una mejoría en la precisión. Guardé el modelo final como modelo_final.h5.

Para el reconocimiento facial, por falta de tiempo no pude hacer un dataset con fotos mías. Busqué en internet algun dataset que contuviera fotos de una sola persona pero no encontré muchos. Buscando en kaggle (https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset?resource=download) encontré un dataset que contenía imagenes de los personajes de Los Simpson. Usé las imágenes de Homero como la persona que la reddebe reconocer y mezclé imaganes de otros personajes en otra carpeta. Con un ciclo for cargué las imagenes y le asigné un 1 a las imágenes de Homero y un 0 a las imágenes de los demás personajes. 
Cargué la red que había entrenado con los datos de Celeba pero ahora no sabía como deshabilitar el entrenamiento en todas las capas menos en la capa de salida. De nuevo, buscando en internet encontré la función model.summary() que muestra el nombre de todas las capas del modelo. Al final no sirvió mucho pues el código que venía en el PDF de la tarea sí me funcionó esta vez. Al entrenar la red me marcó un error en las dimensiones de los datos. 
No sabía como arreglarlo. Investigando encontré la función ImageDataGenerator de tensorflow que crea nuevas imágenes, al usar está función en el entrenamiento de la red se solucionó el error. Obtuve una precisión similar a la obtenida al usar los datos de Celeba.
Las capturas de pantalla del entrenamiento se encuentran en el repositorio, "entrenamiento de red.PNG" es el primer entrenamiento de la red con datos de Celeba, 
"entrenamiento de clasificador.PNG" es cuando solo se entrenó el clasificador y "re-entrenamiento de red.PNG" es cuando se volvió a entrenar la red completa.
"entrenamiento de red con simpson_dataset.PNG" es el entrenamiento de la red con las imágenes de Los Simpson.

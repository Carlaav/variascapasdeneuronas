print("Bienvenido. Vamos a entrenar un modelo para los datos sonares ")


def validar_entero(message):
    numero = input(message)
    while True:
        try:
            numero = int(numero)
            return numero
        except:
            print("ENTERO NO VALIDO. INTENTAR DE NUEVO")
            numero = input(message)


def validar_flotante(message):
    numero = input(message)
    while True:
        try:
            numero = float(numero)
            return numero
        except:
            print("FLOTANTE NO VALIDO. INTENTAR DE NUEVO")
            numero = input(message)


neuronas_capa_oculta = validar_entero("Ingresar la cantidad de neuronas en la capa oculta: ")
epochs = validar_entero("Ingresar la cantidad de epochs: ")
tasa_aprendizaje = validar_flotante("Ingresar la tasa de aprendizaje: ")
test_size_split = validar_flotante("Fracción de datos para testeo. (ej: 0.20) ")

import pandas as pnd

observaciones = pnd.read_csv("datas/sonar.all-data.csv")

# ---------------------------------------------
# PREPARACIÓN DE LOS DATOS
# ---------------------------------------------

print("N.º columnas: ", len(observaciones.columns))
# Para el parendizaje solo se toman los datos procedentes del sonar
X = observaciones[observaciones.columns[0:60]].values

# Solo se toman los etiquetados
y = observaciones[observaciones.columns[60]]

# Se codifica: Las minas son iguales a 0 y las rocas son iguales a 1
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

# Se añade un cifrado para crear clases:
# Si es una mina [1,0]
# Si es una roca [0,1]
import numpy as np

n_labels = len(y)
n_unique_labels = len(np.unique(y))
one_hot_encode = np.zeros((n_labels, n_unique_labels))
one_hot_encode[np.arange(n_labels), y] = 1
Y = one_hot_encode

# Verificación tomando los registros 0 y 97
print("Clase Rocha:", int(Y[0][1]))
print("Clase Mina:", int(Y[97][1]))

# ---------------------------------------------
# CREACIÓN DE LOS CONJUNTOS DE APRENDIZAJE Y DE PRUEBA
# ---------------------------------------------

# Mezclamos
from sklearn.utils import shuffle

X, Y = shuffle(X, Y, random_state=1)

# Creación de los conjuntos de aprendizaje
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=test_size_split, random_state=42)

# ---------------------------------------------
# PARAMETRIZACIÓN DE LA RED NEURONAL
# ---------------------------------------------
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()

cantidad_neuronas_entrada = 60
cantidad_neuronas_salida = 2

# Variable TensorFLow correspondiente a los 60 valores de las neuronas de entrada
tf_neuronas_entradas_X = tf.placeholder(tf.float32, [None, 60])

# Variable TensorFlow correspondiente a 2 neuronas de salida
tf_valores_reales_Y = tf.placeholder(tf.float32, [None, 2])

pesos = {
    # 60 neuronas de entradas hacia X Neuronas de la capa oculta
    'capa_entrada_hacia_oculta': tf.Variable(tf.random_uniform([60, neuronas_capa_oculta], minval=-0.3, maxval=0.3),
                                             tf.float32),

    # X neuronas de la capa oculta hacia 2 de la capa de salida
    'capa_oculta_hacia_salida': tf.Variable(tf.random_uniform([neuronas_capa_oculta, 2], minval=-0.3, maxval=0.3),
                                            tf.float32),
}

peso_sesgo = {
    # 1 sesgo de la capa de entrada hacia las X neuronas de la capa oculta
    'peso_sesgo_capa_entrada_hacia_oculta': tf.Variable(tf.zeros([neuronas_capa_oculta]), tf.float32),

    # 1 sesgo de la capa oculta hacia las 2 neuronas de la capa de salida
    'peso_sesgo_capa_oculta_hacia_salida': tf.Variable(tf.zeros([2]), tf.float32),
}


# ---------------------------------------------
# FUNCIÓN DE CREACIÓN DE RED NEURONAL
# ---------------------------------------------


def red_neuronas_multiccapas(observaciones_en_entradas, pesos, peso_sego):
    # Cálculo de la activación de la primera capa
    primera_activacion = tf.sigmoid(tf.matmul(tf_neuronas_entradas_X, pesos['capa_entrada_hacia_oculta']) + peso_sesgo[
        'peso_sesgo_capa_entrada_hacia_oculta'])

    # Cálculo de la activación de la segunda capa
    activacion_capa_oculta = tf.sigmoid(tf.matmul(primera_activacion, pesos['capa_oculta_hacia_salida']) + peso_sesgo[
        'peso_sesgo_capa_oculta_hacia_salida'])

    return activacion_capa_oculta


# ---------------------------------------------
# CREACIÓN DE LA RED NEURONAL
# ---------------------------------------------
red = red_neuronas_multiccapas(tf_neuronas_entradas_X, pesos, peso_sesgo)

# ---------------------------------------------
# ERROR Y OPTIMIZACIÓN
# ---------------------------------------------

# Función de error de media cuadrática MSE
funcion_error = tf.reduce_sum(tf.pow(tf_valores_reales_Y - red, 2))

# Descenso de gradiente con una tasa de aprendizaje fijada a 0,1
optimizador = tf.train.GradientDescentOptimizer(learning_rate=tasa_aprendizaje).minimize(funcion_error)

# ---------------------------------------------
# APRENDIZAJE
# ---------------------------------------------

# Inicialización de la variable
init = tf.global_variables_initializer()

# Inicio de una sesión de aprendizaje
sesion = tf.Session()
sesion.run(init)

# Para la realización de la gráfica para la MSE
Grafica_MSE = []

# Para cada epoch
for i in range(epochs):
    # Realicación del aprendizaje con actualización de los pesos
    sesion.run(optimizador, feed_dict={tf_neuronas_entradas_X: train_x, tf_valores_reales_Y: train_y})

    # Calcular el error
    MSE = sesion.run(funcion_error, feed_dict={tf_neuronas_entradas_X: train_x, tf_valores_reales_Y: train_y})

    # Visualización de la información
    Grafica_MSE.append(MSE)
    print("EPOCH (" + str(i) + "/" + str(epochs) + ") -  MSE: " + str(MSE))

# Visualización gráfica
import matplotlib.pyplot as plt

plt.plot(Grafica_MSE)
plt.ylabel('MSE')
plt.show()

# ---------------------------------------------
# VERIFICACIÓN DEL APRENDIZAJE
# ---------------------------------------------

# Las probabilidades de cada clase 'Mina' o 'roca' procedentes del aprendizaje se almacenan en el modelo.
# Con la ayuda de tf.argmax, se recuperan los índices de las probabilidades más elevados para cada observación
# Ejemplo: Si para una observación tenemos [0.56, 0.89] devolverá 1 porque el valor más elevado se encuentra en el índice 1
# Ejemplo: Si para una observación tenemos [0.90, 0.34] devolverá 0 porque el valor más elevado se encuentra en el índice 0
clasificaciones = tf.argmax(red, 1)

# En la tabla de los valores reales:
# Las minas se codifican como [1,0] y el índice que tiene el mayor valor es 0
# Las rocas tienen el valor [0,1] y el índice que tiene el mayor valor es 1

# Si la clasificación es [0.90, 0.34] el índice que tiene el mayor valor es 0
# Si es una mina [1,0] el índice que tiene el mayor valor es 0
# Si los dos índices son idénticos, entonces se puede afirmar que es una buena clasificación
formula_calculo_clasificaciones_correctas = tf.equal(clasificaciones, tf.argmax(tf_valores_reales_Y, 1))

# La precisión se calcula haciendo la media (tf.mean)
# de las buenas clasificaciones (después de haberlas convertido en decimal tf.cast, tf.float32)
formula_precision = tf.reduce_mean(tf.cast(formula_calculo_clasificaciones_correctas, tf.float32))

# -------------------------------------------------------------------------
# PRECISIÓN EN LOS DATOS DE PRUEBAS
# -------------------------------------------------------------------------

n_clasificaciones = 0;
n_clasificaciones_correctas = 0

# Miramos todo el conjunto de los datos de prueba (text_x)
for i in range(0, test_x.shape[0]):

    # Recuperamos la información
    datosSonar = test_x[i].reshape(1, 60)
    clasificacionEsperada = test_y[i].reshape(1, 2)

    # Realizamos la clasificación
    prediccion_run = sesion.run(clasificaciones, feed_dict={tf_neuronas_entradas_X: datosSonar})

    # Calculamos la precisión de la clasificación con la ayuda de la fórmula establecida antes
    accuracy_run = sesion.run(formula_precision, feed_dict={tf_neuronas_entradas_X: datosSonar,
                                                            tf_valores_reales_Y: clasificacionEsperada})

    # Se muestra para observación la clase original y la clasificación realizada
    print(i, "Clase esperada: ", int(sesion.run(tf_valores_reales_Y[i][1], feed_dict={tf_valores_reales_Y: test_y})),
          "Clasificacion: ", prediccion_run[0])

    n_clasificaciones = n_clasificaciones + 1
    if (accuracy_run * 100 == 100):
        n_clasificaciones_correctas = n_clasificaciones_correctas + 1

print("-------------")
print("Precisión en los datos de pruebas = " + str((n_clasificaciones_correctas / n_clasificaciones) * 100) + "%")

# -------------------------------------------------------------------------
# PRECISIÓN EN LOS DATOS DE APRENDIZAJE
# -------------------------------------------------------------------------

n_clasificaciones = 0;
n_clasificaciones_correctas = 0
for i in range(0, train_x.shape[0]):

    # Recuperamos las informaciones
    datosSonar = train_x[i].reshape(1, 60)
    clasificacionEsperada = train_y[i].reshape(1, 2)

    # Realizamos la claisifcación
    prediccion_run = sesion.run(clasificaciones, feed_dict={tf_neuronas_entradas_X: datosSonar})

    # Calculamos la precisión de la clasificación con la ayuda de la fórmula establecida antes
    accuracy_run = sesion.run(formula_precision, feed_dict={tf_neuronas_entradas_X: datosSonar,
                                                            tf_valores_reales_Y: clasificacionEsperada})

    n_clasificaciones = n_clasificaciones + 1
    if (accuracy_run * 100 == 100):
        n_clasificaciones_correctas = n_clasificaciones_correctas + 1

print("Precisión en los datos de aprendizaje = " + str((n_clasificaciones_correctas / n_clasificaciones) * 100) + "%")

# -------------------------------------------------------------------------
# PRECISIÓN EN EL CONJUNTO DE LOS DATOS
# -------------------------------------------------------------------------


n_clasificaciones = 0;
n_clasificaciones_correctas = 0
for i in range(0, 207):

    prediccion_run = sesion.run(clasificaciones, feed_dict={tf_neuronas_entradas_X: X[i].reshape(1, 60)})
    accuracy_run = sesion.run(formula_precision, feed_dict={tf_neuronas_entradas_X: X[i].reshape(1, 60),
                                                            tf_valores_reales_Y: Y[i].reshape(1, 2)})

    n_clasificaciones = n_clasificaciones + 1
    if (accuracy_run * 100 == 100):
        n_clasificaciones_correctas = n_clasificaciones_correctas + 1

print("Precisión en el conjunto de los datos = " + str((n_clasificaciones_correctas / n_clasificaciones) * 100) + "%")

sesion.close()

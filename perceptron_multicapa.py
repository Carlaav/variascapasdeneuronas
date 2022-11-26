#-----------------------------------------------------------------------------------------
# @Autor: Aurélien Vannieuwenhuyze
# @Empresa: Junior Makers Place
# @Libro:
# @Capítulo: 11 - Usar varias capas de neuronas
#
# Módulos necesarios:
#   NUMPY 1.16.3
#   MATPLOTLIB: 3.0.3
#   TENSORFLOW: 1.13.1
#
# Para instalar un módulo:
#   Haga clic en el menú File > Settings > Project:nombre_del_proyecto > Project interpreter > botón +
#   Introduzca el nombre del módulo en la zona de búsqueda situada en la parte superior izquierda
#   Elegir la versión en la parte inferior derecha
#   Haga clic en el botón install situado en la parte inferior izquierda
#-----------------------------------------------------------------------------------------

import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()
import numpy as np


#-------------------------------------
#    DATOS DE APRENDIZAJE
#-------------------------------------

#Se transforman los datos en decimales

valores_entradas_X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
valores_a_predecir_Y = [[0.], [1.], [1.], [0.]]



#-------------------------------------
#    PARÁMETROS DE LA RED
#-------------------------------------
#Variable TensorFLow correspondiente a los valores de neuronas de entrada
tf_neuronas_entradas_X = tf.placeholder(tf.float32, [None, 2])

#Variable TensorFlow correspondiente a la neurona de salida (predicción real)
tf_valores_realles_Y = tf.placeholder(tf.float32, [None, 1])


#Cantidad de neuronas en la capa oculta
n_neuronas_capa_oculta = 2

#PESOS
#Los primeros están 4 : 2 en la entrada (X1 y X2) y 2 pesos por entrada
pesos = tf.Variable(tf.random_normal([2, 2]), tf.float32)

#los pesos de la capa oculta están 2 : 2 en la entrada (H1 y H2) y 1 peso por entrada
peso_capa_oculta = tf.Variable(tf.random_normal([2, 1]), tf.float32)

#El primer sesgo contiene 2 pesos
sesgo = tf.Variable(tf.zeros([2]))

#El segundo sesgo contiene 1 peso
sesgo_capa_oculta = tf.Variable(tf.zeros([1]))

#Cálculo de la activación de la primera capa
#cálculo de la suma ponderada (tf.matmul) con ayuda de los datos X1, X2, W11, W12, W31, W41 y del sesgo
#después aplicación de la función sigmoide (tf.sigmoid)
activacion = tf.sigmoid(tf.matmul(tf_neuronas_entradas_X, pesos) + sesgo)

#Cálculo de la activación de la capa oculta
#cálculo de la suma ponderada (tf.matmul) con ayuda de los datos H1, H2, W12, W21 y del sesgo
#después aplicación de la función sigmoide (tf.sigmoid)
activacion_capa_oculta = tf.sigmoid(tf.matmul(activacion, peso_capa_oculta) + sesgo_capa_oculta)

#Función de error de media cuadrática MSE
funcion_error = tf.reduce_sum(tf.pow(tf_valores_realles_Y-activacion_capa_oculta,2))

#Descenso del gradiente con una tasa de aprendizaje fijada en 0,1
optimizador = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(funcion_error)

#Cantidad de epochs
epochs = 100000

#Inicialización de la variable
init = tf.global_variables_initializer()

#Inicio de una sesión de aprendizaje
sesion = tf.Session()
sesion.run(init)

#Para la realización de la gráfica para la MSE
Grafica_MSE=[]


#Para cada epoch
for i in range(epochs):

   #Realización del aprendizaje con actualización de los pesos
   sesion.run(optimizador, feed_dict = {tf_neuronas_entradas_X: valores_entradas_X, tf_valores_realles_Y:valores_a_predecir_Y})

   #Calcular el error
   MSE = sesion.run(funcion_error, feed_dict = {tf_neuronas_entradas_X: valores_entradas_X, tf_valores_realles_Y:valores_a_predecir_Y})

   #Visualización de la información
   Grafica_MSE.append(MSE)
   print("EPOCH (" + str(i) + "/" + str(epochs) + ") -  MSE: "+ str(MSE))


#Visualización gráfica
import matplotlib.pyplot as plt
plt.plot(Grafica_MSE)
plt.ylabel('MSE')
plt.show()


print("--- VERIFICACIONES ----")

for i in range(0,4):
    print("Observación:"+str(valores_entradas_X[i])+ " - Esperado: "+str(valores_a_predecir_Y[i])+" - Predicción: "+str(sesion.run(activacion_capa_oculta, feed_dict={tf_neuronas_entradas_X: [valores_entradas_X[i]]})))



sesion.close()

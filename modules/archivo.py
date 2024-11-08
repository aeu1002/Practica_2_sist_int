# Andrés Estévez Ubierna
def funcion_escalon(a,z):
    """ Función de activación 
    Parámetros
    ----------
        a -- array con los valores de sumatorio de los elementos de un caso de entrenamiento
        z -- valor del umbral para la función de activación

    Devolución
    --------
        yhat_vec -- array con valores obtenidos de g(f)
    """ 
    # función de activación
    yhat_vec = 1 if a > z else 0
    return yhat_vec
def entrena_perceptron(X, y, z, eta, t, funcion_activacion):
    """ Entrena un perceptron simple
    Parámetros
    ----------
        X -- valores de x_i para cada uno de los datos de entrenamiento
        y -- valor de salida deseada para cada uno de los datos de entrenamiento
        z -- valor del umbral para la función de activación
        eta -- coeficiente de aprendizaje
        t -- numero de epochs o iteraciones que se quieren realizar con los datos de entrenamiento
        funcion_activacion -- función de activación para el perceptrón.
    
    Devolución
    --------
        w -- valores de los pesos del perceptron
        J -- error cuadrático obtenido de comparar la salida deseada con la que se obtiene 
            con los pesos de cada iteración
    """  
    
    import numpy as np
    
    # inicialización de los pesos
    w = np.zeros(len(X[0]))       
    n = 0                           # numero de iteraciones se inicializa a 0                     
    
    # Inicialización de variables adicionales 
    yhat_vec = np.zeros(len(y))     # array para las predicciones de cada ejemplo
    errors = np.zeros(len(y))       # array para los errores (valor real - predicción)
    J = []                          # error total del modelo     
    
    
    while n < t:  
        ############## a completar (desde aqui) #################
        # para cada ejemplo x del conjunto de datos X
        #        Calcular el sumatorio de las entradas por los pesos (np.dot) 
        #        Se hace pasar el valor resultante por la función de activación (funcion_escalon)
        #        Para cada peso
        #              actualizar peso
        #######################################################
        for i in range(len(X)):
            a_i = np.dot(w,X[i])
            yhat_vec[i] = funcion_escalon(a_i,z)
            for j in range(len(w)):
                w[j] = w[j] + eta * (y[i] - yhat_vec[i]) * X[i][j]
                
          
        ############## a completar (hasta aqui) #################
        
        n += 1 # se incrementa el número de iteraciones 
        # calculo del error cuadrático del modelo
        # esto no es más que la suma del cuadrado de la resta entre el valor real 
        # y la predicción que se tiene en esa iteración
        for i in range(0,len(y)):     
            errors[i] = (y[i]-yhat_vec[i])**2
        J.append(0.5*np.sum(errors))
           
    # Devuelve los pesos y el error cuadrático
    return w, J

def predice(w,x,z, funcion_activacion):
    """ Función de para la predicción 
    Parámetros
    ----------
        w -- array con los pesos obtenidos en el entrenamiento del perceptrón
        x -- valores de x_i para cada uno de los datos de test
        z -- valor del umbral para la función de activación
        funcion_activacion -- función de activación para el perceptrón.

    Devolución
    --------
        y -- array con los valores predichos para los datos de test
    """ 
    import numpy as np
    y_pred=np.zeros(len(x))
    for i in range(len(x)):
        a_i=np.dot(w,x[i])
        y_pred[i]=funcion_escalon(a_i,z)
    
    return y_pred


def evalua(y_test, y_pred):
    """ Función de activación 
    Parámetros
    ----------
        y_test -- array con los valores salida conocidos para los datos de test
        y_test -- array con los valores salida estimados por el perceptrón para los datos de test

    Devolución
    --------
        acierto -- float con el valor del porcentaje de valores acertados con respecto al total de elementos
    """ 
    aciertos = 0 
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i]:
            aciertos += 1 
    acierto= aciertos / len(y_test)
   
    return acierto


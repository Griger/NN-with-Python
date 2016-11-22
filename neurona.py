class Neurona:
    pesos

    def __init__(self, pesos, fActivacion):
        self.pesos = pesos
        self.f = fActivacion

    def getOutput(self,entradas):
        #producto cartesiano de las entradas con los pesos

class Capa:

    def __init__(self, neuronas):
        self.neuronas = neuronas


    def getWeighs(self):
        #obtener por capas los pesos de las neuronas de la NN

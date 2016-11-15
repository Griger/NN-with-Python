En primer lugar hemos de asegurarnos de tener disponible en nuestra máquina Theano que va a ser la herramienta que empleemos para realizar los cálculos en **GPU** y así acelerar el proceso de aprendizaje de nuestras redes neuronales. En la propia [web de Theano](http://deeplearning.net/software/theano/install.html) tenemos instrucciones para instalarlo en diversos sistemas operativos, incluso empleando el cómodo gestor de módulos Python *pip*.

En mi caso empleo Arch en mi portátil así que he obtado por instalarlo como otro paquete más con el siguiente comando:

```bash
pacaur -S python-theano
```
Para comprobar que hemos instalado correctamente este paquete en nuestra máquina no tenemos más que arrancar una consola Python (comando python sin más) y comprobar que las dos instrucciones siguientes se ejecutan sin problemas:

```python
from theano import *
import thean.tensor as T
```
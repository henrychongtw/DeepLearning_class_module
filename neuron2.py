import theano
import theano.tensor as T
import random
import numpy

x = T.vector()
w = theano.shared(numpy.array([1., 1.]))
b = theano.shared(0.)

z = T.dot(w,x) + b
y = 1/(1 + T.exp(-z))

neuron = theano.function(inputs=[x],outputs=[y])

print(w.get_value())
# w.set_value([0.,0.])

for i in range(100):
    x = [random.random(), random.random()]
    print(x)
    print(neuron(x))

import theano
import theano.tensor as T
import numpy
from itertools import izip

x = T.vector()
w1 = theano.shared(numpy.random.randn(2))
w2 = theano.shared(numpy.random.randn(2))
b1 = theano.shared(numpy.random.randn(1))
b2 = theano.shared(numpy.random.randn(1))
w = theano.shared(numpy.random.randn(2))
b = theano.shared(numpy.random.randn(1))

# a1 = 1 / (1 + T.exp(-1 * (T.dot(w1,x) + b1 )))
# a2 = 1 / (1 + T.exp(-1 * (T.dot(w2,x) + b2 )))
# y = 1 / (1 + T.exp(-1 * (T.dot(w,[a1,a2]) + b )))
#


z1 = T.dot(w1,x) + b1
z2 = T.dot(w2,x) + b2
a1 = 1 / (1 + T.exp(-1 * z1))
a2 = 1 / (1 + T.exp(-1 * z2))

z = T.dot(w,[a1,a2]) + b
y = 1 / (1 + T.exp(-1 * z))

y_hat = T.scalar()
cost = T.sum((y-y_hat)**2)

#dw1,db1,dw2,db2,dw,db = T.grad(cost,[w1,b1,w2,b2,w,b])
dw,db,dw1,db1,dw2,db2 = T.grad(cost,[w,b,w1,b1,w2,b2])


def myUpdate(parameters, gradients):
    mu = 0.01
    parameters_updates = [(p, p - mu * g) for p,g in izip(parameters,gradients)]
    return parameters_updates

#gradient = theano.function(inputs=[x,y_hat],outputs=[y,cost],updates=myUpdate([w1,b1,w2,b2,w,b],[dw1,db1,dw2,db2,dw,db]))
gradient = theano.function(inputs=[x,y_hat],outputs=[y,cost],updates=myUpdate([w,b,w1,b1,w2,b2],[dw,db,dw1,db1,dw2,db2]))

for i in range(100000):
    y1,cost1 = gradient([0,0],0)
    y2,cost2 = gradient([0,1],0)
    y3,cost3 = gradient([1,0],0)
    y4,cost4 = gradient([1,1],1)
    #print cost1,cost2,cost3,cost4
    if i == 99999:
        print 'weight w :'
        print w.get_value()
        print '\n'

        print 'weight w1 :'
        print w1.get_value()
        print '\n'

        print 'weight w2 :'
        print w2.get_value()
        print '\n'

        print 'bias b :'
        print b.get_value()
        print '\n'

        print 'bias b1 :'
        print b1.get_value()
        print '\n'

        print 'bias b2 :'
        print b2.get_value()
        print '\n'

        print 'Testing input with 00,01,10,11:'
        print y1,y2,y3,y4

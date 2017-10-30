import theano
import theano.tensor as T


#f = theano.function(inputs=[x],outputs=y)
x1 = T.scalar()
x2 = T.scalar()

y1 = x1 * x2
y2 = x1 ** 2 + x2 ** 0.5

f = theano.function([x1,x2],[y1,y2])
z = f(3,4)
print(z)

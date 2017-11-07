import numpy as np
import pylab
from scipy.optimize import line_search

def steepest_descent(grad_fun,params,num_iters, *varargs):
    ## Learning Rates
    #eta = 0.1
    eta = 2
    #eta = 3

    ## Momentum
    alpha=0.7
    momentum=True

    d = np.ones(params.shape)
    d = d / np.linalg.norm(d)

    mom = np.zeros(params.shape)
    for i in range(num_iters):
        grad = grad_fun(params,*varargs)
        params_old = params
        if momentum:
            # Add momentum to the update
            mom = -eta*grad + alpha*mom
        else:
            # Just use the gradient
            mom = -eta*grad
        params = params + mom

        pylab.plot([params_old[0],params[0]],[params_old[1],params[1]],'-k',lw=2)
        raw_input("Press Enter to continue...")

def ls_fun(params,A):
    return ls(params,A)[0]

def ls_grad(params,A):
    return ls(params,A)[1]

def ls(params,A):
    f  = 0.5*np.dot(params,A).dot(params)
    df = np.dot(A,params)

    return f,df

def ls_contour(X,Y,A):
    x = X.ravel()[:,None]
    y = Y.ravel()[:,None]
    data = np.hstack((x,y))
    z = 0.5*(np.dot(data,A)*data).sum(1)
    return z.reshape(X.shape)

if __name__ == '__main__':
    np.random.seed(0)
    A = np.random.randn(2,2)
    A = np.dot(A.T,A)
    A = np.dot(A.T,A)
    A = A / np.linalg.norm(A)
    x = np.linspace(-5,5,100)
    X,Y = np.meshgrid(x,x)
    Z = ls_contour(X,Y,A)
    #Z = rosenbrock_contour(x)

    pylab.ion()
    pylab.contour(X,Y,Z,100)
    pylab.show()

    init_params = np.array([4,-4])
    #init_params  = np.array([-3,-4])
    pylab.plot(init_params[0],init_params[1],'.r',ms=25)

    raw_input("Press Enter to continue...")

    steepest_descent(ls_grad,init_params,1000,A)

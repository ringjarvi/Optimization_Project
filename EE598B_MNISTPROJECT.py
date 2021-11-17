import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
from mlxtend.data import loadlocal_mnist

X, y = loadlocal_mnist(
        images_path='/home/alexander/Documents/CSM-Statistics /Semester 2/Numerical Optimization/Project/train-images-idx3-ubyte',
        labels_path='/home/alexander/Documents/CSM-Statistics /Semester 2/Numerical Optimization/Project/train-labels-idx1-ubyte')

X_test, y_test = loadlocal_mnist(
        images_path='/home/alexander/Documents/CSM-Statistics /Semester 2/Numerical Optimization/Project/t10k-images-idx3-ubyte',
        labels_path='/home/alexander/Documents/CSM-Statistics /Semester 2/Numerical Optimization/Project/t10k-labels-idx1-ubyte')


X = X[0:1000,:]
X = X/255
y = y[0:1000,]
y = np.equal.outer(y,np.unique(y)).astype(int)

X_test = X_test[0:1000,:]
X_test = X_test/255
y_test = y_test[0:1000,]
y_test = np.equal.outer(y_test,np.unique(y_test)).astype(int)



def softmax(x):
    ps = np.empty(x.shape)
    for i in range(x.shape[0]):
        ps[i,:] = np.exp(x[i,:] - np.max(x[i,:]))
        ps[i,:]/= np.sum(ps[i,:])
    return ps

def h(X,w,b):
    #The logistic function
    #X : Data
    #w : Weightsdef softmax(x):
    return softmax(X@w + (np.ones((X.shape[0],1))@b))


def cross_entropy(X, w, b, y):
    #The Cross Entropy Function
    #X : Data
    #w : Weights
    #y : labels
    #lam : regularization coefficient for Tikonoff Regularization
    n = X.shape[0]
    a = h(X, w, b)
    #return (1/n)*np.sum(-y*np.log(h(X, w, b)) + (1 - y)*np.log(1 - h(X, w, b)))
    return -(1/n)*np.sum(y*np.log(a))

def cross_entropy_grad(X,w,b,y):
    #First uses correct function to zero out any values of 1's that match y one-hot encoded
    a=h(X,w,b)
    gradw = X.T@(a - y)
    gradb = np.ones((1,X.shape[0]))@(a - y)
    return gradw, gradb

def backtracking(xk, dk, feval, grad, alpha0, rho, c):
    fvk = feval(xk)
    gk = grad(xk)
    alpha = alpha0
    while feval(xk + alpha*dk) > fvk + c*alpha*gk@dk:
        alpha = rho*alpha
    return alpha

def eval(x,w,b,y):

    a=np.argmax(softmax(x@w+b),axis=1).tolist()
    b=np.argmax(y,axis=1).tolist()
    total_percent=0
    o=0
    t=0
    th=0
    f=0
    fi=0
    s=0
    se=0
    e=0
    n=0
    z=0
    for i in range(x.shape[0]):
            if a[i]==b[i]:
                total_percent+=1
                if a[i]==1:
                    o+=1
                if a[i]==2:
                    t+=1
                if a[i]==3:
                    th+=1
                if a[i]==4:
                    f+=1
                if a[i]==5:
                    fi+=1
                if a[i]==6:
                    s+=1
                if a[i]==7:
                    se+=1
                if a[i]==8:
                    e+=1
                if a[i]==9:
                    n+=1
                if a[i]==0:
                    z+=1
    print("Total percent correct=",(total_percent/x.shape[0])*100)
    print("Percent of 0's correct=", z/b.count(0)*100)
    print("Percent of 1's correct=", o/b.count(1)*100)
    print("Percent of 2's correct=", t/b.count(2)*100)
    print("Percent of 3's correct=", th/b.count(3)*100)
    print("Percent of 4's correct=", f/b.count(4)*100)
    print("Percent of 5's correct=", fi/b.count(5)*100)
    print("Percent of 6's correct=", s/b.count(6)*100)
    print("Percent of 7's correct=", se/b.count(7)*100)
    print("Percent of 8's correct=", e/b.count(8)*100)
    print("Percent of 9's correct=", n/b.count(9)*100)


def steepest_descent(w0, b0, feval, grad, stepsize, options = {'gtol':1e-6,'MaxIter':1e3,'disp':True, 'disp_interval':100}):

    """
    x0: initialization for the algorithm
    feval: function handle that evaluates the objective function value at a given point
    grad: function handle that evaluates the gradient value at a given point
    stepsize: a function handle to compute stepsize given the current point and the direction.
              Using a function handle allows us to deal with different stepsize schemes such
              as constant, exact line search, and Wolfes' cnoditions
    options: options for the algorithms that provides the gradient tolerance for stopping, display, etc.
    """
    if 'gtol' in options:
        gtol = options['gtol']
    else:
        gtol = 1e-6

    if 'disp' in options:
        disp = options['disp']
    else:
        disp = True

    if 'MaxIter' in options:
        MaxIter = options['MaxIter']
    else:
        MaxIter = 1e3

    if 'disp_interval' in options:
        disp_interval = options['disp_interval']
    else:
        disp_interval = 100
    b = b0
    w = w0
    fv = feval(w, b)
    gw = grad(w, b)
    gb = gradb(w, b)
    ng = np.linalg.norm(gw)
    history = []
    iter_count = 0
    if disp:
        print('{:15s}{:20s}{:20s}{:20s}{:15s}'.format('Iteration #', 'Function Value', 'Gradient Norm w', 'Gradient Norm b', 'Step Size'))
    while ng > gtol and iter_count <= MaxIter: # Use the gradient as a stopping criteria
        alpha_a = stepsize_a(w, b, -gw, iter_count)
        alpha_b = stepsize_b(w, b, -gw, iter_count)
        w = w - alpha_a*gw
        b = b - alpha_b*gb
        # Record the history
        fv = feval(w, b)
        gw = grad(w, b)
        gb = gradb(w, b)
        ng = np.linalg.norm(gw)
        ngb = np.linalg.norm(gb)
        history.append((fv, ng))
        iter_count += 1
        if disp and iter_count % disp_interval == 0:
            print('{:<15d}{:<20.1f}{:<20.8f}{:<20.8f}{:<15.6f}'.format(iter_count, fv, ng, ngb, alpha_a))

    return w, b, fv, gw, history, iter_count
'''
#Initialization using orthonormal basis
w0 = np.random.normal(0,.1, (X.shape[1],y.shape[1]))
b0 = np.random.normal(0,.1, (1,y.shape[1]))

eps = 1e-1
MaxIter = 20000
disp_interval = 10
options = {'gtol':eps, 'MaxIter':MaxIter, 'disp_interval':disp_interval}

lam = 0

feval = lambda w, b: cross_entropy(X, w, b, y)
grad = lambda w, b: cross_entropy_grad(X, w, b, y)[0]
gradb = lambda w, b: cross_entropy_grad(X, w, b, y)[1]

# the stepsize is a function handle, which returns a constant stepsize in this case
# The given value is only a place holder
bk = b0
wk = w0
dk = grad(w0, b0)
stepsize_a = lambda wk, bk, dk, i: 1/np.sqrt(i+1)
stepsize_b = lambda wk, bk, dk, i:  1/np.sqrt(i+1)
#stepsize = lambda wk, dk: backtracking(wk, dk, feval, grad, 1, 0.5, 1e-4)
w, b, fv, g, history, iter_count1 = steepest_descent(w0, b0, feval, grad, stepsize_a, options)

# extract the function values and gradient norms
history_fv1 = [item[0] for item in history]
history_ng1 = [item[1] for item in history]

evaluation = eval(X,w,b,y)
print('The model evaluation is:', evaluation)

testeval = eval(X_test,w,b,y_test)
print('The test model evaluation is:', testeval)
'''








def SGD(w0, b0, feval, grad, stepsize, options = {'gtol':1e-6,'MaxIter':1e3,'disp':True, 'disp_interval':100}):

    """
    x0: initialization for the algorithm
    feval: function handle that evaluates the objective function value at a given point
    grad: function handle that evaluates the gradient value at a given point
    stepsize: a function handle to compute stepsize given the current point and the direction.
              Using a function handle allows us to deal with different stepsize schemes such
              as constant, exact line search, and Wolfes' cnoditions
    options: options for the algorithms that provides the gradient tolerance for stopping, display, etc.
    """
    if 'gtol' in options:
        gtol = options['gtol']
    else:
        gtol = 1e-6

    if 'disp' in options:
        disp = options['disp']
    else:
        disp = True

    if 'MaxIter' in options:
        MaxIter = options['MaxIter']
    else:
        MaxIter = 1e3

    if 'disp_interval' in options:
        disp_interval = options['disp_interval']
    else:
        disp_interval = 100
    b = np.concatenate(X,axis = 0).ravel(order = 'F').reshape(X.shape[0],y.shape[1])
    w = np.concatenate(X,axis =0).ravel(order = 'F').reshape(X.shape[1,y.shape[1]])
    fv = feval(w, b)
    gw = grad(w, b)
    gb = gradb(w, b)
    ng = np.linalg.norm(gw)
    history = []
    iter_count = 0
    if disp:
        print('{:15s}{:20s}{:20s}{:20s}{:15s}'.format('Iteration #', 'Function Value', 'Gradient Norm w', 'Gradient Norm b', 'Step Size'))
    while ng > gtol and iter_count <= MaxIter: # Use the gradient as a stopping criteria
        alpha_a = stepsize_a(w, b, -gw, iter_count)
        alpha_b = stepsize_b(w, b, -gw, iter_count)
        w = w - alpha_a*gw
        b = b - alpha_b*gb
        # Record the history
        fv = feval(w, b)
        gw = grad(np.concatenate(X,axis = 0).ravel(order = 'F'), np.concatenate(X,axis = 0).ravel(order = 'F'))
        gb = gradb(np.concatenate(X,axis = 0).ravel(order = 'F'), np.concatenate(X,axis = 0).ravel(order = 'F'))
        ng = np.linalg.norm(gw)
        ngb = np.linalg.norm(gb)
        history.append((fv, ng))
        iter_count += 1
        if disp and iter_count % disp_interval == 0:
            print('{:<15d}{:<20.1f}{:<20.8f}{:<20.8f}{:<15.6f}'.format(iter_count, fv, ng, ngb, alpha_a))

    return w, b, fv, gw, history, iter_count
'''
#Initialization using orthonormal basis
w0 = np.random.normal(0,0.1, (X.shape[1],y.shape[1]))
b0 = np.random.normal(0,0.1, (1,y.shape[1]))

eps = 1e-2
MaxIter = 20000
disp_interval = 10
options = {'gtol':eps, 'MaxIter':MaxIter, 'disp_interval':disp_interval}

lam = 0

feval = lambda w, b: cross_entropy(X, w, b, y)
grad = lambda w, b: cross_entropy_grad(X, w, b, y)[0]
gradb = lambda w, b: cross_entropy_grad(X, w, b, y)[1]

# the stepsize is a function handle, which returns a constant stepsize in this case
# The given value is only a place holder
bk = b0
wk = w0
dk = grad(w0, b0)
stepsize_a = lambda wk, bk, dk, i: 1/np.sqrt(i+1)
stepsize_b = lambda wk, bk, dk, i:  1/np.sqrt(i+1)
#stepsize = lambda wk, dk: backtracking(wk, dk, feval, grad, 1, 0.5, 1e-4)
w, b, fv, g, history, iter_count1 = steepest_descent(w0, b0, feval, grad, stepsize_a, options)

# extract the function values and gradient norms
history_fv1 = [item[0] for item in history]
history_ng1 = [item[1] for item in history]

evaluation = eval(X,w,b,y)
print('The training model evaluation is:', evaluation)

testeval = eval(X_test,w,b,y_test)
print('The test model evaluation is:', testeval)



'''
def L_BFGS(w0, b0, feval, grad, m, options = {'gtol':1e-6,'MaxIter':1e3,'disp':True, 'disp_interval':100}):

    if 'gtol' in options:
        gtol = options['gtol']
    else:
        gtol = 1e-6

    if 'disp' in options:
        disp = options['disp']
    else:
        disp = True

    if 'MaxIter' in options:
        MaxIter = options['MaxIter']
    else:
        MaxIter = 1e3

    if 'disp_interval' in options:
        disp_interval = options['disp_interval']
    else:
        disp_interval = 100

    w = w0
    fv = feval(w, b)
    g = grad(w, b)
    ng = np.linalg.norm(g)
    # Initialize the Memoery for storing sk and yk using gradient descent
    alpha = backtracking(w, -g, feval, grad, 1, 0.5, 1e-4)
    s = -alpha*g
    y = grad(x+s) - g
    Memory = [[s, y]]
    history = []
    iter_count = 0

    def two_loop(q0, Memory):
        q = q0
        m = len(Memory)
        alpha = np.zeros((m,1))
        for i, (s, y) in enumerate(reversed(Memory)):
            rho = 1/(y@s)
            alpha[-(i+1)] = rho*s@q
            q = q - alpha[-(i+1)]*y
        s, y = Memory[-1]
        gamma = s@y/(y@y)
        r = gamma*q
        for i, (s, y) in enumerate(Memory):
            rho = 1/(y@s)
            beta = rho*y@r
            r = r + s*(alpha[i]-beta)
        return r

    if disp:
        print('{:15s}{:20s}{:20s}{:15s}'.format('Iteration #', 'Function Value', 'Gradient Norm', 'Step Size'))

    while ng > eps and iter_count <= MaxIter:
        d = -two_loop(g, Memory)
        alpha = backtracking(x, d, feval, grad, 1, 0.5, 1e-4)
        x_prev = x
        x = x_prev + alpha*d
        s = x - x_prev
        g_prev = g
        g = grad(x)
        ng = np.linalg.norm(g)
        y = g - g_prev
        if len(Memory) >= m:
            Memory.pop(0)
        Memory.append([s,y])
        fv = feval(x)
        history.append((fv, ng))
        iter_count += 1
        if disp and iter_count % disp_interval == 0:
            print('{:<15d}{:<20.1f}{:<20.8f}{:<15.6f}'.format(iter_count, fv, ng, alpha))
    return x, fv, g, history, iter_count

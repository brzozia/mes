from functools import partial
import numpy as np
import matplotlib.pyplot as plt


N = 10
x_start = 0
x_end = 1
length = x_end - x_start
floor = length/N*2

# def a(x):
#     return 2

# def b(x):
#     return -3

# def c(x):
#     return 1

# def f(x):
#     return x**2

# beta = 1
# gamma = 2
# u1 = -1

def a(x):
    return -(x**2)-1

def b(x):
    return 1+(4*x)

def c(x):
    return -4

def f(x):
    return (2*(x**2))-(4*x)+3

beta = -1/2
gamma = 1
u1 = 0



def derivative(f, x):
    h = 0.00000001
    return (f(x + h) - f(x - h))/(2*h)


def integral(f, start, stop):
    step = 0.01
    sum = 0
    for i in np.arange(start, stop, step):
        sum += f(i+step/2)*step
    return sum
    

def pyramid(top, x):
    if((top - floor/2) <= x and x <= top):
        return (x - (top - floor/2))/(floor/2)
    elif((top + floor/2) >= x and x > top):
        return ((top + floor/2) - x)/(floor/2)
    else:
        return 0

def generate_tops():
    tops = []
    for i in np.arange(x_start, x_end, floor/2):
        tops.append(i)
    tops.append(x_end)
    return tops

def generate_pyramids(tops):
    pyramids = []
    for top in tops:
        pyramids.append(partial(pyramid, top))
    return pyramids

def L_v(f, v, gamma):
    tmp_f = partial(lambda f, v, x: f(x)*v(x), f, v)
    return (integral(tmp_f, x_start, x_end) - gamma*v(0))

def B_u_v(u, v, a, b, c, beta):
    dvdx = partial(derivative, v)
    dudx = partial(derivative, u)
    tmp_func = partial(lambda v, u, a, b, c, dv, du, x: (-1)*dv(x)*a(x)*du(x) + b(x)*du(x)*v(x) + c(x)*u(x)*v(x), v, u, a, b, c, dvdx, dudx)
    return (-1)*(beta*u(0)*v(0))+integral(tmp_func, 0, 1)

def plot_func(f):
    X = []
    Y = []
    step = 0.001
    for j in np.arange(0,1,step):
        X.append( round(j, 4) )
        Y.append( round(f(j),4) )

    plt.plot(X,Y)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.title('Output Function')
    plt.show()



tops = generate_tops()
shape_functions = generate_pyramids(tops)

A = []
row = []
for i in range(0,N):
    for j in  (shape_functions):
        row.append(B_u_v(j,shape_functions[i],a,b,c,beta))
    A.append(row)
    row = []

for j in range(len(shape_functions)-1):
    row.append(0.0)
row.append(1.0)
A.append(row)

B = []
for i in range(len(shape_functions)-1):
    B.append(L_v(f,shape_functions[i], gamma))
B.append(u1)

res = np.linalg.solve(np.array(A), np.array(B))


def result_template(shapes, coef, x):
    value = 0
    for i in range(0,N+1):
        value += coef[i]*shapes[i](x)
    return value

final_func = partial(result_template, shape_functions, res)

plot_func(final_func)


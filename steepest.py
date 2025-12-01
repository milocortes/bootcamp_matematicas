import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import jax
    return jax, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Método Steepest Descent

    Todos los métodos de iteración requieren especificar un punto de inicio $\boldsymbol{\theta}_{0}$. En cada iteración $t$ realizan una actualización siguiendo la siguiente regla:


    \begin{equation}
    	\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_{t} + \rho_t \boldsymbol{d}_{t}
    \end{equation}


    donde $\rho_t$ se le conoce como **tamaño de paso** o **tasa de aprendizaje**, y $\boldsymbol{d}_t$ es una **dirección de descenso**.

    Cuando la dirección de descenso es igual al negativo del gradiente ($\textit{i.e}$ $\boldsymbol{d}_t = - \boldsymbol{g}_t $)(Recuerda que el gradiente apunta en la dirección de máximo incremento en $f$, por eso el negativo apunta en la dirección de máxima disminución), la dirección se le conoce como de **steepest descent**.


    \begin{equation}
    	\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_{t} - \rho_t \boldsymbol{g}_{t}
    \end{equation}


    Utilizando una tasa de aprendizaje constante $\rho_t = \rho$, la regla de actualización es:


    \begin{equation}
    	\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_{t} - \rho \boldsymbol{g}_{t}
    \end{equation}


    Para el caso univariado, la regla de actualización es:



    \begin{equation}
    	x_{t+1} = x_{t} - \rho f^\prime (x_{t})
    \end{equation}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Steepest Descent en 1D

    Sea la función univariada:

    \begin{equation}
        f(x) = 6x^2 - 12x +3
    \end{equation}

    Graficamos la función
    """)
    return


@app.cell
def _(np, plt):
    def f(x):
        return 6*x**2 - 12*x +3

    X = np.linspace(-1, 3, 1000)
    y = f(X)

    plt.plot(X, y, label = r'$f(x) = 6x^2 - 12x +3$')
    plt.legend()
    plt.show()
    return X, f, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Como puese verse, la función es una parábola, por lo cual tiene un mínimo local (global).

    ## Cálculo manual de la derivada
    Obtengamos el mínimo de forma analítica al obtener la derivada e igualar a cero.

    \begin{equation}
        \frac{d}{dx} (6x^2 - 12x +3) = 12x - 12
    \end{equation}

    Igualando $12x - 12=0$, tenemos que el mínimo es $x=1$.

    ## Cálculo de la derivada con diferenciación simbólica

    Obtengamos el minimo mediante el método steepest descent.
    """)
    return


@app.cell
def _(f):
    import sympy

    ## Definimos la variable a usar como expresión en la diferenciación simbólica
    x_sym = sympy.symbols('x')

    ## Creamos la expresión simbólica al pasar la variable a la función
    f_sym = f(x_sym)

    ## Calculamos la derivada simbólica
    df_sym = sympy.diff(f_sym)
    df_sym
    return df_sym, sympy, x_sym


@app.cell
def _(df_sym, sympy, x_sym):
    # Convertimos la expresión de SymPy en una expresión que pueda ser evaluada numéricamente
    df_sympy = sympy.lambdify(x_sym, df_sym) 

    # Evaluamos la derivada para x=1
    df_sympy(1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cálculo de la derivada con Diferenciación Automática

    La [diferenciación automática](https://en.wikipedia.org/wiki/Automatic_differentiation) es un método para evaluar derivadas de funciones representadas como programas usualamente conocido como gráfica de cómputo.[[Automatic Differentiation in Machine Learning: a Survey, Baydin et. al, 2018](https://arxiv.org/abs/1502.05767)]. El programa está compuesto por operaciones elementales como sumas, restas, multiplicaciones y divisiones.

    Una gráfica de cómputo representa una función donde los nodos son operaciones y las aristas son relaciones de entrada-salida. Los nodos hoja de una gráfica computacional son variables de entrada o constantes, y los nodos terminales son valores de salida de la función.

    Hay dos métodos de diferenciación automática usando una gráfica de cómputo.

    * **Forward accumulation** : el método usa **números duales** para recorrer el árbol desde las entradas hasta las salidas.
    * **Backward accumulation** : recorre el árbol de las salidas a las entradas.
    """)
    return


@app.cell
def _(f, jax):
    # Obtenemos la primer derivada de la función con diferenciación automática con JAX
    df_jax = jax.grad(f)

    # Evaluamos la derivada para x=1
    df_jax(1.0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Implementación de Algoritmo de Optimización de Primer Orden Steepest Descent
    """)
    return


@app.cell
def _(X, f, mo, plt, y):
    ## Definimos la función de la primer derivada
    def df(x):
        return 12*x -12

    ## Definimos el punto de inicio
    x = -0.9
    x = xold = -0.9

    ## Definimos tamaño de paso
    rho = 0.16

    ## Definimos cantidad de iteraciones
    n = 20

    plt.plot(X, y)

    for i in range(20):
        plt.plot([xold,x], [f(xold),f(x)], marker='o', linestyle='dotted', color='r')
        xold = x
        # Utilizamos la regla de actualización
        x = x - rho * df(x)

    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Steepest Descent en 2D

    Sea la función que recibe dos argumentos:


    \begin{equation}
            f(x,y) = 6x^2 + 9y^2 - 12x -14y +3
    \end{equation}

    Obtenemos el gradiente:

    \begin{equation}
    \nabla f(x,y)=  \begin{bmatrix}
    \frac{\partial f(x,y)}{\partial x} \\
    \frac{\partial f(x,y)}{\partial y}
    \end{bmatrix} =
     \begin{bmatrix}
    12x -12 \\
    18y -14
    \end{bmatrix}
    \end{equation}
    """)
    return


@app.cell
def _(mo, np, plt):
    #  Gradient descent steps
    from mpl_toolkits.mplot3d import Axes3D 

    #  Function and partial derivatives
    def f2d(x,y):
        return 6*x**2 + 9*y**2 - 12*x - 14*y + 3

    N = 100
    X2d,Y2d = np.meshgrid(np.linspace(-1,3,N), np.linspace(-1,3,N))
    Z = f2d(X2d,Y2d)

    # Graficamos la superficie de la función
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_surface(X2d, Y2d, Z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.set_title('Superficie de la función $f(x,y)$');
    mo.mpl.interactive(plt.gcf())
    return X2d, Y2d, Z


@app.cell
def _(X2d, Y2d, Z, plt):
    def actualiza_superficie():
        plt.contourf(X2d,Y2d,Z,10, cmap="Greys")
        plt.contour(X2d,Y2d,Z,10, colors='k', linewidths=1)
        plt.plot([0,0],[-1,3],color='k',linewidth=1)
        plt.plot([-1,3],[0,0],color='k',linewidth=1)
        plt.plot(1,0.7777778,color='k',marker='+')

    actualiza_superficie()
    plt.show()
    return (actualiza_superficie,)


@app.cell
def _(actualiza_superficie, mo, np, plt):
    # La función del gradiente 
    def gradiente(X):
        x, y = X
        dx = 12*x - 12
        dy = 18*y - 14
        return np.array([dx,dy])

    # Modificamos la función multivariada
    def f2d_vec(X):
        x,y = X
        return 6*x**2 + 9*y**2 - 12*x - 14*y + 3

    ## Utilizamos el método para la función multivariada
    x2d = xold2d = -0.5
    y2d = yold2d = 2.9

    # Tasa de aprendizaje
    rho2d = 0.04

    # Vector inicial
    X_vec = np.array([x2d,y2d])

    n2d = 30 

    actualiza_superficie()

    for it in range(n2d):
        plt.plot([xold2d,x2d],[yold2d,y2d], marker='o', linestyle='dotted', color='red')
        xold2d = x2d
        yold2d = y2d
    
        # Utilizamos la regla de actualización
        X_vec = X_vec - rho2d*gradiente(X_vec)
        x2d, y2d = X_vec

    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    mo.mpl.interactive(plt.gcf())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparación de de distintos Métodos de Optimización de Primer Orden
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(
        src="https://raw.githubusercontent.com/milocortes/mod_04_concentracion/refs/heads/ccm-2025/src/notebooks/python/2D_steepest_newton_momentum_adam.gif",
        alt="Marimo logo",
        width=700,
        height=550,
        rounded=True,
        caption="Comparación de distintos métodos de optimización de primer orden",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(
        src="https://raw.githubusercontent.com/milocortes/mod_04_concentracion/refs/heads/ccm-2025/src/notebooks/julia/optimizadores.gif",
        alt="Marimo logo",
        width=900,
        height=600,
        rounded=True,
        caption="Marimo logo",
    )
    return


if __name__ == "__main__":
    app.run()

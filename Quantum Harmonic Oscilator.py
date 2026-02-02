
# ------------------------------------------------------------------
# Librerias usadas
# ------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Declaración de variables
# ------------------------------------------------------------------

m = 1
hbar = 1
omega = 1
L = 50

# ------------------------------------------------------------------
# Espacio de Trabajo
# ------------------------------------------------------------------

N = 500
x = np.linspace(-L, L ,N) # Discretizamos
dx = x[1] - x[0]

# ------------------------------------------------------------------
# Matriz del operador cinético (Segunda Derivada)
# ------------------------------------------------------------------

T = np.zeros((N,N))
for i in range(N):
    T[i,i] = -2.0
    if i > 0:
        T[i,i-1] = 1.0
    if i < N-1:
        T[i,i+1] = 1.0

T = (-hbar**2 / (2*m*dx**2)) * T 

# ------------------------------------------------------------------
# Potencial Oscilador Armónico
# ------------------------------------------------------------------

def opX(m,omega,x):
    return 0.5*m* omega**2 * x**2



Pot = [opX(m, omega, j) for j in x]

V = np.diag(Pot)

# ------------------------------------------------------------------
# Hamiltoniano Total
# ------------------------------------------------------------------

H = T + V

# ------------------------------------------------------------------
# Resolviendo el problema de Schrödinger
# ------------------------------------------------------------------

E, psi = np.linalg.eigh(H)

    # Normalización
psi = psi / np.sqrt(dx)

# ------------------------------------------------------------------
# Calculamos la densidad de probabilidad
# ------------------------------------------------------------------

psi2 = np.square(psi)

# ------------------------------------------------------------------
# Energías analíticas para comparación
# ------------------------------------------------------------------

n_lvl = 5
n = np.arange(1, n_lvl+1)
E_anal = []
for i in n-1:
    E_anal_Temp = (i + 0.5 ) * hbar * omega
    E_anal.append(E_anal_Temp)


# ------------------------------------------------------------------
# Gráficas
# ------------------------------------------------------------------

plt.figure(figsize=(15,10))

    # (1) Espectro de energías
plt.subplot(1,2,1)
plt.plot(range(1, n_lvl+1), E[:n_lvl], 'o', label='Numérico')
plt.plot(range(1, n_lvl+1), E_anal[:n_lvl], 'x', label='Analítico')
plt.xlabel("n")
plt.ylabel("Energía")
plt.title("Energías del oscilador armónico")
plt.legend()

    # (2) Funciones de onda
plt.subplot(1,2,2)
for i in range(n_lvl):
    plt.plot(x, psi[:,i] + E[i], label = f'n={i+1}')
    plt.plot(x, psi2[:,i] + E[i], label = f'n= {i+1}')
plt.xlabel("x")
plt.ylabel(r" $\psi_n(x)$ (desplazada) ")
plt.title("Funciones de onda")
plt.legend()

plt.tight_layout()
plt.show()

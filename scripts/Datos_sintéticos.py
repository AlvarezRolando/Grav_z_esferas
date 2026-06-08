import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. Configuración de calidad de publicación
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['savefig.dpi'] = 300 # Alta resolución


#Constantes de la ecuación
alpha = 1*10**5 #Constante convertir a mGales
gamma = (6.67e-11)  #Constante de gravitación universal
k = alpha*4*np.pi*gamma/3

datox = np.linspace(-2500,2500,251)

#Radio de la esfera:    150 y 200
#Contraste de densidad: 1200 y 800
#Profundidad:            300 y 500
#Centro de la esfera :  -500 y 750

anomalia_sintetica = (k * 150**3 * 1200 * 300) / ( ((datox-(-500))**2 + 300**2 )**(3/2)) + \
                     (k * 200**3 * 800 * 500) /  ( ((datox- (750))**2 + 500**2 )**(3/2)) 

#plt.plot (datox,f2)
datos ={
    'X': datox,
    'Anomalia': anomalia_sintetica
}

datos = pd.DataFrame(datos)
datos.to_csv('datos/anomalía_grav_2_esf.csv', index=False)


fig, ax = plt.subplots(figsize=(10, 6)) # Tamaño en pulgadas
ax.plot(datox, anomalia_sintetica, label='', color='black', linewidth=1.5)
ax.set_title('Perfil de anomalía gravimétrica (Dos esferas)')
ax.set_xlabel('Posición [m]')
ax.set_ylabel('Anomalía [mGal]')
ax.grid(True, linestyle='--', linewidth=0.5)
plt.savefig('img/Perfil_gravimétrico_2_esf.png', bbox_inches='tight')
plt.show()


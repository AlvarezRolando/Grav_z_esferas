"""
Inversión Gravimétrica mediante Optimización por Enjambre de Partículas (PSO)

==============================================================================
Objetivo:
    Estimar simultáneamente los parámetros geométricos y físicos de dos
    esferas enterradas a partir de una anomalía gravimétrica compuesta
    observada en superficie, minimizando el error cuadrático entre el modelo
    teórico y los datos de campo.

Método:
    Particle Swarm Optimization (Kennedy & Eberhart, 1995).
    Función de anomalía: Telford et al. (1990) — Applied Geophysics, Cap. 6.

Parámetros estimados (8 en total — 4 por esfera):
    R₁, R₂   — radio de cada esfera [m]
    ρ₁, ρ₂   — contraste de densidad de cada esfera [kg/m³]
    Z₁, Z₂   — profundidad del centro de cada esfera [m]
    Xc₁, Xc₂ — posición lateral del centro de cada esfera [m]


Autor: Rolando Álvarez Gómez
Fecha: ultima actualización 2024
La idea central de este trabjo surgió de la notas del curso de Inversión de Datos Geofísicos del semestre 2022-1 
a cargo del Dr. Mauricio Nava Flores.

UNIVERSIDAD NACIONAL AUTÓNOMA DE MÉXICO
FACULTAD DE INGENIERÍA
DIVISIÓN DE INGENIERÍA EN CIENCIAS DE LA TIERRA
DEPARTAMENTO DE GEOFÍSICA
"""

# =============================================================================
# BIBLIOTECAS
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# CONFIGURACIÓN GLOBAL DE GRÁFICAS
# Estilo de gráficas tipografía serif, alta resolución y ejes
# =============================================================================

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman"],
    "font.size":          12,
    "axes.titlesize":     13,
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "legend.framealpha":  0.9,
    "legend.fontsize":    10,
    "figure.dpi":         150,
    "savefig.dpi":        300,
})

# Paleta de colores coherente para todas las figuras
# Esfera 1 → azul  |  Esfera 2 → coral  |  Combinado → negro
COLORES = {
    "datos":        "#2171B5",   # azul     — datos observados
    "modelo_ini":   "#A8A8A8",   # gris     — modelo inicial combinado
    "modelo_fin":   "#222222",   # negro    — modelo final combinado
    "esfera_1":     "#2171B5",   # azul     — contribución de la esfera 1
    "esfera_2":     "#D85A30",   # coral    — contribución de la esfera 2
    "residuales":   "#E24B4A",   # rojo     — residuales / error
    "convergencia": "#2171B5",   # azul     — curva de convergencia
    # Un color por esfera para el panel de evolución de parámetros
    "params_s1":    "#2171B5",   # azul     — trayectorias de la esfera 1
    "params_s2":    "#D85A30",   # coral    — trayectorias de la esfera 2
}

IMG_DIR = Path("img") #Carpeta que almacena las imágenes
IMG_DIR.mkdir(exist_ok=True)


# =============================================================================
# FUNCIÓN DE ESTILO COMÚN
# Aplica cuadrícula y márgenes uniformes a cualquier eje
# =============================================================================

def _aplicar_estilo(ax: plt.Axes, grid: bool = True) -> None:
    """Aplica estilo  uniforme a un objeto Axes de Matplotlib."""
    if grid:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, color="gray")
    ax.tick_params(direction="out", length=4)


# =============================================================================
# CARGA DE DATOS
# =============================================================================

datos = pd.read_csv("datos/anomalía_grav_2_esf.csv")
X = datos.iloc[:, 0].to_numpy()   # Coordenada X del perfil [m]
d = datos.iloc[:, 1].to_numpy()   # Anomalía gravimétrica observada [mGal]


# =============================================================================
# MODELO FÍSICO
# Anomalía gravimétrica de una o dos esferas homogéneas enterradas
# Telford et al. (1990), ecuación 6.33
# =============================================================================

# Constantes físicas
GAMMA = 6.67e-11          # Constante de gravitación universal [m³/(kg·s²)]
ALPHA = 1e5               # Factor de conversión a mGal
K     = ALPHA * (4 * np.pi * GAMMA) / 3


def anomalia_esfera(
    x_obs: np.ndarray,
    R:     float,
    rho:   float,
    Z:     float,
    Xc:    float,
    k:     float = K,
) -> np.ndarray:
    """
    Anomalía gravimétrica vertical de una esfera enterrada.

    Parámetros
    ----------
    x_obs : Posiciones de observación en superficie [m]
    R     : Radio de la esfera [m]
    rho   : Contraste de densidad [kg/m³]
    Z     : Profundidad del centro de la esfera [m]
    Xc    : Posición lateral del centro [m]
    k     : Constante de escala (por defecto: K global)

    Retorna
    -------
    Anomalía gravimétrica [mGal] de tipo np.ndarray
    """
    return (k * R**3 * rho * Z) / (((x_obs - Xc)**2 + Z**2) ** (3 / 2))


def anomalia_dos_esferas(
    x_obs: np.ndarray,
    R1:    float, rho1: float, Z1: float, Xc1: float,
    R2:    float, rho2: float, Z2: float, Xc2: float,
    k:     float = K,
) -> np.ndarray:
    """
    Anomalía gravimétrica compuesta de dos esferas enterradas.

    El principio de superposición permite sumar las contribuciones
    individuales de cada cuerpo de forma lineal.

    Parámetros
    ----------
    x_obs       : Posiciones de observación en superficie [m]
    R1, rho1, Z1, Xc1 : Parámetros geométricos y físicos de la esfera 1
    R2, rho2, Z2, Xc2 : Parámetros geométricos y físicos de la esfera 2
    k           : Constante de escala (por defecto: K global)

    Retorna
    -------
    Anomalía gravimétrica combinada [mGal]
    """
    return (
        anomalia_esfera(x_obs, R1, rho1, Z1, Xc1, k)
        + anomalia_esfera(x_obs, R2, rho2, Z2, Xc2, k)
        #Propiedad de los campos potenciales
    )



def aplicar_limites(
    enjambre: np.ndarray,
    vel:      np.ndarray,
    p_min:    np.ndarray,
    p_max:    np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aplica condiciones de frontera  al enjambre de partículas.

    Cuando una partícula supera un límite del espacio de búsqueda, su
    posición se refleja simétricamente hacia el interior y su velocidad
    se invierte y amortigua, evitando que se salga del espacio de búsqueda
    y que se quede "atrapado" en los límites del espacio de búsqueda.

    Parámetros
    ----------
    enjambre : Matriz de posiciones actuales (N × M)
    vel      : Matriz de velocidades actuales (N × M)
    p_min    : Vector de límites inferiores (M,)
    p_max    : Vector de límites superiores (M,)

    Retorna
    -------
    enjambre, vel : Matrices corregidas (N × M)
    """
    # ── Límite inferior ────────────────────────────────────────────────
    mascara_min = enjambre < p_min # Comparación lógica de todos lo valores del enjambre
    enjambre    = np.where(mascara_min, 2 * p_min - enjambre, enjambre) # Refleja la partícula
    vel         = np.where(mascara_min, - FACTOR_REBOTE * vel, vel)   # Invierte y amortigua

    # ── Límite superior ────────────────────────────────────────────────
    mascara_max = enjambre > p_max
    enjambre    = np.where(mascara_max, 2 * p_max - enjambre, enjambre)
    vel         = np.where(mascara_max, - FACTOR_REBOTE * vel, vel)

    # Si la reflexión saca la partícula del otro lado (velocidad
    # muy alta), se hace clip sin reflexión y la velocidad se anula
    enjambre = np.clip(enjambre, p_min, p_max)

    return enjambre, vel

# =============================================================================
# ESPACIO DE BÚSQUEDA
# Rangos físicamente razonables para cada parámetro (aquí se usa la información a priori).
# Los límites de ambas esferas son idénticos; pueden diferenciarse si se
# dispone de información geológica que los restrinja independientemente.
# Orden: [R1, rho1, Z1, Xc1, R2, rho2, Z2, Xc2]
# =============================================================================

                  #R1      Rho1     Z1     Xc1      R2      Rho2      Z2      Xc2
P_MIN = np.array([  100,    250,    110,    -1000,  100,    250,     110,      0])
P_MAX = np.array([  500,    2000,   500,      0,   500,    2000,    500,   1000])

NOMBRES_PARAMS = [
    # Esfera 1
    "R₁  Radio esfera 1  [m]",
    "ρ₁  Contraste dens. 1  [kg/m³]",
    "Z₁  Profundidad 1  [m]",
    "Xc₁ Posición lateral 1  [m]",
    # Esfera 2
    "R₂  Radio esfera 2  [m]",
    "ρ₂  Contraste dens. 2  [kg/m³]",
    "Z₂  Profundidad 2  [m]",
    "Xc₂ Posición lateral 2  [m]",
]


# =============================================================================
# HIPERPARÁMETROS DEL ALGORITMO PSO
# =============================================================================

M      = 8     # Número de parámetros del modelo (4 por esfera × 2 esferas)
N      = 100   # Número de partículas — aumentado respecto al caso de 1 esfera
               # para mantener cobertura en un espacio de búsqueda de 8D
W      = 0.65   # Factor de inercia — controla la continuidad del movimiento
AL     = 1.33   # Coeficiente cognitivo (atracción hacia el mejor local)
AG     = 1.77   # Coeficiente social    (atracción hacia el mejor global)
NITER  = 300   # Número máximo de iteraciones 
TOL    = 1e-6  # Tolerancia de convergencia
PACIENCIA = 80 # Iteraciones sin mejora antes de detención anticipada
FACTOR_REBOTE = 0.5   #Factor de amortiguamiento, para evitar salirse del espacio de búsqueda 

# =============================================================================
# INICIALIZACIÓN DEL ENJAMBRE
# Distribución uniforme aleatoria dentro del espacio de búsqueda
# =============================================================================

SEED = 14121996
rng  = np.random.default_rng(seed=SEED)   # Semilla usada, generador reproducible 

# Posiciones y velocidades iniciales
enjambre_actual = rng.uniform(P_MIN, P_MAX, size=(N, M))  # posición actual
enjambre_pbest  = enjambre_actual.copy()                   # mejor posición personal
vel             = np.zeros((N, M))                         # velocidades iniciales

# Registro del óptimo global y del modelo inicial (partícula 0 como referencia)
modelo_inicial = enjambre_actual[0, :].copy()
particula_opt  = np.zeros(M)
min_global     = np.inf   # np.inf garantiza que cualquier error inicial sea aceptado, auqnue este sea muy grande

# Errores personales de cada partícula
errores_pbest = np.full(N, np.inf)

# Historial de convergencia y evolución de parámetros
historial_optimos = np.zeros(NITER)
historial_params  = np.zeros((NITER, M))

print("=" * 60)
print("  INVERSIÓN PSO — ANOMALÍA GRAVIMÉTRICA (2 ESFERAS)")
print("=" * 60)
print(f"  Partículas : {N}   |   Iteraciones máx. : {NITER}")
print(f"  Parámetros : {M}   |   Tolerancia       : {TOL:.0e}")
print(f"  Modelo inicial — Esfera 1: {modelo_inicial[:4].round(2)}")
print(f"  Modelo inicial — Esfera 2: {modelo_inicial[4:].round(2)}")
print("=" * 60)


# =============================================================================
# ALGORITMO PSO
# =============================================================================

sin_mejora = 0

for it in range(NITER):

    # ----------------------------------------------------------------
    # Evaluación de la función objetivo (error cuadrático total)
    # ----------------------------------------------------------------
    for j in range(N):
        residuo      = d - anomalia_dos_esferas(X, *enjambre_actual[j, :]) #Nota se desempaquietan y se mandan de forma indivdual con "*"
        error_actual = np.sum(residuo ** 2)

        # Penalización por incoherencia física en cada esfera:
        # R > Z implica que la esfera emerge a superficie (geológicamente inviable)
        if enjambre_actual[j, 0] > enjambre_actual[j, 2]:   # R1 > Z1
            error_actual *= 1.15 # Se penaliza con un %
        if enjambre_actual[j, 4] > enjambre_actual[j, 6]:   # R2 > Z2
            error_actual *= 1.15

        # ── Actualización del optimo local ──────────────────────────
        if error_actual < errores_pbest[j]:
            enjambre_pbest[j, :] = enjambre_actual[j, :].copy()
            errores_pbest[j]     = error_actual

        # ── Actualización del optimo global  ─────────────────────────
        # Se evalúa siempre, incluso si también se actualizó el mejor personal
        if error_actual < min_global:
            particula_opt = enjambre_actual[j, :].copy()
            min_global    = error_actual

    # ----------------------------------------------------------------
    # Actualización de velocidades (ecuación canónica PSO)
    # v_{t+1} = w·v_t + c1·r1·(pbest - x_t) + c2·r2·(gbest - x_t)
    # ----------------------------------------------------------------
    r1 = rng.random((N, M))
    r2 = rng.random((N, M))

    vel = (
        W  * vel
        + AL * r1 * (enjambre_pbest - enjambre_actual)
        + AG * r2 * (particula_opt  - enjambre_actual)
    )

    # ----------------------------------------------------------------
    # Desplazamiento: x_{t+1} = x_t + v_{t+1}  (posición actual)
    # ----------------------------------------------------------------
    enjambre_actual = enjambre_actual + vel
    
     # Verificar y corregir límites antes de evaluar la siguiente iteración
    enjambre_actual, vel = aplicar_limites(enjambre_actual, vel, P_MIN, P_MAX)

    # Registro del historial
    historial_optimos[it]   = min_global
    historial_params[it, :] = particula_opt # Va almacenando la "mejor partícula" en cada etapa del enjambre

    # ----------------------------------------------------------------
    # Criterio de parada por convergencia
    # ----------------------------------------------------------------
    if it > 0:
        mejora = historial_optimos[it - 1] - min_global
        if mejora < TOL:
            sin_mejora += 1
            if sin_mejora >= PACIENCIA:
                print(f"\n  Convergencia alcanzada en la iteración {it + 1}.")
                historial_optimos = historial_optimos[:it + 1]
                historial_params  = historial_params[:it + 1, :]
                break
        else:
            sin_mejora = 0


# =============================================================================
# RESULTADOS
# =============================================================================

n_iters_reales = len(historial_optimos)

# Separar parámetros por esfera para mayor claridad
opt_s1 = particula_opt[:4]   # R1, rho1, Z1, Xc1
opt_s2 = particula_opt[4:]   # R2, rho2, Z2, Xc2
ini_s1 = modelo_inicial[:4]
ini_s2 = modelo_inicial[4:]

print("\n" + "=" * 60)
print("  RESULTADOS FINALES")
print("=" * 60)
print("  ESFERA 1:")
for nombre, valor_ini, valor_fin in zip(NOMBRES_PARAMS[:4], ini_s1, opt_s1):
    print(f"    {nombre:<34}  ini: {valor_ini:>8.2f}   fin: {valor_fin:>8.2f}")
print("  ESFERA 2:")
for nombre, valor_ini, valor_fin in zip(NOMBRES_PARAMS[4:], ini_s2, opt_s2):
    print(f"    {nombre:<34}  ini: {valor_ini:>8.2f}   fin: {valor_fin:>8.2f}")

print(f"\n  Error cuadrático final : {min_global:.6e}")
print(f"  Iteraciones ejecutadas : {n_iters_reales}")

# Advertencias de incoherencia física
for idx, (R, Z, label) in enumerate(
    [(opt_s1[0], opt_s1[2], "Esfera 1"), (opt_s2[0], opt_s2[2], "Esfera 2")]
):
    if R > Z:
        print(f"\n  ADVERTENCIA ({label}): R > Z — la esfera emergería a superficie.")

print("=" * 60)

# Evaluación del modelo final sobre el perfil completo
modelo_final_vals = anomalia_dos_esferas(X, *particula_opt)
modelo_inic_vals  = anomalia_dos_esferas(X, *modelo_inicial)
contrib_s1        = anomalia_esfera(X, *opt_s1)   # contribución individual esfera 1
contrib_s2        = anomalia_esfera(X, *opt_s2)   # contribución individual esfera 2
residuales_final  = d - modelo_final_vals


# =============================================================================
# VISUALIZACIÓN 1: Ajuste del modelo compuesto a los datos observados
# Muestra la evolución desde el modelo inicial hasta el modelo final combinado
# =============================================================================

fig1, ax1 = plt.subplots(figsize=(10, 5))

ax1.scatter(X, d,
            label="Datos observados", color=COLORES["datos"],
            s=15, zorder=4, linewidths=0)
ax1.plot(X, modelo_inic_vals,
         label="Modelo inicial (combinado)", color=COLORES["modelo_ini"],
         linewidth=1.2, linestyle="--")
ax1.plot(X, modelo_final_vals,
         label="Modelo final PSO (combinado)", color=COLORES["modelo_fin"],
         linewidth=1.8)
ax1.plot(X, residuales_final,
         label="Residuales", color=COLORES["residuales"],
         linewidth=1.0, linestyle=":")
ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")

ax1.set_title("Ajuste del modelo de dos esferas a la anomalía gravimétrica observada")
ax1.set_xlabel("Distancia a lo largo del perfil [m]")
ax1.set_ylabel("Anomalía gravimétrica [mGal]")
ax1.legend(loc="upper right")
_aplicar_estilo(ax1)

fig1.tight_layout()
fig1.savefig(IMG_DIR / "01_ajuste_exploracion.png", bbox_inches="tight")


# =============================================================================
# VISUALIZACIÓN 2: Modelo final con contribuciones individuales y residuales
# Descomposición de la anomalía compuesta en las dos fuentes esféricas
# =============================================================================

fig2, ax2 = plt.subplots(figsize=(10, 5))

ax2.scatter(X, d,
            label="Datos observados", color=COLORES["datos"],
            s=15, zorder=5, linewidths=0)
ax2.plot(X, modelo_final_vals,
         label="Modelo PSO final (combinado)", color=COLORES["modelo_fin"],
         linewidth=2.0, zorder=4)
ax2.plot(X, contrib_s1,
         label="Contribución esfera 1", color=COLORES["esfera_1"],
         linewidth=1.2, linestyle="--", zorder=3)
ax2.plot(X, contrib_s2,
         label="Contribución esfera 2", color=COLORES["esfera_2"],
         linewidth=1.2, linestyle="--", zorder=3)
ax2.fill_between(X, residuales_final, 0,
                 alpha=0.20, color=COLORES["residuales"], label="Residuales")
ax2.plot(X, residuales_final,
         color=COLORES["residuales"], linewidth=0.8, zorder=2)
ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")

# Anotación de parámetros estimado para cada esfera
txt_s1 = (
    f"Esfera 1\n"
    f"$R_1$ = {opt_s1[0]:.1f} m\n"
    f"$\\rho_1$ = {opt_s1[1]:.1f} kg/m³\n"
    f"$Z_1$ = {opt_s1[2]:.1f} m\n"
    f"$X{{c_1}}$ = {opt_s1[3]:.1f} m"
)
txt_s2 = (
    f"Esfera 2\n"
    f"$R_2$ = {opt_s2[0]:.1f} m\n"
    f"$\\rho_2$ = {opt_s2[1]:.1f} kg/m³\n"
    f"$Z_2$ = {opt_s2[2]:.1f} m\n"
    f"$X{{c_2}}$ = {opt_s2[3]:.1f} m"
)
caja = dict(boxstyle="round,pad=0.4", alpha=0.88, ec="lightgray")
ax2.text(0.02, 0.97, txt_s1, transform=ax2.transAxes, va="top", ha="left",
         fontsize=9, family="monospace",
         bbox={**caja, "fc": "#DDEEFF"})
ax2.text(0.18, 0.97, txt_s2, transform=ax2.transAxes, va="top", ha="left",
         fontsize=9, family="monospace",
         bbox={**caja, "fc": "#FFE8DE"})

ax2.set_title("Modelo gravimétrico final — descomposición en dos fuentes esféricas")
ax2.set_xlabel("Distancia a lo largo del perfil [m]")
ax2.set_ylabel("Anomalía gravimétrica [mGal]")
ax2.legend(loc="upper right", ncol=2)
_aplicar_estilo(ax2)

fig2.tight_layout()
fig2.savefig(IMG_DIR / "02_modelo_final.png", bbox_inches="tight")


# =============================================================================
# VISUALIZACIÓN 3: Curva de convergencia del algoritmo
# Muestra la reducción del error cuadrático a lo largo de las iteraciones
# =============================================================================

fig3, ax3 = plt.subplots(figsize=(10, 4))

iteraciones = np.arange(1, n_iters_reales + 1)
ax3.semilogy(iteraciones, historial_optimos,
             color=COLORES["convergencia"], linewidth=1.6)
ax3.fill_between(iteraciones, historial_optimos,
                 historial_optimos.max(),
                 alpha=0.08, color=COLORES["convergencia"])

ax3.set_title("Curva de convergencia del PSO — evolución del error cuadrático mínimo")
ax3.set_xlabel("Iteración")
ax3.set_ylabel("Error cuadrático mínimo  (escala logarítmica)")
_aplicar_estilo(ax3)

fig3.tight_layout()
fig3.savefig(IMG_DIR / "03_convergencia.png", bbox_inches="tight")


# =============================================================================
# VISUALIZACIÓN 4: Evolución de los parámetros estimados por iteración
# Panel de 2×4 — fila superior: esfera 1 | fila inferior: esfera 2
# =============================================================================

fig4, axs = plt.subplots(2, 4, figsize=(14, 7), sharex=True)
fig4.suptitle(
    "Evolución de los parámetros estimados durante la optimización PSO",
    fontsize=13, y=1.01
)

# Etiquetas cortas para los subtítulos del panel
ETIQUETAS_PANEL = [
    "Radio $R_1$ [m]", "Densidad $\\rho_1$ [kg/m³]",
    "Profundidad $Z_1$ [m]", "Posición $X_{c_1}$ [m]",
    "Radio $R_2$ [m]", "Densidad $\\rho_2$ [kg/m³]",
    "Profundidad $Z_2$ [m]", "Posición $X_{c_2}$ [m]",
]

for i, ax in enumerate(axs.flat):
    # Esfera 1 (fila 0, índices 0-3) → azul | Esfera 2 (fila 1, índices 4-7) → coral
    color = COLORES["params_s1"] if i < 4 else COLORES["params_s2"]

    ax.plot(iteraciones, historial_params[:, i],
            color=color, linewidth=1.4)
    ax.axhline(particula_opt[i], linestyle="--", linewidth=0.8,
               color="gray", alpha=0.7, label=f"Final: {particula_opt[i]:.1f}")
    ax.set_title(ETIQUETAS_PANEL[i], fontsize=10)
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Valor estimado")
    ax.legend(fontsize=8, handlelength=1.5)
    _aplicar_estilo(ax)

# Etiquetas de fila para identificar cada esfera visualmente
fig4.text(0.005, 0.73, "Esfera 1", va="center", rotation="vertical",
          fontsize=11, color=COLORES["params_s1"], fontweight="bold")
fig4.text(0.005, 0.27, "Esfera 2", va="center", rotation="vertical",
          fontsize=11, color=COLORES["params_s2"], fontweight="bold")

fig4.tight_layout()
fig4.savefig(IMG_DIR / "04_evolucion_parametros.png", bbox_inches="tight")

plt.show()

print("\nGráficas guardadas en:", IMG_DIR.resolve())
#------------------------------------------------------------
# Codi per a la resolució numèrica de la NLSE normalitzada a primer
# ordre, gastant l'algorisme SSFM simètric (automodulació primer)
# per a qualsevol pols inicial.

# Fet per Víctor Camús Hernández
#------------------------------------------------------------

# Paquets i funcions importats:
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift, ifftshift

# Definició de funcions:
def sech(z):
    return 1 / np.cosh(z) # Secant hiperbòlica

# Paràmetres inicials del càlcul:
ModesFourierN = 1024 # Nombre de modes de Fourier (punts temporals)
PassosEspaiN = 1024 # Nomre de punts espacials
EspaiTotal = 5 * np.pi # Espai total considerat (unitats L_D)
TempsMàxim = 20 # Temps total considerat (unitats To)
Ordre = 1 # Ordre del solitó
Beta2 = -1 # Signe del paràmetre DVG
Graf3D = True

# Variables del càlcul:
DeltaEspai = EspaiTotal / PassosEspaiN # Amplada passos del càlcul
DeltaTemps = 2 * TempsMàxim / ModesFourierN # Precissió temporal
IndexosTemps = np.arange(-ModesFourierN / 2, ModesFourierN / 2, 1) # Índexos temporals
Tau = IndexosTemps * DeltaTemps # Vector temporal
Omega = IndexosTemps * np.pi / TempsMàxim # Freqüències

# Variables de la fibra:
Gamma = 0.03
delta3 = 0.07

# Funció inicial injectada:
FuncioInicial = sech(Tau)

# Preparació al bucle:
print(f"Càlcul numèric: bucle de {PassosEspaiN} passos.")
VectorFuncions = np.zeros((PassosEspaiN + 1, ModesFourierN),\
 dtype=np.cfloat) # Vector a omplir amb el resultat a cada pas
VectorFuncions[0] = FuncioInicial
Zeta = np.zeros(PassosEspaiN + 1) # Vector a omplir amb les posicions

# --------------| Començament del BUCLE de càlcul |--------------
# esquema: 1/2N -> D -> 1/2N; primer mig pas efecte no lineal (SFM)
for m in range(PassosEspaiN):
    Funcio = VectorFuncions[m]
    # Automodulació per a mig avanç a l'espai físic
    Funcio = np.exp(-DeltaEspai * 0.5j * (abs(Funcio) * Ordre)** 2) \
     * Funcio
    # Transformada de Fourier (centrada en freqüència zero)
    Funcio = fftshift(fft(Funcio))
    # Càlcul de la dispersió a l'espai de freqüències
    Funcio = np.exp(-DeltaEspai * (Beta2 *  0.5j * Omega ** 2 +\
     0.5 * Gamma + 1j* delta3 * Omega ** 3)) * Funcio
    # Retorn a l'espai físic
    Funcio = ifft(fftshift(Funcio))
    # Resta d'avanç amb l'automodulació
    Funcio = np.exp(-DeltaEspai * 0.5j * (abs(Funcio) * Ordre)** 2) \
     * Funcio
    # Comptador espacial
    Zeta[m + 1] = (m + 1) * DeltaEspai
    VectorFuncions[m + 1] =  Funcio
    if m % round(PassosEspaiN / np.log2(PassosEspaiN)) == 0:
        print(f"{m}/{PassosEspaiN}") # Marcador de progrés
# ----------------| Final del BUCLE de càlcul |----------------
print("Càlcul finalitzat satisfactòriament.")

# Representació de resultats:
fig, ax = plt.subplots(constrained_layout=True)
idx1 = round(1 * ModesFourierN / 4)
idx2 = round(3 * ModesFourierN / 4)
idx3 = round(PassosEspaiN / 2)
ax.plot(Tau[idx1:idx2], abs(FuncioInicial[idx1:idx2]), '-',\
 color='r', label=r"Pols inicial ($\xi = 0$)")
ax.plot(Tau[idx1:idx2], abs(VectorFuncions[idx3,idx1:idx2]), '--',\
 color='y', label=r"Pols a mitja dispersió ($\xi = \frac{5\pi}{2}$)")
ax.plot(Tau[idx1:idx2], abs(VectorFuncions[-1,idx1:idx2]), '-.',\
 color='b', label=r"Pols final ($\xi = 5 \pi$)")
ax.grid()
ax.legend(loc="best")
plt.xlabel(r'$\tau$')
plt.ylabel(r'|$u$|')
plt.title(r'Solitó amb TOD i Atenuació ($\delta_3 = 0.07, \theta = 0.03$)')
if Graf3D: # Representació 3D
    fig2 = plt.figure(constrained_layout=True)
    ax2 = fig2.add_subplot(1, 1, 1, projection='3d')
    TAU, ZETA = np.meshgrid(Tau, Zeta)
    ax2.set(xlabel=r'$\tau$', ylabel=r'$\xi$',
        zlabel=r'|$u$|')
    ax2.plot_surface(TAU, ZETA, np.abs(VectorFuncions),\
        cmap=plt.get_cmap('jet'), linewidth=0, antialiased=True,
        alpha=0.7)
    plt.title(r'Solitó amb TOD i Atenuació ($\delta_3 = 0.07, \theta = 0.03$) Representació 3D')
    plt.show()

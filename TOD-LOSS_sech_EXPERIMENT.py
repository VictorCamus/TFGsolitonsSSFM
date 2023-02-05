#------------------------------------------------------------
# Codi per a la resolució numèrica de la NLSE amb atenuació,
# gastant l'algorisme SSFM simètric (automodulació primer)
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
ModesFourierN = 2**11 # Nombre de modes de Fourier (punts temporals)
PassosEspaiN = 2**11 # Nomre de punts espacials
EspaiTotal = 0.7 # Espai total considerat (km)
TempsMàxim = 60 # Temps total considerat (ps)
Graf3D = True

# Variables del càlcul:
DeltaEspai = EspaiTotal / PassosEspaiN # Amplada passos del càlcul
DeltaTemps = 2 * TempsMàxim / ModesFourierN # Precissió temporal
IndexosTemps = np.arange(-ModesFourierN / 2, ModesFourierN / 2, 1) # Índexos temporals
Tau = IndexosTemps * DeltaTemps # Vector temporal
Omega = IndexosTemps * np.pi / TempsMàxim # Freqüències

#Definició de constants
c = 3e05# Velocitat de la llum (nm / ps)

# Paràmetres inicials de la fibra:
D = 16 # Factor de dispersió (ps/nm/km)
Alfa_dB = 0.2 # Pèrdues de la fibra (dB/km)
n = 1.45 # Índex de refracció de la fibra
tau_0 = 6.1 # FWHM del pols (ps)
Pw = 5 #Potència del pic (W)
lo = 1550 # Longitud d'ona (nm)
n2 = 3.18e-02 # Índex de refracció de l'efecte Kerr (nm²/W)
Radi = 4650 # Radi de la fibra (nm)

# Variables de la fibra:
tau = tau_0 #/ (2 * np.arccosh(np.sqrt(2))) # Amplada del pic
Beta2 = -lo ** 2 * D / (2 * np.pi * c) # GVD (ps²/km)
A_ef = 1.5 * np.pi * Radi ** 2 # Àrea efectiva
Alfa = Alfa_dB / 4.343
Cnl = n2 * Pw / (A_ef * lo * 1e-12) # Coeficient no-linealitat

print(Beta2, Alfa, Cnl)

# Funció inicial injectada:
FuncioInicial =  np.sqrt(Pw) * sech(Tau / tau)

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
    Funcio = np.exp(-DeltaEspai * 0.5j * Cnl * abs(Funcio) ** 2) \
     * Funcio
    # Transformada de Fourier (centrada en freqüència zero)
    Funcio = fftshift(fft(Funcio))
    # Càlcul de la dispersió a l'espai de freqüències
    Funcio = np.exp(-DeltaEspai * (Beta2 *  0.5j * Omega ** 2 +\
     0.5 * Alfa)) * Funcio
    # Retorn a l'espai físic
    Funcio = ifft(fftshift(Funcio))
    # Resta d'avanç amb l'automodulació
    Funcio = np.exp(-DeltaEspai * 0.5j * Cnl * abs(Funcio) ** 2) \
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
ax.plot(Tau[idx1:idx2], abs(FuncioInicial[idx1:idx2]) ** 2, '-',\
 color='r', label=r"Pols inicial")
ax.plot(Tau[idx1:idx2], abs(VectorFuncions[-1,idx1:idx2]) ** 2, '-.',\
 color='b', label=r"Pols final")
ax.grid()
ax.legend(loc="best")
plt.xlabel(r'$t$ (ps)')
plt.ylabel(r'|$A$|$^{2}$ (W)')
plt.title(r'Simulació Fibra Experiment ($P_0 = 5.0$ W, $T_0 = 6.1$ ps)')
if Graf3D: # Representació 3D
    fig2 = plt.figure(constrained_layout=True)
    ax2 = fig2.add_subplot(1, 1, 1, projection='3d')
    TAU, ZETA = np.meshgrid(Tau, Zeta)
    ax2.set(xlabel=r'$t$ (ps)', ylabel=r'$z$ (km)',
        zlabel=r'|$A$|$^{2}$ (W)')
    ax2.plot_surface(TAU, ZETA, np.abs(VectorFuncions)**2,\
        cmap=plt.get_cmap('jet'), linewidth=0, antialiased=True,
        alpha=0.7)
    plt.title(r'Simulació Fibra Experiment Representació 3D')
    plt.show()

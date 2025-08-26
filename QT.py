import numpy as np  
import matplotlib.pyplot as plt  
  
# Constants  
hbar = 1.0545718e-34  # Reduced Planck's constant (J·s)  
m = 9.10938356e-31    # Electron mass (kg)  
E = 1.0e-19           # Energy of the particle (J)  
V0 = 1.5e-19          # Height of the potential barrier (J)  
width = 1e-10         # Width of the barrier (m)  
  
# Define the potential barrier  
def potential(x):  
    if 0 < x < width:  
        return V0  
    else:  
        return 0  
  
# Create an array of x values  
x = np.linspace(-5e-10, 5e-10, 1000)  
V = np.array([potential(i) for i in x])  
  
# Calculate the wave function (simplified)  
def wave_function(x):  
    k = np.sqrt(2 * m * (E - V)) / hbar  # Wave number  
    return np.exp(1j * k * x)  
  
# Calculate the wave function values  
psi = wave_function(x)  
  
# Plot the potential barrier and wave function  
plt.figure(figsize=(10, 5))  
plt.plot(x, V, label='Potential Barrier', color='blue')  
plt.plot(x, np.abs(psi)**2, label='Wave Function', color='red')  
plt.title('Quantum Tunneling Simulation')  
plt.xlabel('Position (m)')  
plt.ylabel('Potential / Wave Function Amplitude')  
plt.axhline(0, color='black', lw=0.5)  
plt.axvline(0, color='black', lw=0.5)  
plt.legend()  
plt.grid()  
plt.show()  

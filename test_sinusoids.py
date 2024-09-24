import numpy as np
import matplotlib.pyplot as plt
from signal_generator import SignalGenerator


signal_generator = SignalGenerator(x_min=-5, x_max=5)


levels = np.arange(-2, 2.1, 0.1)

signal_generator.add_sinusoid1(k=0.1, omega=0.5)
plt.figure()
contour = plt.contourf(
    signal_generator.T,
    signal_generator.X,
    signal_generator.components[0]["signal"],
    levels=levels,
    cmap='bwr',
    extend='both'
    )
plt.xlabel('Time')
plt.ylabel('Space')
plt.colorbar(contour)
plt.title('Signal 1')

signal_generator.add_sinusoid2(omega=1.5)
plt.figure()
contour = plt.contourf(
    signal_generator.T,
    signal_generator.X,
    signal_generator.components[1]["signal"],
    levels=levels,
    cmap='bwr',
    extend='both'
    )
plt.xlabel('Time')
plt.ylabel('Space')
plt.colorbar(contour)
plt.title('Signal 2')

# signal_generator.add_sinusoid3(omega=2.5)
# plt.figure()
# contour = plt.contourf(
#     signal_generator.T,
#     signal_generator.X,
#     signal_generator.components[2]["signal"],
#     levels=levels,
#     cmap='bwr',
#     extend='both'
#     )
# plt.xlabel('Time')
# plt.ylabel('Space')
# plt.colorbar(contour)
# plt.title('Signal 3')

signal_generator.add_noise(random_seed=42)
plt.figure()
contour = plt.contourf(
    signal_generator.T,
    signal_generator.X,
    signal_generator.signal,
    levels=levels,
    cmap='bwr',
    extend='both'
    )
plt.xlabel('Time')
plt.ylabel('Space')
plt.colorbar(contour)
plt.title('All + Noise')

plt.show()

from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np
import cmath

d = 1.e-3  # Thickness of the plate
fs = 1e3  # Frequency resolution
c_l = 6149.0  # Bulk longitudinal velocity
c_t = 3097.3  # Bulk transverse velocity

f_range = np.arange(fs, 1e6+fs, fs)
c = np.zeros(len(f_range), dtype='f')
c2 = np.zeros(len(f_range), dtype='f')
f = f_range[0]


def fun(x):
    k = 2 * cmath.pi * f / x
    p = cmath.sqrt((2 * cmath.pi * f)**2/c_l**2 - k**2)
    q = cmath.sqrt((2 * cmath.pi * f)**2/c_t**2 - k**2)
    return (cmath.tan(q * d / 2) / q + 4 * k**2 * p * cmath.tan(p * d / 2) / (k**2 - q**2)**2).real


def fun2(x):
    k = 2 * cmath.pi * f / x
    p = cmath.sqrt((2 * cmath.pi * f) ** 2 / c_l ** 2 - k ** 2)
    q = cmath.sqrt((2 * cmath.pi * f) ** 2 / c_t ** 2 - k ** 2)
    return (cmath.tan(q * d / 2) * q + (k**2 - q**2)**2 * cmath.tan(p * d / 2) / (4 * k**2 * p)).real


def c_g(x):
    diff = np.zeros(len(x), float)
    for k in range(len(x)-2):
        diff[k+1] = 0.5 * (x[k+2] - x[k]) / fs
    result = x**2 / (x - f_range * diff)
    result[0] = result[1]
    result[-1] = result[-2]
    return result


for i in range(len(f_range)):
    f = f_range[i]
    c[i] = fsolve(fun, np.array(5500.))
    c2[i] = fsolve(fun2, np.array(2000.))

c3 = c_g(c)
c4 = c_g(c2)

plt.rcParams['figure.figsize'] = [3.5, 2.5]
plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['font.size'] = 8
plt.rcParams['legend.fontsize'] = 'small'

plt.plot((f_range[1:-1]/1000).astype('int'), c[1:-1]/1000, c='#2166ac')
plt.plot((f_range[1:-1]/1000).astype('int'), c2[1:-1]/1000, c='#b2182b')
plt.plot((f_range[1:-1]/1000).astype('int'), c3[1:-1]/1000, '--', c='#2166ac')
plt.plot((f_range[1:-1]/1000).astype('int'), c4[1:-1]/1000, '--', c='#b2182b')
plt.legend([r'Phase velocity - $S_0$ mode', r'Phase velocity - $A_0$ mode',
            r'Group velocity - $S_0$ mode', r'Group velocity - $A_0$ mode'])
plt.xlabel('Frequency [kHz]')
plt.ylabel('Velocity [km/s]')
plt.grid()
plt.show()


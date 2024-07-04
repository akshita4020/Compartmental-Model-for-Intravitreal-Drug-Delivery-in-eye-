import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the system of ODEs
def system(t, y, kin, k12, k13, kb, kd, k21, k20, k31, k30):
    Cv, Cr, Cac = y
    dCv_dt = kin * (-4.709 * np.log(t) + 34) - (k12 + k13 + kb + kd) * Cv
    dCr_dt = k12 * Cv - (k21 + k20 + kd) * Cr
    dCac_dt = k13 * Cv - (k31 + k30 + kd) * Cac
    return [dCv_dt, dCr_dt, dCac_dt]

# Define parameters
kin = 1.0   # You can change this value based on your specific problem
k12 = 0.1   # You can change this value based on your specific problem
k13 = 0.1   # You can change this value based on your specific problem
kb = 0.1    # You can change this value based on your specific problem
kd = 0.1    # You can change this value based on your specific problem
k21 = 0.1   # You can change this value based on your specific problem
k20 = 0.1   # You can change this value based on your specific problem
k31 = 0.1   # You can change this value based on your specific problem
k30 = 0.1   # You can change this value based on your specific problem

# Initial conditions
Cv0 = 0.0
Cr0 = 0.0
Cac0 = 0.0

# Time span
t_span = (0.1, 100)  # start at 0.1 to avoid log(0) issue
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Solve the system of ODEs
solution = solve_ivp(system, t_span, [Cv0, Cr0, Cac0], args=(kin, k12, k13, kb, kd, k21, k20, k31, k30), t_eval=t_eval)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(solution.t, solution.y[0], label='Cv')
plt.plot(solution.t, solution.y[1], label='Cr')
plt.plot(solution.t, solution.y[2], label='Cac')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend()
plt.title('Concentration vs Time')
plt.grid()
plt.show()
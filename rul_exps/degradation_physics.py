import tensorflow as tf
import numpy as np

# XJTU-SY constants
C = 4000.0             # baseline load rating
p = 3.0                # fatigue exponent
beta = 1e-6            # coupling fatigue to EDV growth
phi = 1e-5             # EDV growth coefficient
q = 4.0                # EDV load exponent
Dm = 0.025             # ball diameter [m]

# Lubricant & thermal
k_B = 8.617e-5         # Boltzmann constant [eV/K]
alpha = 1e-5           # viscosity degradation coefficient
nu0 = 1e-5             # baseline lubricant viscosity
E_vis = 0.1            # viscosity activation energy (eV)
T0 = 298.0             # reference temperature [K]
k_o = 1e-4             # oxidation rate coefficient
O_max = 1.0            # max oxidation level

# Wear parameters
A_v = 1e-6            # Archard wear coefficient
H_hard = 1.5e9         # hardness of material
A_a = 1e-6             # roughness-induced wear coefficient
gamma_r = 1e-3         # roughness evolution from wear
delta_c = 1e-5         # roughness evolution from debris
rho = 1e6              # debris generation from wear

# Geometric correction
eta = 1e6              # saturation coefficient for volume
zeta = 1e12            # saturation coefficient for roughness

# Thermal & friction
gamma_w = 0.1           # weight of wear in degradation
zeta_L = 0.1            # weight of thermal effects
m_cp = 3.77e6           # mass * specific heat
mu_f = 0.005            # friction coefficient
h_A = 5.0               # heat transfer coefficient
T_a = 293.15            # ambient temperature
xi = 100.0              # exothermic contribution from oxidation


params = {
    'C_load': C,
    'p': p,
    'q': q,
    'Dm':Dm,
    'phi': phi,
    'beta': beta,
    'A_v': A_v,
    'H_hard': H_hard,
    'A_a': A_a,
    'gamma_r': gamma_r,
    'delta_c': delta_c,
    'eta': eta,
    'zeta': zeta,
    'k_o': k_o,
    'O_max': O_max,
    'E_a': 0.1,
    'nu0': nu0,
    'E_vis': E_vis,
    'T0': T0,
    'psi': 1e-6,
    'rho': rho,
    'gamma_w': gamma_w,
    'alpha': alpha,
    'zeta_L': zeta_L,
    'm_cp': m_cp,
    'mu_f': mu_f,
    'h_A': h_A,
    'T_a': T_a,
    'xi': xi
}



# Physics-based degradation model
@tf.function
def dDdt(y, Load, RPM, T):
    """
    Compute the derivatives of the degradation model.
    
    Args:
        y: Tensor of shape (6,) -> [V_d, V, R, O, C_debris, D]
        Load: scalar tensor or float
        RPM: scalar tensor or float
        T: scalar tensor or float (temperature)
    
    Returns:
        Tensor of shape (6,) with derivatives
    """
    V_d, V, R, O, C_debris, D = tf.unstack(y)


    # Angular velocity
    omega = 2.0 * np.pi * RPM / 60.0

    # Oxidation dynamics
    dO_dt = params['k_o'] * (params['O_max'] - O) * tf.exp(-params['E_a'] / (k_B * T))

    # Thermal dynamics
    dT_dt = (1.0 / params['m_cp']) * (
        params['mu_f'] * Load * omega
        - params['h_A'] * (T - params['T_a'])
        + params['xi'] * dO_dt
    )

    # Viscosity
    nu = params['nu0'] * tf.exp(-params['alpha'] * O - params['E_vis'] / (k_B * T) + params['E_vis'] / (k_B * params['T0']))

    # Sliding speed
    ds_dt = np.pi * params['Dm'] * (RPM / 60.0)

    # Geometric/debris corrections
    W_mod = 1.0 / (1.0 + params['eta'] * V + params['zeta'] * R**2)
    C_eff = params['C_load'] * (nu / params['nu0']) * W_mod * tf.exp(-params['psi'] * C_debris)

    # Fatigue and wear
    dVd_dt = params['phi'] * (Load / C_eff)**params['q'] * RPM
    dV_dt = (params['A_v'] * Load / params['H_hard']) * ds_dt + params['A_a'] * R
    dR_dt = params['gamma_r'] * dV_dt + params['delta_c'] * C_debris
    dCdebris_dt = params['rho'] * dV_dt
    kf = (Load / C_eff)**params['p'] * RPM / (60.0 * 1e6)

    # Total degradation
    dD_dt = kf + params['beta'] * dVd_dt + params['gamma_w'] * dV_dt + params['zeta_L'] * dT_dt

    return tf.stack([dVd_dt, dV_dt, dR_dt, dO_dt, dCdebris_dt, dD_dt])

"""
Ecuación (9) de Gabillon / Gibson-Schwartz — extendido con Heston (1993)

Modelo original (ec. 9):
    dS/S = mu dt + sigma1 dZ1
    dδ   = k(alpha - δ) dt + sigma2 dZ2
    dZ1 dZ2 = rho dt

Extension con volatilidad estocástica (Heston 1993, ec. 6.4a-b):
    dS/S = mu dt + sqrt(v_t) dZ1         <- sqrt(v_t) estocastica
    dδ   = k(alpha - δ) dt + sigma2 dZ2
    dv_t = kappa_v*(m - v_t) dt + xi*sqrt(v_t) dZ3   <- varianza del spot

Correlaciones:
    dZ1 dZ2 = rho dt       <- spot vs CY
    dZ1 dZ3 = rho_sv dt    <- spot vs vol
    dZ2 dZ3 = 0            <- CY independiente de la varianza del spot (para simplificar)
"""

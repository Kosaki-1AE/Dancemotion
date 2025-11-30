# inputs: dE_dt(t), dM_dt(t)  # それぞれ時系列
eps = 1e-6
Delta = (dE_dt - dM_dt) / (dE_dt + dM_dt + eps)
TCI   = 1 - np.abs(Delta)
zone_mask = TCI >= theta

# SQ 効き関数
def g(dt_sync, mu, sigma):
    return np.exp(- (dt_sync - mu)**2 / (2*sigma**2))

# PIで間を微調整
err = theta - TCI_t
dt_sync = clip(dt_sync + k_p*err + k_i*cum_err, dt_min, dt_max)

# 指標
tau_c     = np.sum(g(dt_sync_events, mu, sigma))
rho0      = zero_crossing_rate(Delta, around_zero=True)
t_zone    = contiguous_time(zone_mask)

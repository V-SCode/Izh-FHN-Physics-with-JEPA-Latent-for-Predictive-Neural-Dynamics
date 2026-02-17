class HybridIzhWorldModel(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        # learnable Izh parameters
        self.a = nn.Parameter(torch.tensor(0.02))
        self.b = nn.Parameter(torch.tensor(0.2))
        self.c = nn.Parameter(torch.tensor(-65.0))
        self.d = nn.Parameter(torch.tensor(8.0))
        self.I0 = nn.Parameter(torch.tensor(10.0))

        self.residual = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def f_phys(self, z):
        v, u = z[...,0], z[...,1]
        a = self.a; b = self.b; I0 = self.I0
        dv = 0.04*v**2 + 5*v + 140 - u + I0
        du = a*(b*v - u)
        return torch.stack([dv, du], dim=-1)

    def forward(self, z, dt):
        f = self.f_phys(z) + self.residual(z)
        z_next = z + dt * f

      return F.normalize(z_next, dim=-1)

# 1) make dataset
V_izh, L_izh, Fs_izh = make_izh_dataset(Fs=1000.0)

# 2) train AE
Tw_ms_izh = 128
delta_ms_izh = 20

enc_izh, dec_izh = train_autoencoder(
    V_izh, Fs_izh,
    Tw_ms=Tw_ms_izh,
    batch_size=64,
    steps=2000,
    lr=1e-3,
    d_latent=256,
)

# 3) train world model
wm_izh = train_world_model(
    V_izh, Fs_izh,
    enc_izh, dec_izh,
    Tw_ms=Tw_ms_izh,
    delta_ms=delta_ms_izh,
    batch_size=64,
    steps=2000,
    lr=1e-3,
)

# 4) reconstruction check
Tw_samp = int(round(Tw_ms_izh/1000.0 * Fs_izh))
t0 = 2000
x = V_izh[t0:t0+Tw_samp, :].T
X = torch.from_numpy(x[None,...]).float().to(device)
with torch.no_grad():
    z = enc_izh(X); x_hat = dec_izh(z).cpu().numpy()[0]

plt.figure(figsize=(8,3))
plt.plot(x[0], label="orig")
plt.plot(x_hat[0], label="recon", alpha=0.7)
plt.legend(); plt.title("Izhikevich AE reconstruction")
plt.tight_layout(); plt.show()

# 5) future prediction
x_ctx, x_true, x_pred = predict_future_trace(
    enc_izh, wm_izh, dec_izh,
    V_izh, Fs_izh,
    t0=t0,
    neuron_idx=0,
    Tw_ms=Tw_ms_izh,
    delta_ms=delta_ms_izh,
)

t, v, L = simulate_izh(T=3.0, dt=0.001, I_const=14.0)
plt.figure(figsize=(8,3))
plt.plot(t, v)
plt.xlabel("time (s)"); plt.ylabel("V (mV)")
plt.title("Izhikevich neuron (should spike)")
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,3))
plt.plot(x_true, label="true future")
plt.plot(x_pred, label="predicted future", alpha=0.7)
plt.legend(); plt.title("Izhikevich future prediction")
plt.tight_layout(); plt.show()


#Test runs on Izh spiking 

def simulate_izh(T=5.0, dt=0.001,
                 a=0.02, b=0.2, c=-65.0, d=8.0,
                 I_const=14.0,
                 v0=-65.0, u0=None):
 
    t_eval = np.arange(0, T, dt)
    v = np.zeros_like(t_eval)
    u = np.zeros_like(t_eval)

    v[0] = v0
    u[0] = b * v0 if u0 is None else u0

    for i in range(len(t_eval) - 1):
        # spike threshold + reset
        if v[i] >= 30.0:
            v[i] = 30.0
            v[i+1] = c
            u[i+1] = u[i] + d
            continue

        dv = 0.04*v[i]**2 + 5.0*v[i] + 140.0 - u[i] + I_const
        du = a*(b*v[i] - u[i])

        v[i+1] = v[i] + dt*dv
        u[i+1] = u[i] + dt*du

    L = np.stack([v, u], axis=-1)
    return t_eval, v, L

t_ex, v_ex, _ = simulate_izh(T=3.0, dt=0.001, I_const=18.0)
plt.figure(figsize=(8,3))
plt.plot(t_ex, v_ex)
plt.xlabel("time (s)"); plt.ylabel("V (mV)")
plt.title("Izhikevich neuron (example voltage trace)")
plt.tight_layout(); plt.show()

t_ex, v_ex, _ = simulate_izh(T=10.0, dt=0.001,
                             a=0.02, b=0.2, c=-65, d=8,
                             I_const=14.0)
plt.figure(figsize=(8,3))
plt.plot(t_ex, v_ex)
plt.xlabel("time (s)"); plt.ylabel("V (mV)")
plt.title("Izhikevich RS with constant I")
plt.tight_layout(); plt.show()

#Variation of pulse rate 

def make_step_current(t, baseline=0.0, pulse_amp=14.0, pulse_times=None, pulse_width=0.2):
    
    I = np.full_like(t, baseline, dtype=np.float32)
    if pulse_times is None:
        pulse_times = [0.5, 1.5, 2.5]  # example pulse onsets
    for t0 in pulse_times:
        I[(t >= t0) & (t < t0 + pulse_width)] += pulse_amp
    return I

def simulate_izh_time_varying(T=5.0, dt=0.001,
                              a=0.02, b=0.2, c=-65, d=8,
                              I_fn=None,
                              v0=-65.0, u0=None):
    t_eval = np.arange(0, T, dt)
    if I_fn is None:
        I_fn = lambda t: 10.0  # default constant

    v = np.zeros_like(t_eval)
    u = np.zeros_like(t_eval)
    v[0] = v0
    u[0] = b*v0 if u0 is None else u0

    for i in range(len(t_eval) - 1):
        if v[i] >= 30.0:
            v[i] = 30.0
            v[i+1] = c
            u[i+1] = u[i] + d
            continue

        I = I_fn(t_eval[i])
        dv = 0.04*v[i]**2 + 5.0*v[i] + 140.0 - u[i] + I
        du = a*(b*v[i] - u[i])
        v[i+1] = v[i] + dt*dv
        u[i+1] = u[i] + dt*du

    L = np.stack([v, u], axis=-1)
    return t_eval, v, L

t = np.arange(0, 5.0, 0.001)
I_trace = make_step_current(t, baseline=0.0, pulse_amp=14.0,
                            pulse_times=[0.5, 1.0, 1.5, 2.0, 3.0],
                            pulse_width=0.1)

t_ex, v_ex, _ = simulate_izh_time_varying(T=5.0, dt=0.001,
                                          a=0.02, b=0.2, c=-65, d=8,
                                          I_fn=lambda t_: np.interp(t_, t, I_trace))

plt.figure(figsize=(8,3))
plt.plot(t_ex, v_ex, label="V")
plt.twinx()
plt.plot(t, I_trace, 'r--', alpha=0.3, label="I(t)")
plt.title("Izhikevich neuron with pulsed input")
plt.tight_layout(); plt.show()

#Variation of current insertion 

def make_step_current(t, baseline=0.0, pulse_amp=14.0,
                      pulse_times=None, pulse_width=0.2):
   
    I = np.full_like(t, baseline, dtype=np.float32)
    if pulse_times is None:
        pulse_times = [0.5, 1.5, 2.5]  # example onsets
    for t0 in pulse_times:
        I[(t >= t0) & (t < t0 + pulse_width)] += pulse_amp
    return I

def simulate_izh_time_varying(T=5.0, dt=0.001,
                              a=0.02, b=0.2, c=-65.0, d=8.0,
                              I_fn=None,
                              v0=-65.0, u0=None):
    t_eval = np.arange(0, T, dt)
    if I_fn is None:
        I_fn = lambda t: 10.0  # default

    v = np.zeros_like(t_eval)
    u = np.zeros_like(t_eval)
    v[0] = v0
    u[0] = b*v0 if u0 is None else u0

    for i in range(len(t_eval) - 1):
        if v[i] >= 30.0:
            v[i] = 30.0
            v[i+1] = c
            u[i+1] = u[i] + d
            continue

        I = I_fn(t_eval[i])
        dv = 0.04*v[i]**2 + 5.0*v[i] + 140.0 - u[i] + I
        du = a*(b*v[i] - u[i])
        v[i+1] = v[i] + dt*dv
        u[i+1] = u[i] + dt*du

    L = np.stack([v, u], axis=-1)
    return t_eval, v, L

T = 10.0
dt = 0.001
t = np.arange(0, T, dt)

# 1) Choose a baseline that alone can give RS if you simulate with constant I
baseline_I = 12.0

# 2) Pulses added on top of baseline
pulse_amp   = 6.0         # extra current during pulse
pulse_times = [1.0, 2.5, 4.0, 6.0, 8.0]   # onset times (s)
pulse_width = 0.2         # each pulse lasts 200 ms

I_trace = make_step_current(
    t,
    baseline=baseline_I,
    pulse_amp=pulse_amp,
    pulse_times=pulse_times,
    pulse_width=pulse_width,
)

t_ex, v_ex, L_ex = simulate_izh_time_varying(
    T=T,
    dt=dt,
    a=0.02, b=0.2, c=-65.0, d=8.0,
    I_fn=lambda tau: np.interp(tau, t, I_trace),  # time-varying I(t)
)

plt.figure(figsize=(10,4))

ax1 = plt.gca()
ax1.plot(t_ex, v_ex, label="V(t)", color="C0")
ax1.set_xlabel("time (s)")
ax1.set_ylabel("V (mV)", color="C0")
ax1.tick_params(axis='y', labelcolor="C0")

ax2 = ax1.twinx()
ax2.plot(t, I_trace, 'r--', alpha=0.4, label="I(t)")
ax2.set_ylabel("Current I(t)", color="C1")
ax2.tick_params(axis='y', labelcolor="C1")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc="upper right")

plt.title("Izhikevich neuron: tonic RS + pulsed input")
plt.tight_layout()
plt.show()

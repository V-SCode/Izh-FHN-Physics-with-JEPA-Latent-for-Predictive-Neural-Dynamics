#JEPA.mat loader

def load_jepa_input(mat_path: str):
    m = loadmat(mat_path, squeeze_me=True, simplify_cells=True)

    if "V_TN" in m:
        V = m["V_TN"].astype(np.float32)
    elif "V_TN_sub" in m:
        V = m["V_TN_sub"].astype(np.float32)
    else:
        raise KeyError("Could not find 'V_TN' or 'V_TN_sub' in MAT file.")

    Fs_raw = float(m["Fs"]) if "Fs" in m else 1000.0
    Fs = 1000.0  

    if "state" in m:
        state = m["state"]
    elif "state_sub" in m:
        state = m["state_sub"]
    else:
        raise KeyError("Could not find 'state' or 'state_sub' in MAT file.")
    state = np.array(state, dtype=np.int64).ravel()

    if "t" in m:
        t = np.array(m["t"], dtype=np.float32).ravel()
    elif "t_sub" in m:
        t = np.array(m["t_sub"], dtype=np.float32).ravel()
    else:
        t = None

    print(f"Loaded {mat_path}")
    print(f"  V[T,N] = {V.shape}, Fs_raw(from file)={Fs_raw} Hz, USING Fs={Fs} Hz")
    print(f"  state len = {len(state)}")
    return V, Fs, state, t

#Future prediction

class FuturePairs(Dataset):
    def __init__(self, V, state, Tw, stride, deltas):
        assert V.ndim == 2
        self.V = V
        self.state = state
        self.T, self.N = V.shape
        self.Tw = int(Tw)
        self.stride = int(stride)
        self.deltas = list(map(int, deltas))
        self.index = []

        Tmargin = max(self.deltas) + self.Tw + 1
        if self.T <= Tmargin:
            raise ValueError(
                f"Not enough T={self.T} for Tw={self.Tw} and max delta={max(self.deltas)}"
            )

        for i in range(self.N):
            for t0 in range(0, self.T - self.Tw - max(self.deltas) - 1, self.stride):
                for d in self.deltas:
                    self.index.append((i, t0, d))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, t0, d = self.index[k]
        x_c = self.V[t0:t0+self.Tw, i]
        x_t = self.V[t0+d:t0+d+self.Tw, i]
        s   = np.bincount(self.state[t0:t0+self.Tw]).argmax()
        return (torch.from_numpy(x_c[None, :]).float(),
                torch.from_numpy(x_t[None, :]).float(),
                torch.tensor(int(s), dtype=torch.long),
                torch.tensor(int(i), dtype=torch.long))

  #Encoder, predictor, JEPA setup

class TimeEncoder(nn.Module):
    def __init__(self, in_ch=1, d=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, 7, padding=3), nn.GELU(),
            nn.Conv1d(64, 128, 5, padding=2, dilation=2), nn.GELU(),
            nn.Conv1d(128, 128, 5, padding=4, dilation=4), nn.GELU(),
            nn.Conv1d(128, 256, 3, padding=4, dilation=4), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(256, d)
        )
    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=-1)

class Predictor(nn.Module):
    def __init__(self, d=256, n_states=1, d_s=8):
        super().__init__()
        self.state_emb = nn.Embedding(max(1, n_states), d_s)
        self.mlp = nn.Sequential(
            nn.Linear(d + d_s, 2*d), nn.GELU(),
            nn.Linear(2*d, d)
        )
    def forward(self, zc, s):
        es = self.state_emb(s)
        out = self.mlp(torch.cat([zc, es], dim=-1))
        return F.normalize(out, dim=-1)

class VoltageDecoder(nn.Module):
    def __init__(self, d=256, Tw=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 512), nn.GELU(),
            nn.Linear(512, 1024), nn.GELU(),
            nn.Linear(1024, Tw)   # output is a flat [Tw] trace
        )
    def forward(self, z):
        return self.net(z)

@torch.no_grad()
def ema_update(teacher, student, m=0.996):
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.data.mul_(m).add_(sp.data, alpha=1.0 - m)

def vicreg_var_cov(z, gamma=1.0):
    std = z.std(dim=0) + 1e-4
    var_loss = F.relu(gamma - std).mean()
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / (z.shape[0] - 1)
    cov = cov.fill_diagonal_(0.0)
    cov_loss = (cov ** 2).sum() / z.shape[1]
    return var_loss, cov_loss

def train_jepa(
    V, state, Fs,
    Tw_ms=512,
    stride_ms=None,
    deltas_ms=(25, 100, 250),
    steps=1500,
    batch_size=128,
    lr=1e-3,
    wd=5e-2,
    ema_m=0.996,
    seed=0,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    Tw = int(round((Tw_ms/1000.0) * Fs))
    stride = int(round(((stride_ms if stride_ms is not None else Tw_ms/2)/1000.0) * Fs))
    deltas = [max(1, int(round((ms/1000.0) * Fs))) for ms in deltas_ms]

    print(f"  Training JEPA: Tw={Tw} (~{Tw_ms} ms), stride={stride}, deltas={deltas}")

    ds = FuturePairs(V, state, Tw=Tw, stride=stride, deltas=deltas)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    student = TimeEncoder(in_ch=1, d=256).to(device)
    teacher = TimeEncoder(in_ch=1, d=256).to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad_(False)

    n_states = int(state.max()) + 1 if state.size > 0 else 1
    pred = Predictor(d=256, n_states=n_states, d_s=8).to(device)

    opt = torch.optim.AdamW(
        list(student.parameters()) + list(pred.parameters()),
        lr=lr, weight_decay=wd
    )

    student.train(); pred.train()
    it = iter(dl)

    for step in range(1, steps+1):
        try:
            Xc, Xt, St, _ = next(it)
        except StopIteration:
            it = iter(dl)
            Xc, Xt, St, _ = next(it)

        Xc, Xt, St = Xc.to(device), Xt.to(device), St.to(device)

        zc = student(Xc)
        with torch.no_grad():
            zt = teacher(Xt)
        zhat = pred(zc, St)

        pred_loss = ((zhat - zt)**2).sum(dim=-1).mean()
        v1, c1 = vicreg_var_cov(zc)
        v2, c2 = vicreg_var_cov(zt)
        loss = pred_loss + 0.25*(v1+v2) + 0.005*(c1+c2)

        opt.zero_grad(); loss.backward(); opt.step()
        ema_update(teacher, student, m=ema_m)

        if step % 200 == 0 or step == 1:
            print(f"    step {step}/{steps} | loss={loss.item():.5f}")

    student.eval(); teacher.eval(); pred.eval()
    return student, teacher, pred

@torch.no_grad()
def embed_all_windows(encoder, V, Fs, Tw_ms=512, stride_ms=None):
    encoder = encoder.to(device).eval()
    T, N = V.shape
    Tw = int(round((Tw_ms/1000.0) * Fs))
    stride = int(round(((stride_ms if stride_ms is not None else Tw_ms/2)/1000.0) * Fs))

    Z_list, neuron_id, t0_list = [], [], []
    for i in range(N):
        for t0 in range(0, T - Tw + 1, stride):
            x = torch.from_numpy(V[t0:t0+Tw, i][None, None, :]).float().to(device)
            z = encoder(x).cpu().numpy()[0]
            Z_list.append(z); neuron_id.append(i); t0_list.append(t0)
    Z = np.stack(Z_list, axis=0)
    meta = {
        "neuron_id": np.array(neuron_id, dtype=np.int32),
        "t0": np.array(t0_list, dtype=np.int32),
        "Tw": Tw,
        "Fs": Fs,
    }
    print(f"  Embedded windows: Z shape = {Z.shape}")
    return Z, meta

def train_decoder_on_future(V, state, Fs,
                            student, teacher, pred,
                            Tw_ms=512,
                            stride_ms=None,
                            deltas_ms=(25, 100, 250),
                            steps=2000,
                            batch_size=128,
                            lr=1e-3):
    # Reuse the same FuturePairs dataset
    Tw = int(round((Tw_ms/1000.0) * Fs))
    stride = int(round(((stride_ms if stride_ms is not None else Tw_ms/2) / 1000.0) * Fs))
    deltas = [max(1, int(round((ms/1000.0) * Fs))) for ms in deltas_ms]

    ds = FuturePairs(V, state, Tw=Tw, stride=stride, deltas=deltas)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    # Freeze encoder + predictor
    student.eval(); teacher.eval(); pred.eval()
    for p in student.parameters(): p.requires_grad_(False)
    for p in pred.parameters():    p.requires_grad_(False)

    # Decoder to train
    decoder = VoltageDecoder(d=256, Tw=Tw).to(device)
    opt = torch.optim.Adam(decoder.parameters(), lr=lr)

    it = iter(dl)
    for step in range(1, steps+1):
        try:
            Xc, Xt, St, _ = next(it)
        except StopIteration:
            it = iter(dl)
            Xc, Xt, St, _ = next(it)

        Xc = Xc.to(device)   # [B,1,Tw]
        Xt = Xt.to(device)   # [B,1,Tw]
        St = St.to(device)   # [B]

        with torch.no_grad():
            zc   = student(Xc)             # [B,d]
            zhat = pred(zc, St)            # predicted future embedding [B,d]
            zhat = zhat.detach()           # make sure no grads flow back

        x_pred = decoder(zhat)             # [B,Tw]
        x_true = Xt.squeeze(1)             # [B,Tw]

        loss = F.mse_loss(x_pred, x_true)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 200 == 0 or step == 1:
            print(f"[decoder] step {step}/{steps} | MSE = {loss.item():.6f}")

    decoder.eval()
    return decoder

#Visualization support

def scatter_windows_2d(Z, meta, title="Window embeddings (2D)"):
    ids = meta["neuron_id"]
    if HAVE_UMAP:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.1,
            metric="cosine",
            random_state=0,
        )
        Z2 = reducer.fit_transform(Z)
        print("  2D projection: UMAP")
    else:
        reducer = PCA(n_components=2)
        Z2 = reducer.fit_transform(Z)
        print("  2D projection: PCA (install umap-learn for UMAP).")

    plt.figure(figsize=(6,5))
    sc = plt.scatter(Z2[:,0], Z2[:,1], c=ids, s=6, cmap="tab20", alpha=0.8)
    plt.title(title)
    plt.xlabel("dim 1"); plt.ylabel("dim 2")
    plt.colorbar(sc, label="neuron id")
    plt.tight_layout()
    plt.show()

def scatter_windows_3d(Z, meta, title="Window embeddings (3D)"):
    ids = meta["neuron_id"]
    if HAVE_UMAP:
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=30,
            min_dist=0.1,
            metric="cosine",
            random_state=0,
        )
        Z3 = reducer.fit_transform(Z)
        print("  3D projection: UMAP")
    else:
        reducer = PCA(n_components=3)
        Z3 = reducer.fit_transform(Z)
        print("  3D projection: PCA (install umap-learn for UMAP).")

    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(Z3[:,0], Z3[:,1], Z3[:,2],
                   c=ids, s=6, cmap="tab20", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_zlabel("dim 3")
    fig.colorbar(p, ax=ax, label="neuron id", shrink=0.6)
    plt.tight_layout()
    plt.show()

def neuron_rdm(Z, meta, metric="cosine"):
    ids = meta["neuron_id"]
    d = Z.shape[1]
    N = ids.max() + 1
    means = np.zeros((N, d), dtype=np.float32)
    counts = np.zeros(N, dtype=np.int32)

    for z, i in zip(Z, ids):
        means[i] += z
        counts[i] += 1
    means /= np.maximum(1, counts)[:, None]

    D = pairwise_distances(means, metric=metric)

    plt.figure(figsize=(5,4))
    plt.imshow(D, cmap="viridis")
    plt.title(f"Neuron RDM (1 - {metric} sim)")
    plt.colorbar(label=f"{metric} distance")
    plt.xlabel("neuron")
    plt.ylabel("neuron")
    plt.tight_layout()
    plt.show()

    return means, D

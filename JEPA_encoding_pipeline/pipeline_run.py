base_dir = "/Users/venkatvege/Desktop"

mat_paths = [
    f"{base_dir}/JEPA_Input.mat",
    f"{base_dir}/JEPA_Input_5s.mat",
    f"{base_dir}/JEPA_Input_10s.mat",
    f"{base_dir}/JEPA_Input_12s.mat",
]

results = {}

for mat_path in mat_paths:
    print("\n=== Processing file ===")
    print(mat_path)
    V, Fs, state, t = load_jepa_input(mat_path)

    teacher = train_jepa(
        V, state, Fs,
        Tw_ms=512,
        deltas_ms=(25, 100, 250),
        steps=1500,
        batch_size=128,
    )

    Z, meta = embed_all_windows(teacher, V, Fs, Tw_ms=512)

    base_name = pathlib.Path(mat_path).name
    scatter_windows_2d(Z, meta, title=f"JEPA 2D embeddings\n{base_name}")
    scatter_windows_3d(Z, meta, title=f"JEPA 3D embeddings\n{base_name}")
    means, D = neuron_rdm(Z, meta, metric="cosine")

    results[mat_path] = {
        "Z": Z,
        "meta": meta,
        "neuron_means": means,
        "neuron_rdm": D,
    }

print("\nDone with all files.")

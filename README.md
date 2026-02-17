# Izh FHN Physics with JEPA Latent for Predictive Neural Dynamics
Implementing JEPA style architecture for neural dynamics

Working Abstract: 

We study whether self-supervised latent prediction can yield reconstructive and
predictive representations of neuronal voltage dynamics, and how explicit bio-
physical structure can prevent degenerate solutions. Starting from a JEPA-style
embedding-only objective on short voltage windows, we observe strong failure
modes (embedding collapse and weak waveform-level interpretability) when the
model is trained without a generative pathway. We then introduce a predictive
autoencoding framework (PAN-Lite) that couples an encoder–decoder with a latent
world model trained to forecast future windows, enabling direct reconstruction and
prediction diagnostics in the observation space. To inject mechanistic inductive
bias, we pretrain and regularize latent dynamics using synthetic trajectories from
FitzHugh–Nagumo and Izhikevich models, including pulsed current protocols that
produce spike-like waveforms with known ground-truth hidden states. Across syn-
thetic benchmarks, the physics-informed prior improves reconstruction and helps
recover meaningful latent structure. We apply the same pipeline to real optical
voltage recordings, reporting reconstruction quality, forecast horizon curves, and
embedding geometry analyses to characterize what aspects of neural activity are
learnably predictable and what remains stochastic or unobserved. There remains
more pressure testing to be done to fully understand the system’s interpretation
of the neural dynamics underlying the voltage traces. This document acts to be
a reference point in the implementation of this style of work to neural dynamics
representation - any and all improvements are welcome.

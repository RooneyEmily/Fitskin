# Pansor app capture recommendation (from CameraSettings factorial)

Camera-settings sweeps (Giana, Keaton, Parker, Wooj; Emily Lab-only / no photos) show that
chart-free D65 ΔE is dominated by absolute exposure:

- **Preferred band:** ISO ≤ 100 and shutter ≈ 1/250–1/120 s (cells A / C / E / I).
  Mean ΔE ≈ 3–5 with pipeline L* near FitSkin skin (~50–65).
- **Avoid:** shutter ≈ 1/60 s (cell B) or ISO ≥ 200 (cells F / G / H). These push
  pipeline L* into the ~78–85 range and mean ΔE ≈ 15–21.

Current Pansor-20 indoor captures already sit in the preferred band
(ISO median ≈ 64.0, shutter median ≈ 0.008333333333333333 s;
in-band fraction 1.0). Exposure residual LOO on
camera-settings chose **L_residual**
(uncorrected LOO mean ΔE 5.1692 → chosen 2.869).

Post-Lab exposure residual fitted on these four (lighter-skin) subjects **does not transfer**
to Pansor-20: applying it raises mean ΔE (especially on Black participants). Keep the frozen
D65 + FairFace7 path; use camera-settings as a **capture-policy** signal, not a Lab corrector.

**App lock suggestion:** fix AE near ISO 64–100 and 1/120–1/250; reject or re-prompt if
ISO ≥ 200, shutter ≥ 1/60, or estimated cheek L* ≳ 75 after the frozen path.


### Laura Chalk and the experimental evidence of quantum mechanics
# the objective  of this script is to compute the transition relative amplitudes of the Balmer
# series in the Hydrogen  atom under the effect of a strong Magnetic field.

import numpy as np
from scipy.special import eval_genlaguerre, factorial, sph_harm_y
from scipy.integrate import quad
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from pathlib import Path
from matplotlib.patches import Patch

def R_nl(n, l, r):
    #Hydrogenic radial function R_nl(r), normalized
    rho = 2.0 * r / n
    norm = np.sqrt((2.0/ n)**3
                   * factorial(n - l - 1) / (2.0 * n * factorial(n + l)))
    return norm * np.exp(-rho / 2) * rho**l * eval_genlaguerre(n - l - 1, 2*l + 1, rho)

def Y_lm(l, m, theta, phi):
    #Complex spherical harmonic
    return sph_harm_y(l, m, theta, phi)

def psi_nlm(n, l, m, r, theta, phi):
    return R_nl(n, l, r) * Y_lm(l, m, theta, phi)


### Compute the matrix elements <phi(n,l,m)H_epsilon phi(n',l',m)>
def matrix_element(n1, n2, l1, m1, l2, m2, operator):
    if operator == "z":
        # selection rules:
        if m1 != m2 or abs(l1 - l2) != 1:
            return 0.0

        # radial part
        f = lambda r: R_nl(n1, l1, r) * r * R_nl(n2, l2, r) * r**2
        radial_part, err = quad(f, 0, np.inf)

        # angular part
        n_pts = 400
        theta = np.linspace(0, np.pi, n_pts)
        phi = np.linspace(0, 2*np.pi, n_pts)
        TH, PH = np.meshgrid(theta, phi, indexing="ij")
        integrand = np.conj(Y_lm(l1, m1, TH, PH)) * np.cos(TH) * Y_lm(l2, m2, TH, PH) * np.sin(TH)
        angular_part = np.trapz(np.trapz(integrand, phi, axis=1), theta)
    if operator == "x":
        # selection rules for x
        if abs(m1 - m2) != 1 or abs(l1 - l2) != 1:
            return 0.0
        # radial part
        f = lambda r: R_nl(n1, l1, r) * r * R_nl(n2, l2, r) * r**2
        radial_part, err = quad(f, 0, np.inf)

        # angular part
        n_pts = 400
        theta = np.linspace(0, np.pi, n_pts)
        phi = np.linspace(0, 2*np.pi, n_pts)
        TH, PH = np.meshgrid(theta, phi, indexing="ij")
        integrand = np.conj(Y_lm(l1, m1, TH, PH)) * np.cos(PH) * np.sin(TH) * Y_lm(l2, m2, TH, PH) * np.sin(TH)
        angular_part = np.trapz(np.trapz(integrand, phi, axis=1), theta)

    return radial_part * angular_part

def build_matrix(n1, n2, operator):
    basis1 = [(l, m) for l in range(n1) for m in range(-l, l + 1)]
    basis2 = [(l, m) for l in range(n2) for m in range(-l, l + 1)]
    M = np.zeros((len(basis1), len(basis2)))
    for i, (l1, m1) in enumerate(basis1):
        for j, (l2, m2) in enumerate(basis2):
            M[i, j] = matrix_element(n1, n2, l1, m1, l2, m2, operator)
    return M

def compute_transition_probabilities(val_1, vec_1, val_2, vec_2, H_n1n2, state1, state2):
    result = 0

    for i in range(H_n1n2.shape[0]):
        for j in range(H_n1n2.shape[1]):
            result += H_n1n2[i, j] * vec_1[i, state1] * vec_2[j, state2]

    intensity = abs(result)**2

    energy = val_1[state1] - val_2[state2]
    return intensity, energy

def collect_lines(val_1, vec_1, val_2, vec_2, H_n1n2, states1, states2):
    energies, intensities, labels = [], [], []
    for s1 in states1:
        for s2 in states2:
            inten, en = compute_transition_probabilities(
                val_1, vec_1, val_2, vec_2, H_n1n2, s1, s2)
            energies.append(float(en))
            intensities.append(float(inten))
            labels.append((s1, s2))
    return np.array(energies), np.array(intensities), labels

def merge_degenerate(E, I, tol):
    #  Merge lines whose energies agree within tol
    order = np.argsort(E)
    E, I = E[order], I[order]
    groups_E, groups_I = [], []
    start = 0
    for k in range(1, len(E) + 1):
        if k == len(E) or E[k] - E[k - 1] > tol:
            w = I[start:k]
            groups_E.append(np.average(E[start:k], weights=w) if w.sum() > 0
                            else E[start:k].mean())
            groups_I.append(w.sum())
            start = k
    return np.array(groups_E), np.array(groups_I)

def assign_positions(E):
    unit = 1.5
    return np.rint(E / unit).astype(int)

def plot_intensities(E, I, min_rel_intensity,
                     n1, n2, polarization, title=None, ax=None):
    
 
    COLOR_THEORY, COLOR_OBS, COLOR_KRAMERS, INK = "#2a78d6", "#eb6834", "#2f9e6e", "#0b0b0b"
 
    pos = assign_positions(E)
    I_rel = I / I.max()
    keep = I_rel >= min_rel_intensity
    pos, I_rel = pos[keep], I_rel[keep]
 
    theory = {int(k): float(np.mean(I_rel[np.abs(pos) == k])) for k in np.unique(np.abs(pos))}
    if 0 in theory:
        theory[0] /= 2.0
 
    pol_code = {"parallel": "p", "p": "p", "transverse": "s", "s": "s"}[polarization]
    rows = []
    data_file = Path(__file__).with_name("chalk_observed_intensities.txt")
    if n1 == 2 and data_file.exists():
        for raw in data_file.read_text().splitlines():
            raw = raw.split("#", 1)[0].strip()
            if raw:
                name, n, pol, num, den, observed, calc = raw.split()
                if int(n) == n2 and pol == pol_code:
                    rows.append((int(num), int(den), float(observed), float(calc)))

    obs = {}                           
    if rows:
        comps = sorted({a for a, _, _, _ in rows} | {b for _, b, _, _ in rows})
        parent = {c: c for c in comps}  
        def find(c):
            while parent[c] != c:
                c = parent[c]
            return c
        for a, b, _, _ in rows:
            parent[find(a)] = find(b)
        for root in sorted({find(c) for c in comps}):
            members = [c for c in comps if find(c) == root]
            col = {c: j for j, c in enumerate(members)}
            A, y = [], []
            for a, b, o, _ in rows:    
                if a in col:
                    r = np.zeros(len(members)); r[col[a]] += 1; r[col[b]] -= 1
                    A.append(r); y.append(np.log(o))
            A.append(np.ones(len(members))); y.append(0.0)
            x = np.linalg.lstsq(np.array(A), np.array(y), rcond=None)[0]
            ref = [c for c in members if theory.get(c, 0) > 0]
            shift = np.mean([np.log(theory[c]) - x[col[c]] for c in ref]) if ref else -x.max()
            for c in members:
                obs[c] = float(np.exp(x[col[c]] + shift))
        obs_pos = np.array([s * k for k in obs for s in ((1, -1) if k else (1,))])
        obs_h = np.array([obs[abs(p)] * (2.0 if p == 0 else 1.0) for p in obs_pos])
 
    kram = {}
    kdata_file = Path(__file__).with_name("kramers_intensities.txt")
    if n1 == 2 and kdata_file.exists():
        for raw in kdata_file.read_text().splitlines():
            raw = raw.split("#", 1)[0].strip()
            if raw:
                f = raw.split()
                delta, rp, rpp, rp2, rpp2 = int(f[4]), *(float(x) for x in f[8:12])
                if len(f) > 12 and f[12] == "!":
                    rp2, rpp2 = rp * rp, rpp * rpp
                if int(f[1]) == n2 and f[2] == pol_code:
                    kram[delta] = kram.get(delta, 0.0) + rp2 + rpp2
    if n1 == 2 and not kram:
        print(f"plot_intensities: no Kramers data for n2={n2}, {polarization}")
    kram_pos = kram_h = np.array([])
    if kram:
        theory_full = {int(k): float(np.mean(I_rel[np.abs(pos) == k])) for k in np.unique(np.abs(pos))}
        common = [k for k, v in kram.items() if v > 0 and theory_full.get(k, 0) > 0]
        scale = np.exp(np.mean([np.log(theory_full[k] / kram[k]) for k in common])) if common else 1.0 / max(kram.values())
        kram_pos = np.array([s * k for k, v in kram.items() if v > 0 for s in ((1, -1) if k else (1,))])
        kram_h = np.array([kram[abs(p)] * scale for p in kram_pos])
 
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 2.6))
    fig = ax.figure

    all_pos = sorted(set(pos.tolist()) | (set(obs_pos.tolist()) if obs else set())
                         | set(kram_pos.tolist()))
    for p in all_pos:
        ax.text(p, -0.04, f"{abs(p)}", ha="center", va="top", fontsize=14 if len(all_pos) <= 14 else 8)

    
    series = (["kramers"] if len(kram_pos) else []) + ["theory"] + (["obs"] if obs else [])
    gap = min(np.diff(all_pos)) if len(all_pos) > 1 else 1
    w = 0.75 * gap / len(series) - 0.02  # the width  of  the bars scales with the minimum gap to get always thick bars
    off = {k: (i - (len(series) - 1) / 2) * (w + 0.02) for i, k in enumerate(series)}
    ax.bar(pos + off["theory"], I_rel, width=w, color=COLOR_THEORY, edgecolor=INK,
           linewidth=0.6, zorder=3)
    with plt.rc_context({"hatch.linewidth": 1.0}):
        if obs:
            ax.bar(obs_pos + off["obs"], obs_h, width=w, color=COLOR_OBS, hatch="////",
                   edgecolor=INK, linewidth=0.6, zorder=3)
        if len(kram_pos):
            ax.bar(kram_pos + off["kramers"], kram_h, width=w, color=COLOR_KRAMERS, hatch="\\\\",
                   edgecolor=INK, linewidth=0.6, zorder=3)
 
    
    top = max([1.0] + (list(obs_h) if obs else []) + list(kram_h))
    ax.set_ylim(0, top * 1.05)
    ax.set_xlim(min(all_pos) - 1, max(all_pos) + 1)
    ax.set_xticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.yaxis.set_visible(False)

    
    handles = []
    if len(kram_pos):
        handles.append(Patch(facecolor=COLOR_KRAMERS, edgecolor=INK, linewidth=0.6, hatch="\\\\",
                             label="Kramers 1919"))
    handles.append(Patch(facecolor=COLOR_THEORY, edgecolor=INK, linewidth=0.6, label="Schrödinger 1926"))
    if obs:
        handles.append(Patch(facecolor=COLOR_OBS, edgecolor=INK, linewidth=0.6, hatch="////",
                             label="Chalk 1928"))
    if len(handles) > 1:
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12),
                  ncol=len(handles), frameon=False, fontsize=11)
    
    greek = {3: "alpha", 4: "beta", 5: "gamma", 6: "delta", 7: "epsilon"}
    line = rf"$\mathrm{{H}}_\{greek[n2]}$" if n1 == 2 and n2 in greek else rf"${n2}\to{n1}$"
    pol_sym = r"$\parallel$" if pol_code == "p" else r"$\perp$"
    ax.text(0.0, 1.0, f"{line}{pol_sym}", transform=ax.transAxes, ha="left", va="top", fontsize=18,
            zorder=10,   # above the bars
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85))

    fig.tight_layout()
    pol_name = "parallel" if pol_code == "p" else "transverse"
    line_name = f"H{greek[n2]}" if n1 == 2 and n2 in greek else f"n{n2}_to_n{n1}"
    fig.savefig(f"stark_{line_name}_{pol_name}.pdf", bbox_inches="tight")

    return ax

def main():
    n1, n2 = 2, 6
    polarization = "parallel"
    states1 = range(n1 ** 2)
    states2 = range(n2 ** 2)

    tol = 1e-3
    min_rel_intensity = 1e-4

    H_n1n1 = build_matrix(n1, n1, "z")
    val_1, vec_1 = np.linalg.eigh(H_n1n1)
    H_n2n2 = build_matrix(n2, n2, "z")
    val_2, vec_2 = np.linalg.eigh(H_n2n2)
    if polarization == "parallel":
        H_n1n2 = build_matrix(n1, n2, "z")
    if polarization == "transverse":
            H_n1n2 = build_matrix(n1, n2, "x")


    E, I, labels = collect_lines(val_1, vec_1, val_2, vec_2, H_n1n2, states1, states2)

    E, I = merge_degenerate(E, I, tol)

    plot_intensities(E, I, min_rel_intensity,
                     n1, n2, polarization)


if __name__ == "__main__":
    main()
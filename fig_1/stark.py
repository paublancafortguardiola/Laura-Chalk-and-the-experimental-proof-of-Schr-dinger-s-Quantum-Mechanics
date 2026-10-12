## Python code for the MQST - Advanced Quantum Mechanics coursework essay
## This piece of code is intended to calculate and plot the Stark shift transition lines of the hydrogen atom

## Atomic units are used throughout : hbar = 1, electron charge = 1, electron mass = 1, permittivity = 1.

import numpy as np
from scipy.integrate import quad, dblquad
from scipy.special import eval_genlaguerre, gammaln, sph_harm_y
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import cycle
from matplotlib.ticker import ScalarFormatter

def R_nl(r, n, l):
    """Normalized hydrogen radial wavefunction in atomic units."""
    prefactor = 2 / n**2 * np.exp(
        0.5 * (
            gammaln(n - l) - gammaln(n + l + 1)
        )
    )
    rho = 2 * r / n

    return (
        prefactor
        * rho**l
        * np.exp(-r / n)
        * eval_genlaguerre(n - l - 1, 2*l + 1, rho)
    )


def radial_z(n, l, lp):
    """Radial integral integral R_nl R_nlp r^3 dr."""
    value, error = quad(
        lambda r: R_nl(r, n, l)
                  * R_nl(r, n, lp) * r**3,
        0, np.inf,
        epsabs=1e-10,
        epsrel=1e-10
    )
    return value

def angular_cos(l, lp, m):
    """<l,m|cos(theta)|lp,m>, for lp = l +/- 1."""
    if abs(l - lp) != 1 or abs(m) > min(l, lp):
        return 0.0

    lower_l = min(l, lp)

    return np.sqrt(
        ((lower_l + 1)**2 - m**2)
        / ((2*lower_l + 1) * (2*lower_l + 3))
    )

## spherical harmonics and functions involving non-z components


def angular_cartesian(lf, mf, li, mi, component):
    """Angular matrix element of x/r, y/r, or z/r."""

    def integrand(phi, theta):
        Yf = np.conj(sph_harm_y(lf, mf, theta, phi))
        Yi = sph_harm_y(li, mi, theta, phi)

        if component == "x":
            coord = np.sin(theta) * np.cos(phi)
        elif component == "y":
            coord = np.sin(theta) * np.sin(phi)
        elif component == "z":
            coord = np.cos(theta)
        else:
            raise ValueError("component must be 'x', 'y', or 'z'")

        return Yf * coord * Yi * np.sin(theta)

    real_part, _ = dblquad(
        lambda theta, phi: np.real(integrand(phi, theta)),
        0, 2*np.pi,
        lambda phi: 0,
        lambda phi: np.pi,
        epsabs=1e-9
    )

    imag_part, _ = dblquad(
        lambda theta, phi: np.imag(integrand(phi, theta)),
        0, 2*np.pi,
        lambda phi: 0,
        lambda phi: np.pi,
        epsabs=1e-9
    )

    return real_part + 1j * imag_part

def z_matrix(n):
    """
    Matrix of z in the fixed-n hydrogen basis.
    Basis ordering: (l, m), with l increasing and
    m running from -l to +l.
    """
    basis = [
        (l, m)
        for l in range(n)
        for m in range(-l, l + 1)
    ]

    N = len(basis)
    Z = np.zeros((N, N))

    for a, (l, m) in enumerate(basis):
        for b, (lp, mp) in enumerate(basis):
            if m != mp or abs(l - lp) != 1:
                continue

            Z[a, b] = (
                radial_z(n, l, lp)
                * angular_cos(l, lp, m)
            )

    return basis, Z


def solve_stark(n):
    basis, Z = z_matrix(n)
    z_values, C = np.linalg.eigh(Z)

    # In atomic units, Delta E = F * z_values
    return basis, Z, z_values, C


def radial_dipole(nf, lf, ni, li):
    """Radial integral: integral R_nf,lf * R_ni,li * r^3 dr."""
    value, error = quad(
        lambda r: R_nl(r, nf, lf) * R_nl(r, ni, li) * r**3,
        0, np.inf,
        epsabs=1e-10,
        epsrel=1e-10
    )
    return value


def dipole_z_matrix(nf, ni):
    """
    Matrix of d_z = -z in atomic units.
    Rows: final states (nf); columns: initial states (ni).
    Basis ordering: (l, m), with l increasing and m from -l to +l.
    """
    basis_f = [
        (l, m)
        for l in range(nf)
        for m in range(-l, l + 1)
    ]
    basis_i = [
        (l, m)
        for l in range(ni)
        for m in range(-l, l + 1)
    ]

    D = np.zeros((len(basis_f), len(basis_i)))

    for a, (lf, mf) in enumerate(basis_f):
        for b, (li, mi) in enumerate(basis_i):
            if mf != mi or abs(lf - li) != 1:
                continue

            D[a, b] = (
                -radial_dipole(nf, lf, ni, li)
                * angular_cos(lf, li, mf)
            )

    return basis_f, basis_i, D

basis_f, basis_i, Dalpha = dipole_z_matrix(2, 3)
basis_fb, basis_ib, Dbeta = dipole_z_matrix(2, 4)

print("Matrix shape:", Dalpha.shape)
print("Nonzero matrix elements:", np.count_nonzero(np.abs(Dalpha) > 1e-10))
print(np.round(Dalpha, 5))



_, Z2, shifts2, C2 = solve_stark(2)
_, Z3, shifts3, C3 = solve_stark(3)
_, Z4, shifts4, C4 = solve_stark(4)

D_stark_Halpha = C2.conj().T @ Dalpha @ C3
D_stark_Hbeta = C2.conj().T @ Dbeta @ C4

print("Transformed matrix shape:", D_stark_Halpha.shape)
print("Dipole matrix in Stark basis:")
print(np.round(D_stark_Halpha, 5))

intensities = np.abs(D_stark_Halpha)**2
print("Relative intensities:")
print(np.round(intensities, 5))

## LEVEL SHIFTING

# Electric field in V/m
F_SI = 1e5

# Convert to atomic units: 1 a.u. = 5.142206747e11 V/m
F = F_SI / 5.142206747e11

# Unperturbed hydrogen energies, in Hartree
E2_0 = -1 / (2 * 2**2)
E3_0 = -1 / (2 * 3**2)
E4_0 = -1 / (2 * 4**2)

# Stark-shifted energies, in Hartree
E2 = E2_0 + F * shifts2
E3 = E3_0 + F * shifts3
E4 = E4_0 + F * shifts4

# Emission photon energies: rows are final states, columns initial states
delta_E = E3[np.newaxis, :] - E2[:, np.newaxis]
delta_E_beta = E4[np.newaxis, :] - E2[:, np.newaxis]

# Convert to wavelength in nm
wavelengths_nm = 45.5633525 / delta_E
wavelengths_nm_beta = 45.5633525 / delta_E_beta

print("Wavelengths (nm):")
print(np.round(wavelengths_nm, 5))

def dipole_matrix(nf, ni, component):
    basis_f = [
        (l, m) for l in range(nf) for m in range(-l, l+1)
    ]
    basis_i = [
        (l, m) for l in range(ni) for m in range(-l, l+1)
    ]

    D = np.zeros((len(basis_f), len(basis_i)), dtype=complex)

    for a, (lf, mf) in enumerate(basis_f):
        for b, (li, mi) in enumerate(basis_i):
            if abs(lf-li) != 1 or abs(mf-mi) > 1:
                continue

            radial = radial_dipole(nf, lf, ni, li)
            angular = angular_cartesian(lf, mf, li, mi, component)
            D[a, b] = -radial * angular

    return basis_f, basis_i, D


_, _, Dx0 = dipole_matrix(2, 3, "x")
_, _, Dy0 = dipole_matrix(2, 3, "y")
_, _, Dz0 = dipole_matrix(2, 3, "z")

_, _, Dx0_b = dipole_matrix(2, 4, "x")
_, _, Dy0_b = dipole_matrix(2, 4, "y")
_, _, Dz0_b = dipole_matrix(2, 4, "z")

Dx_stark = C2.conj().T @ Dx0 @ C3
Dy_stark = C2.conj().T @ Dy0 @ C3
Dz_stark = C2.conj().T @ Dz0 @ C3

Dx_b_stark = C2.conj().T @ Dx0_b @ C4
Dy_b_stark = C2.conj().T @ Dy0_b @ C4
Dz_b_stark = C2.conj().T @ Dz0_b @ C4

Ix = np.abs(Dx_stark)**2
Iy = np.abs(Dy_stark)**2
Iz = np.abs(Dz_stark)**2

Ix_b = np.abs(Dx_b_stark)**2
Iy_b = np.abs(Dy_b_stark)**2
Iz_b = np.abs(Dz_b_stark)**2



# Frequency shift relative to the unperturbed line, in atomic units
offsets = shifts3[np.newaxis, :] - shifts2[:, np.newaxis]
offsets_b = shifts4[np.newaxis, :] - shifts2[:, np.newaxis]

# Group transitions with the same shift
line_strengths = {}
line_strengths_b = {}

for i in range(Iz.shape[0]):
    for j in range(Iz.shape[1]):
        shift = round(offsets[i, j], 6)
        line_strengths[shift] = (
            line_strengths.get(shift, 0.0) + Iz[i, j]
        )

for i in range(Iz_b.shape[0]):
    for j in range(Iz_b.shape[1]):
        shift = round(offsets_b[i, j], 6)
        line_strengths_b[shift] = (
            line_strengths_b.get(shift, 0.0) + Iz_b[i, j]
        )


print("Parallel-polarization line strengths:")
for shift, strength in sorted(line_strengths.items()):
    if strength > 1e-10:
        print(f"Shift = {shift:+7.3f}, intensity = {strength:.6f}")

transitions = []

for i in range(Iz.shape[0]):
    for j in range(Iz.shape[1]):
        if Iz[i, j] > 1e-10:
            transitions.append((
                wavelengths_nm[i, j],
                shifts3[j] - shifts2[i],
                Iz[i, j],
                i + 1,
                j + 1
            ))

transitions.sort(key=lambda t: t[0])

transitions_beta = []

for i in range(Iz_b.shape[0]):
    for j in range(Iz_b.shape[1]):
        if Iz_b[i, j] > 1e-10:
            transitions_beta.append((
                wavelengths_nm_beta[i, j],
                shifts4[j] - shifts2[i],
                Iz_b[i, j],
                i + 1,
                j + 1
            ))

transitions_beta.sort(key=lambda t: t[0])



# ---------- Consistent visual style ----------
COLOR_THEORY = "#2a78d6"   # blue
COLOR_OBS = "#eb6834"      # orange
COLOR_KRAMERS = "#2f9e6e"  # green
INK = "#0b0b0b"
from matplotlib.colors import ListedColormap

STARK_COLORS = [
    "#264653",  # dark blue-green
    "#287271",  # teal
    "#2a9d8f",  # green-teal
    "#76a974",  # muted green
    "#e9c46a",  # yellow
    "#f4a261",  # soft orange
    "#e76f51",  # coral
    "#c8553d",  # muted red
    "#6d597a",  # purple
]

HARTREE_TO_EV = 27.211386
ATOMIC_FIELD_V_CM = 5.142206747e9
E2_0 = -1 / (2 * 2**2)
E3_0 = -1 / (2 * 3**2)


def plot_stark_results(
    F_values=None,
    figsize=(3.6, 5.5),
    save_pdf="stark_n3_energy_and_composition.pdf",
    save_png="stark_n3_energy_and_composition.png",
    show=True,
):
    """
    Plot relative Stark energy shifts and Stark-state compositions
    for the n=3 hydrogen manifold.

    Requires the existing solve_stark(n) function.
    """

    # ---------- Publication typography ----------
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.9,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "axes.unicode_minus": False,
    })

    # ---------- Electric-field range ----------
    if F_values is None:
        F_values = np.linspace(0, 1e5, 250)

    F_values = np.asarray(F_values)
    F_au = F_values / ATOMIC_FIELD_V_CM

    # ---------- Stark solver ----------
    basis3, Z3, shifts3, C3 = solve_stark(3)

    # ---------- Figure: energy plot above composition ----------
    fig, axes = plt.subplots(
        2, 1,
        figsize=figsize,
        gridspec_kw={"height_ratios": [1.0, 1.25]},
    )
    fig.patch.set_facecolor("white")

    ax_energy, ax_comp = axes
    palette = [COLOR_THEORY, COLOR_OBS, COLOR_KRAMERS]

    # ---------- Top panel: relative energy shifts ----------
    colors = cycle(palette)
    E0 = E3_0

    for i,lam in enumerate(shifts3):
        delta_E = (F_au * lam) / E0

        ax_energy.plot(
            F_values,
            delta_E,
            color=STARK_COLORS[i % len(STARK_COLORS)],
            linewidth=1.6,
            alpha=0.95,
            solid_capstyle="round",
        )

    ax_energy.axhline(
        0,
        color=INK,
        linewidth=0.8,
        alpha=0.7,
        zorder=0,
    )

    ax_energy.set_xlabel("Electric field (V/cm)")
    ax_energy.set_ylabel(
        r"$\Delta E / E_3^{(0)}$"
    )

    # Scientific notation on both axes
    formatter_x = ScalarFormatter(useMathText=True)
    formatter_x.set_scientific(True)
    formatter_x.set_powerlimits((0, 0))
    ax_energy.xaxis.set_major_formatter(formatter_x)

    formatter_y = ScalarFormatter(useMathText=True)
    formatter_y.set_scientific(True)
    formatter_y.set_powerlimits((0, 0))
    ax_energy.yaxis.set_major_formatter(formatter_y)

    ax_energy.grid(
        axis="y",
        color="#d9d9d9",
        linewidth=0.6,
        alpha=0.65,
    )
    ax_energy.grid(axis="x", visible=False)

    ax_energy.spines["top"].set_visible(False)
    ax_energy.spines["right"].set_visible(False)
    ax_energy.tick_params(
        direction="out",
        length=3.5,
        width=0.8,
        pad=3,
    )

    # ---------- Bottom panel: wavefunction composition ----------
    probabilities = np.abs(C3)**2

    basis_labels = [
        rf"$|3,{l},{m}\rangle$"
        for l, m in basis3
    ]
    state_labels = [
        rf"$|3,{k+1}\rangle$"
        for k in range(C3.shape[1])
    ]

    im = ax_comp.imshow(
        probabilities,
        origin="lower",
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="Blues",
        interpolation="nearest",
    )

    ax_comp.set_xlabel("Stark eigenstate")
    ax_comp.set_ylabel("Hydrogen basis state")

    ax_comp.set_xticks(np.arange(C3.shape[1]))
    ax_comp.set_xticklabels(
        state_labels,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=8,
    )

    ax_comp.set_yticks(np.arange(len(basis3)))
    ax_comp.set_yticklabels(
        basis_labels,
        fontsize=8,
    )

    # Fine cell boundaries
    ax_comp.set_xticks(
        np.arange(-0.5, C3.shape[1], 1),
        minor=True,
    )
    ax_comp.set_yticks(
        np.arange(-0.5, len(basis3), 1),
        minor=True,
    )

    ax_comp.grid(
        which="minor",
        color="gray",
        linewidth=0.6,
        alpha=0.9,
    )

    ax_comp.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )
    ax_comp.tick_params(
        direction="out",
        length=3,
        width=0.8,
        pad=3,
    )

    ax_comp.spines["top"].set_visible(False)
    ax_comp.spines["right"].set_visible(False)

    # ---------- Single colorbar ----------
    cbar = fig.colorbar(
        im,
        ax=ax_comp,
        fraction=0.046,
        pad=0.035,
    )
    cbar.set_label(
        r"$|c_{lm}|^2$",
        rotation=90,
        labelpad=5,
        fontsize=9,
    )
    cbar.outline.set_linewidth(0.7)
    cbar.outline.set_edgecolor(INK)
    cbar.ax.tick_params(
        labelsize=8,
        length=2.5,
        width=0.7,
    )

    # ---------- Layout ----------
    fig.tight_layout(
        pad=0.8,
        h_pad=0.4,
    )

    # ---------- Export ----------
    if save_pdf is not None:
        fig.savefig(
            save_pdf,
            bbox_inches="tight",
            facecolor="white",
        )

    if save_png is not None:
        fig.savefig(
            save_png,
            bbox_inches="tight",
            facecolor="white",
            dpi=600,
        )

    if show:
        plt.show()

    return fig, axes

# ---------- Generate the figure ----------
fig, axes = plot_stark_results()
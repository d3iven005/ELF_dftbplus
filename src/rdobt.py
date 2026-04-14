import os
import re
import numpy as np


L_MAP = {
    0: "s",
    1: "p",
    2: "d",
    3: "f",
}


def _extract_brace_block(text, start_idx):
    """Return the content inside the first {...} starting at start_idx."""
    i = text.find("{", start_idx)
    if i == -1:
        raise ValueError("Cannot find opening '{'.")

    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j], j + 1

    raise ValueError("Unmatched braces in HSD file.")


def _parse_number_block(block_text, key):
    """
    Parse blocks like:
    Exponents = { ... }
    Coefficients = { ... }
    """
    m = re.search(rf"{key}\s*=\s*\{{", block_text)
    if not m:
        return None

    content, _ = _extract_brace_block(block_text, m.start())
    nums = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?", content)
    return np.array([float(x) for x in nums], dtype=float)


def _parse_scalar(block_text, key, cast=float):
    m = re.search(rf"{key}\s*=\s*([^\s\}}]+)", block_text)
    if not m:
        return None
    return cast(m.group(1))


def _parse_wfc_hsd(filename):
    """Parse wfc.hsd into a nested dict."""
    with open(filename, "r") as f:
        text = f.read()

    data = {}
    pos = 0

    # Match element block: C = { ... }, H = { ... }, Mg = { ... }
    element_pat = re.compile(r"\b([A-Z][a-z]?)\s*=\s*\{")

    while True:
        m = element_pat.search(text, pos)
        if not m:
            break

        elem = m.group(1)
        elem_block, next_pos = _extract_brace_block(text, m.start())
        pos = next_pos

        atomic_number = _parse_scalar(elem_block, "AtomicNumber", int)

        orbital_list = []
        orb_pos = 0
        orb_pat = re.compile(r"\bOrbital\s*=\s*\{")

        while True:
            mo = orb_pat.search(elem_block, orb_pos)
            if not mo:
                break

            orb_block, orb_next = _extract_brace_block(elem_block, mo.start())
            orb_pos = orb_next

            l = _parse_scalar(orb_block, "AngularMomentum", int)
            occ = _parse_scalar(orb_block, "Occupation", float)
            cutoff = _parse_scalar(orb_block, "Cutoff", float)
            exponents = _parse_number_block(orb_block, "Exponents")
            coefficients = _parse_number_block(orb_block, "Coefficients")

            orbital_list.append({
                "l": l,
                "occupation": occ,
                "cutoff": cutoff,
                "exponents": exponents,
                "coefficients": coefficients,
            })

        data[elem] = {
            "AtomicNumber": atomic_number,
            "Orbitals": orbital_list,
        }

    return data


def _build_r_grid(rmax, dr=0.01):
    n = int(np.floor(rmax / dr)) + 1
    return np.linspace(0.0, dr * (n - 1), n)


def _sto_radial_from_block(r, l, exponents, coefficients, cutoff):
    nexp = len(exponents)
    ncoef = len(coefficients)

    if ncoef % nexp != 0:
        raise ValueError("Coefficient size mismatch")

    npow = ncoef // nexp

    coeff_mat = coefficients.reshape((nexp, npow))

    radial = np.zeros_like(r)

    for i in range(nexp):
        alpha = exponents[i]

        for p in range(npow):
            c = coeff_mat[i, p]
            radial += c * r**(l + p) * np.exp(-alpha * r)

    radial[r > cutoff] = 0.0

    return np.column_stack((r, radial))


def rdobt(X, dr=0.01, default_rmax=None):
    """
    Read DFTB+ wfc.hsd and return dict of numerical orbitals.

    Parameters
    ----------
    X : str
        filename, e.g. 'wfc.hsd' or 'Odata/wfc.hsd'
    dr : float
        radial grid spacing
    default_rmax : float or None
        if not None, override orbital cutoff and use this rmax for all orbitals

    Returns
    -------
    obt_dict : dict
        keys like 'H_s', 'C_s', 'C_p', ...
        values are np.ndarray with shape (nr, 2):
            [:,0] = r
            [:,1] = orbital value
    """
    if os.path.isfile(X):
        filename = X
    else:
        filename = os.path.join("./Odata", X)

    data = _parse_wfc_hsd(filename)

    obt_dict = {}

    for elem, elem_info in data.items():
        for orb in elem_info["Orbitals"]:
            l = orb["l"]
            label = L_MAP.get(l, f"l{l}")

            rmax = default_rmax if default_rmax is not None else orb["cutoff"]
            r = _build_r_grid(rmax, dr=dr)

            arr = _sto_radial_from_block(
                r=r,
                l=l,
                exponents=orb["exponents"],
                coefficients=orb["coefficients"],
                cutoff=orb["cutoff"],
            )

            key = f"{elem}_{label}"
            obt_dict[key] = arr

    return obt_dict

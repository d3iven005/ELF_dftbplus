import os
import re
import numpy as np


def _resolve_file(path_or_name, default_name=None):
    candidates = []

    if path_or_name is not None:
        candidates.append(path_or_name)
        candidates.append(os.path.join("./00_inputdata", path_or_name))

    if default_name is not None:
        candidates.append(default_name)
        candidates.append(os.path.join("./00_inputdata", default_name))

    for f in candidates:
        if f is not None and os.path.isfile(f):
            return f

    raise FileNotFoundError(f"Cannot find file from candidates: {candidates}")


def _parse_band_file(band_file):
    """
    Parse DFTB+ band.out.

    Returns
    -------
    K_coe : np.ndarray, shape (nk,)
    E_list : np.ndarray, shape (nk, n_orb)
    Occ_list : np.ndarray, shape (nk, n_orb)
    K_point : np.ndarray, shape (nk, 3)
        DFTB+ band.out shown here does not contain explicit kx,ky,kz,
        so for now Gamma-only -> [0,0,0].
    """
    with open(band_file, "r") as f:
        lines = [line.rstrip("\n") for line in f]

    k_weights = []
    e_blocks = []
    occ_blocks = []
    k_points = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("KPT"):
            # Example:
            # KPT            1  SPIN            1  KWEIGHT    1.0000000000000000
            m = re.search(r"KWEIGHT\s+([-+0-9Ee\.]+)", line)
            if not m:
                raise ValueError(f"Failed to parse KWEIGHT from line: {line!r}")
            kweight = float(m.group(1))

            eigs = []
            occs = []
            i += 1

            while i < len(lines):
                s = lines[i].strip()
                if not s:
                    i += 1
                    continue
                if s.startswith("KPT"):
                    break

                parts = s.split()
                # band line: idx energy occ
                if len(parts) >= 3:
                    try:
                        _idx = int(parts[0])
                        energy = float(parts[1])
                        occ = float(parts[2])
                    except ValueError:
                        break
                    eigs.append(energy)
                    occs.append(occ)
                    i += 1
                else:
                    break

            k_weights.append(kweight)
            e_blocks.append(eigs)
            occ_blocks.append(occs)
            k_points.append([0.0, 0.0, 0.0])
            continue

        i += 1

    if not k_weights:
        raise ValueError(f"No KPT blocks found in {band_file}")

    return (
        np.array(k_weights, dtype=float),
        np.array(e_blocks, dtype=float),
        np.array(occ_blocks, dtype=float),
        np.array(k_points, dtype=float),
    )


def _parse_one_eigenvector_block(block_lines):
    """
    Parse one 'Eigenvector: n (up)' block from eigenvec.out.

    Returns
    -------
    coeffs : list of float
        AO coefficients in the basis order expected by PHI_cal.py:
        - H-like 1 orbital atom: [s]
        - sp atom: [s, pz, px, py]
    """
    coeffs = []

    current_atom_idx = None
    current_atom_orbs = {}

    def flush_atom(atom_orbs):
        if not atom_orbs:
            return []

        # s-only atom
        if "s" in atom_orbs and all(k not in atom_orbs for k in ("p_x", "p_y", "p_z")):
            return [atom_orbs["s"]]

        # sp atom: reorder to match PHI_cal.py
        if "s" in atom_orbs:
            out = [atom_orbs["s"]]
            if "p_z" in atom_orbs and "p_x" in atom_orbs and "p_y" in atom_orbs:
                out.extend([atom_orbs["p_z"], atom_orbs["p_x"], atom_orbs["p_y"]])
                return out

        raise ValueError(f"Unsupported or incomplete orbital set for one atom: {atom_orbs}")

    for raw in block_lines:
        line = raw.rstrip()
        if not line.strip():
            continue

        parts = line.split()
        if not parts:
            continue

        # New atom line
        if parts[0].isdigit():
            # flush previous atom
            if current_atom_idx is not None:
                coeffs.extend(flush_atom(current_atom_orbs))

            current_atom_idx = int(parts[0])
            current_atom_orbs = {}

            if len(parts) < 4:
                raise ValueError(f"Malformed atom/orbital line: {line!r}")

            orb_label = parts[2]
            coeff = float(parts[3])
            current_atom_orbs[orb_label] = coeff

        else:
            # continuation line
            if current_atom_idx is None:
                raise ValueError(f"Found continuation line before any atom line: {line!r}")

            if len(parts) < 2:
                raise ValueError(f"Malformed continuation orbital line: {line!r}")

            orb_label = parts[0]
            coeff = float(parts[1])
            current_atom_orbs[orb_label] = coeff

    # flush last atom
    if current_atom_idx is not None:
        coeffs.extend(flush_atom(current_atom_orbs))

    return coeffs


def _parse_eigenvec_file(eigenvec_file):
    """
    Parse DFTB+ eigenvec.out.

    Returns
    -------
    coeff_blocks : list[list[float]]
        coeff_blocks[imo][ibasis]
    """
    with open(eigenvec_file, "r") as f:
        lines = [line.rstrip("\n") for line in f]

    eig_header_idx = []
    for i, line in enumerate(lines):
        if line.strip().startswith("Eigenvector:"):
            eig_header_idx.append(i)

    if not eig_header_idx:
        raise ValueError(f"No 'Eigenvector:' blocks found in {eigenvec_file}")

    eig_header_idx.append(len(lines))

    coeff_blocks = []

    for iblk in range(len(eig_header_idx) - 1):
        start = eig_header_idx[iblk]
        end = eig_header_idx[iblk + 1]
        block_lines = lines[start + 1:end]
        coeffs = _parse_one_eigenvector_block(block_lines)
        coeff_blocks.append(coeffs)

    return coeff_blocks


def rdwf(X=None, y=None):
    """
    Read DFTB+ eigenvector/band files.

    Parameters
    ----------
    X : str or None
        Ignored as a stem in practice; DFTB+ files are expected to be:
        - eigenvec.out
        - band.out
        If X is a directory or filename path, resolution is attempted.
    y : list-like or None
        Kept only for interface compatibility. Not required here.

    Returns
    -------
    K_coe : np.ndarray
        shape (nk,)
    E_list : np.ndarray
        shape (nk, n_orb)
    Occ_list : np.ndarray
        shape (nk, n_orb)
    phi_coe_list : np.ndarray
        shape (nk, n_orb, n_basis)
    K_point : np.ndarray
        shape (nk, 3)
    """
    # Resolve files
    if X is not None and os.path.isdir(X):
        eigenvec_file = _resolve_file(os.path.join(X, "eigenvec.out"))
        band_file = _resolve_file(os.path.join(X, "band.out"))
    else:
        eigenvec_file = _resolve_file("eigenvec.out")
        band_file = _resolve_file("band.out")

    # Parse band.out
    K_coe, E_list, Occ_list, K_point = _parse_band_file(band_file)

    # Parse eigenvec.out
    coeff_blocks = _parse_eigenvec_file(eigenvec_file)

    n_orb_band = E_list.shape[1]
    n_orb_eig = len(coeff_blocks)

    if n_orb_band != n_orb_eig:
        raise ValueError(
            f"Number of orbitals mismatch: band.out has {n_orb_band}, "
            f"but eigenvec.out has {n_orb_eig}"
        )

    n_basis = len(coeff_blocks[0])
    for i, c in enumerate(coeff_blocks):
        if len(c) != n_basis:
            raise ValueError(
                f"Inconsistent basis size in eigenvector block {i+1}: "
                f"expected {n_basis}, got {len(c)}"
            )

    # DFTB+ example here is Gamma-only / one K block
    # so shape should be (1, n_orb, n_basis)
    phi_coe_list = np.array([coeff_blocks], dtype=complex)

    # Safety check against band k-block count
    if len(K_coe) != 1:
        raise ValueError(
            f"Current eigenvec.out parser assumes one K block, but band.out has {len(K_coe)} K blocks."
        )

    return K_coe, E_list, Occ_list, phi_coe_list, K_point

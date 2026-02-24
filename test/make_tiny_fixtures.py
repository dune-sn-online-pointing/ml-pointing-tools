#!/usr/bin/env python3
"""Generate tiny local NPZ fixtures for end-to-end smoke tests.

This avoids relying on EOS/CVMFS datasets while still exercising the real
data-loading + training + analysis code paths.

Outputs are written under test/artifacts (which is gitignored).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO_DIR / "test" / "artifacts"


def _write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def make_ct_volume_fixtures(base_dir: Path, plane: str = "X", n_per_class: int = 6) -> dict[str, str]:
    """Create minimal CT volume fixtures.

    The CT volume trainer expects files matching `*plane{plane}.npz` containing
    an `images` array where each element is shape (208, 1242).
    """

    rng = np.random.default_rng(123)
    height, width = 208, 1242

    es_dir = base_dir / "ct_volume" / "es"
    cc_dir = base_dir / "ct_volume" / "cc"
    es_dir.mkdir(parents=True, exist_ok=True)
    cc_dir.mkdir(parents=True, exist_ok=True)

    # Use uint16 to keep files reasonably small; training script casts to float.
    es_images = rng.integers(0, 4096, size=(n_per_class, height, width), dtype=np.uint16)
    cc_images = rng.integers(0, 4096, size=(n_per_class, height, width), dtype=np.uint16)

    _write_npz(es_dir / f"tiny_es_plane{plane}.npz", images=es_images)
    _write_npz(cc_dir / f"tiny_cc_plane{plane}.npz", images=cc_images)

    return {
        "es_directory": str(es_dir) + "/",
        "cc_directory": str(cc_dir) + "/",
    }


def make_ed_three_plane_fixtures(base_dir: Path, n_samples: int = 6) -> dict[str, str]:
    """Create minimal ED 3-plane matched fixtures.

    `python/data_loader.load_three_plane_matched` expects a flat directory with
    files `*_planeX.npz` and corresponding `*_planeU.npz`/`*_planeV.npz`.
    Each NPZ must contain `images` and `metadata` arrays, where metadata has 14
    columns and `match_id` is column 13.
    """

    rng = np.random.default_rng(456)
    height, width = 128, 16
    out_dir = base_dir / "ed_three_plane"
    out_dir.mkdir(parents=True, exist_ok=True)

    images_u = rng.normal(size=(n_samples, height, width)).astype(np.float32)
    images_v = rng.normal(size=(n_samples, height, width)).astype(np.float32)
    images_x = rng.normal(size=(n_samples, height, width)).astype(np.float32)

    # Minimal 14-column metadata layout, float32.
    # We only really need: event_id (col 0), momentum px/py/pz (cols 7-9),
    # true_particle_energy (col 10, MeV), match_id (col 13).
    metadata = np.zeros((n_samples, 14), dtype=np.float32)
    event_id = 123.0
    metadata[:, 0] = event_id
    metadata[:, 13] = np.arange(n_samples, dtype=np.float32)  # match_id

    mom = rng.normal(size=(n_samples, 3)).astype(np.float32)
    mom[np.linalg.norm(mom, axis=1) < 1e-6] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    metadata[:, 7:10] = mom

    energies = rng.uniform(5.0, 50.0, size=(n_samples,)).astype(np.float32)
    metadata[:, 10] = energies

    # Write three plane files with consistent metadata.
    _write_npz(out_dir / "tiny_planeU.npz", images=images_u, metadata=metadata)
    _write_npz(out_dir / "tiny_planeV.npz", images=images_v, metadata=metadata)
    _write_npz(out_dir / "tiny_planeX.npz", images=images_x, metadata=metadata)

    return {"data_directory": str(out_dir)}


def main() -> int:
    fixtures_dir = ARTIFACTS / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    ct = make_ct_volume_fixtures(fixtures_dir)
    ed = make_ed_three_plane_fixtures(fixtures_dir)

    print("✓ Wrote tiny fixtures under:", fixtures_dir)
    print("CT:")
    print("  ES:", ct["es_directory"])
    print("  CC:", ct["cc_directory"])
    print("ED:")
    print("  3-plane:", ed["data_directory"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

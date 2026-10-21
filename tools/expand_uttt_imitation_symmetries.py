#!/usr/bin/env python3
"""Create a symmetry-expanded copy of a UTTT offline history dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from config import load_config
from game_runner import exploit_symmetries, glob_file_triples, resolve_history_dir


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def expand_dataset_symmetries(
    input_dir: Path,
    output_dir: Path,
    config,
    *,
    overwrite: bool = False,
) -> dict:
    """Expand every input position through UTTT's eight exact symmetries.

    ``exploit_symmetries`` already owns the authoritative action and plane
    transformations. This wrapper gives it a disposable staging directory so
    source history remains untouched.
    """
    if config.game != "uttt":
        raise ValueError(f"UTTT symmetry expansion requires game='uttt', got {config.game!r}")

    source_history = Path(resolve_history_dir(str(input_dir)))
    source_triples = glob_file_triples(str(source_history), "*-canonical-*.ptz")
    if not source_triples:
        raise FileNotFoundError(f"No compressed history triples found in {source_history}")

    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)

    staging_dir = output_dir / "tmp_history"
    history_dir = output_dir / "history"
    staging_dir.mkdir(parents=True)
    history_dir.mkdir(parents=True)

    for canonical_path, value_path, policy_path, _ in source_triples:
        for source in (canonical_path, value_path, policy_path):
            _link_or_copy(Path(source), staging_dir / Path(source).name)

    exploit_symmetries(config, {"tmp_history": str(staging_dir)}, iteration=0)

    output_triples = glob_file_triples(str(staging_dir), "*-syms-canonical-*.ptz")
    if not output_triples:
        raise RuntimeError("UTTT symmetry expansion completed without producing history")
    for canonical_path, value_path, policy_path, _ in output_triples:
        for source in (canonical_path, value_path, policy_path):
            shutil.move(source, history_dir / Path(source).name)

    source_positions = sum(size for _, _, _, size in source_triples)
    output_positions = sum(size for _, _, _, size in output_triples)
    metadata = {
        "status": "complete",
        "source": str(input_dir.resolve()),
        "source_positions": source_positions,
        "positions": output_positions,
        "symmetry_count": config.Game.NUM_SYMMETRIES(),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    staging_dir.rmdir()
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, {})
    metadata = expand_dataset_symmetries(
        Path(args.input_dir),
        Path(args.output_dir),
        config,
        overwrite=args.overwrite,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

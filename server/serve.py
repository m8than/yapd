"""Unified worker entry point for all supported speech model engines.

One worker owns one model and one GPU, matching vLLM's worker isolation. A
heterogeneous fleet is composed behind ``server.router``.
"""
from __future__ import annotations

import argparse
import sys
from server.models import FLASH_MODEL, KOKORO_MODEL, TURBO_MODEL, canonical_model


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", required=True)
    selected, worker_args = parser.parse_known_args()
    model_id = canonical_model(selected.model)
    engine = {
        KOKORO_MODEL: "kokoro",
        TURBO_MODEL: "turbo",
        FLASH_MODEL: "flash",
    }.get(model_id)
    if engine is None:
        parser.error(
            "unsupported model; expected kokoro, turbo, flash, or their "
            "canonical Hugging Face model ID"
        )

    if engine == "kokoro":
        from server.kokoro_app import main as worker_main
        worker_args.extend(("--repo-id", model_id))
    else:
        from server.app import main as worker_main
        worker_args.extend((
            "--model", engine,
            "--base-model-id", model_id,
        ))

    sys.argv = [sys.argv[0], *worker_args]
    worker_main()


if __name__ == "__main__":
    main()

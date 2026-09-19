from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kaggrl.structured_actor_critic import StructuredActorCritic, export_numpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()
    model = StructuredActorCritic.from_bc_checkpoint(args.checkpoint)
    model.eval()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    export_numpy(model, out)
    print(out)


if __name__ == "__main__":
    main()
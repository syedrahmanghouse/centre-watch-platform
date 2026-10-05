#!/usr/bin/env python3
"""Evaluate labelled footage; run from any directory without moving raw media."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PLATFORM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLATFORM))
from server.evaluation import EvaluationError, run_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Path to a completed annotation manifest")
    parser.add_argument("--incoming", type=Path, default=PLATFORM.parent / "data" / "incoming")
    parser.add_argument("--output", type=Path, default=PLATFORM.parent / "data" / "results" / datetime.now(timezone.utc).strftime("evaluation-%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        report = run_evaluation(manifest, args.incoming, args.output,
                                progress=lambda pct, message: print(f"{pct}% {message}", file=sys.stderr))
    except (EvaluationError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Evaluation failed: {exc}\n")
    print(json.dumps({"output": str(args.output.resolve()), "metrics": report["metrics"], "coverage": report["coverage"]}, indent=2))


if __name__ == "__main__":
    main()

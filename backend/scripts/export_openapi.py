"""Export the OpenAPI schema to backend/docs/openapi.json.

    uv run python scripts/export_openapi.py
    uv run python scripts/export_openapi.py --output docs/openapi.json --indent 2

Builds the app without starting the server (no lifespan, no database writes), so it is
safe to run in CI before any model has been trained.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.main import create_app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=BACKEND_DIR / "docs" / "openapi.json",
        help="Where to write the schema (default: backend/docs/openapi.json)",
    )
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()

    schema = create_app().openapi()
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(schema, indent=args.indent, ensure_ascii=False) + "\n", "utf-8")

    paths = sorted(schema.get("paths", {}))
    print(f"wrote {output.relative_to(BACKEND_DIR)} ({output.stat().st_size} bytes)")
    print(f"{len(paths)} paths:")
    for path in paths:
        methods = ", ".join(sorted(m.upper() for m in schema["paths"][path]))
        print(f"  {methods:<12} {path}")


if __name__ == "__main__":
    main()
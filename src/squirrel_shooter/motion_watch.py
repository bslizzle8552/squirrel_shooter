"""Compatibility entry point for the shared-camera combined application."""

from __future__ import annotations

from .app import main as combined_main


def main() -> int:
    print(
        "motion_watch now uses the shared camera runtime. By default it also starts "
        "the dashboard; pass --no-dashboard to run motion processing alone."
    )
    return combined_main()


if __name__ == "__main__":
    raise SystemExit(main())

"""Install the pinned MobileNet-SSD model used by the Pi classifier."""

from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable


COMMIT = "bb17b6c3eef36d80be441ae8e5339be66e8e3b7a"
BASE_URL = f"https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/{COMMIT}"
DEFAULT_DESTINATION = Path("models/mobilenet-ssd")


@dataclass(frozen=True)
class ModelFile:
    name: str
    sha256: str


MODEL_FILES = (
    ModelFile("deploy.prototxt", "2d180f723b3109e21f8287f6b3c691390d07b60eed998327cd3259ffa0e50608"),
    ModelFile("mobilenet_iter_73000.caffemodel", "52eed8be80522c152a17fb56740de705b79881bde1a167e0e747310523685fc7"),
    ModelFile("LICENSE", "5de433821cfe672af2fca73c3005f48af16121c95f6a11e1031b07807ea59905"),
)


def install_model(
    destination: Path = DEFAULT_DESTINATION,
    *,
    opener: Callable[..., BinaryIO] = urllib.request.urlopen,
) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    installed: list[Path] = []
    for item in MODEL_FILES:
        target = destination / item.name
        temporary = target.with_suffix(target.suffix + ".download")
        digest = hashlib.sha256()
        with opener(f"{BASE_URL}/{item.name}", timeout=60) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                handle.write(chunk)
        if digest.hexdigest() != item.sha256:
            temporary.unlink(missing_ok=True)
            raise OSError(f"Checksum mismatch for {item.name}")
        os.replace(temporary, target)
        installed.append(target)
    return installed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and verify the pinned MIT-licensed MobileNet-SSD model")
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        installed = install_model(args.destination)
    except Exception as exc:
        print(f"Classifier model setup failed: {type(exc).__name__}: {exc}")
        return 1
    print("Classifier model installed and verified:")
    for path in installed:
        print(f"  {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

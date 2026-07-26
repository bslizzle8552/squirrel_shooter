from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from conftest import write_test_config
from squirrel_shooter.mask_preview import main


def test_mask_preview_command_saves_annotated_image(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    output = tmp_path / "active-mask.jpg"
    cv2.imwrite(str(source), np.full((100, 160, 3), 200, dtype=np.uint8))

    result = main(
        [
            "--config",
            str(write_test_config(tmp_path)),
            "--image",
            str(source),
            "--output",
            str(output),
        ]
    )

    saved = cv2.imread(str(output))
    assert result == 0
    assert saved is not None
    assert saved.shape == (100, 160, 3)
    assert saved[5, 5].mean() < 150

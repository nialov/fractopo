"""
Workflow visualisation of ``fractopo``
======================================

See ``examples/fractopo_workflow_visualisation.py`` for the code.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import fractopo_workflow_visualisation
import matplotlib.pyplot as plt
from PIL import Image

with TemporaryDirectory() as tmp_dir:
    fig_output_path = Path(tmp_dir) / "fractopo_workflow_visualisation.jpg"
    fractopo_workflow_visualisation.main(output_path=fig_output_path)

    figure, ax = plt.subplots(1, 1, figsize=(9, 9))
    with Image.open(fig_output_path) as image:
        ax.imshow(image)
        ax.axis("off")

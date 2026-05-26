import tempfile
from pathlib import Path

from fractopo.interop.porepy import check_porepy_2d_csv_format

EXAMPLE_POREPY_2D_CSV = """# Domain X_MIN, Y_MIN, X_MAX, Y_MAX
-54900, 6730000, -52000, 6733000
# OBJECTID,START_X,START_Y,END_X,END_Y
-5.361122869999999966e+04,6.732113743700000457e+06,-5.466355320000000211e+04,6.730962761900000274e+06
"""


def test_check_porepy_2d_csv_format_passes():
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write(EXAMPLE_POREPY_2D_CSV)
        tmp.flush()
        path = Path(tmp.name)
    try:
        assert check_porepy_2d_csv_format(path)
    finally:
        path.unlink()

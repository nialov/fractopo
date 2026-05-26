import logging
from pathlib import Path

from beartype import beartype

log = logging.getLogger(__name__)


@beartype
def check_porepy_2d_csv_format(csv_path: Path) -> bool:
    """
    Check that input csv file uses porepy 2d csv format.

    Returns True if format matches, raises ValueError if not.
    """

    def _is_float(val: str) -> bool:
        try:
            float(val)
            return True
        except ValueError:
            return False

    lines = [line.strip() for line in csv_path.read_text().splitlines() if line.strip()]
    log.info("Loaded %d non-empty lines from %s", len(lines), csv_path)

    # 1. Domain line (skip comments)
    try:
        domain_idx = next(i for i, line in enumerate(lines) if not line.startswith("#"))
    except StopIteration:
        raise ValueError("No domain line found after comments.")

    first_fields = [x.strip() for x in lines[domain_idx].split(",")]
    log.info("Domain line fields: %s", first_fields)

    if len(first_fields) != 4 or not all(_is_float(x) for x in first_fields):
        raise ValueError(f"Domain line must have 4 float values, got: {first_fields}")

    # 2. Header line (find next comment after domain)
    header_idx = None
    for i in range(domain_idx + 1, len(lines)):
        if lines[i].startswith("#"):
            header_idx = i
            break
    if header_idx is None:
        log.error("No header comment line found after domain line.")
        raise ValueError("No header comment line found after domain line.")

    header = lines[header_idx].replace("#", "").strip().upper()
    log.info("Header line: '%s'", header)

    if header != "OBJECTID,START_X,START_Y,END_X,END_Y":
        raise ValueError(
            f"Header must be OBJECTID,START_X,START_Y,END_X,END_Y, got: {header}"
        )

    # 3. Data lines: all remaining non-comment lines after header
    for i, line in enumerate(lines[header_idx + 1 :], start=header_idx + 2):
        if line.startswith("#"):
            continue
        fields = [x.strip() for x in line.split(",")]
        if len(fields) != 4 or not all(_is_float(x) for x in fields):
            raise ValueError(f"Row {i} must have 4 float values, got: {fields}")

    log.info("PorePy 2D CSV format check PASSED for file: %s", csv_path)
    return True

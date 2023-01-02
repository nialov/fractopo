from pathlib import Path

import cog

SKIP_CONDITIONS = ["full documentation is hosted", "fractopo.readthedocs"]


def main():
    cog.outl("")
    readme = Path("README.rst")
    for line in readme.read_text().splitlines():

        if any(condition in line.lower() for condition in SKIP_CONDITIONS):
            continue
        if "figure::" in line:
            line = line.replace("docs_src/", "")
        # if "figure::" in line:
        #     continue
        # if "full documentation is hosted" in line.lower():
        #     continue
        # if "fractopo.readthedocs" in line.lower():
        #     continue
        cog.outl(line)


if __name__ == "__main__":
    main()

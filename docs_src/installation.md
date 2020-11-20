# Module installation

## From source into pipenv

* Requires Python >3.7 installed with
  [pipenv](https://pipenv.pypa.io/en/latest/)

~~~bash
git clone https://github.com/nialov/geotransform.git
cd geotransform
pipenv sync
pipenv shell
~~~

If if you want to run tests or make documentation add --dev after pipenv
sync. tox runs the test suite, makes documentation and syncs Pipfile
-> setup.py files.

~~~bash
pipenv sync --dev
pipenv shell
tox
~~~

Script now accessible as geotrans inside the pipenv

~~~bash
geotrans --help
~~~

## Install into your Python environment of choice with pip

~~~bash
pip install git+https://github.com/nialov/fractopo#egg=fractopo
~~~

## Dependencies

This module is dependant on third-party Python libraries (and their subsequent
dependancies). These are installed automatically.

Dependencies include:

* Geospatial analysis:
  * [geopandas](https://geopandas.org/)
  * [pandas](https://pandas.pydata.org/docs/)
  * [pygeos](https://pygeos.readthedocs.io/en/latest/)
* Numerical/statistical analysis:
  * [numpy](https://numpy.org/)
  * [scipy](https://www.scipy.org/)
  * [sklearn](https://scikit-learn.org/stable/index.html)
  * [powerlaw](https://github.com/jeffalstott/powerlaw)
* I/O:
  * Spatial data:
    * [GDAL](https://gdal.org/)
    * [Fiona](https://fiona.readthedocs.io/en/latest/)
  * Other:
    * [openpyxl](https://openpyxl.readthedocs.io/en/stable/)
    * [xlrd](https://xlrd.readthedocs.io/en/latest/)
* Graphical plotting
  * [matplotlib](https://matplotlib.org/)
  * [python-ternary](https://github.com/marcharper/Python-ternary)
  * [seaborn](https://seaborn.pydata.org/)

# Module installation

## From source into pipenv

* Requires Python >3.7 installed with
  [pipenv](https://pipenv.pypa.io/en/latest/)

~~~bash
git clone https://github.com/nialov/geotransform.git
cd geotransform
pipenv install
pipenv shell
~~~

If if you want to run tests or make documentation add --dev after pipenv
install. tox runs the test suite, makes documentation and syncs Pipfile
-> setup.py files.

~~~bash
pipenv install --dev
pipenv shell
tox
~~~

Script now accessible as geotrans inside the pipenv

~~~bash
geotrans --help
~~~

## Install into your Python environment of choice with pip

~~~bash
git clone https://github.com/nialov/geotransform.git
cd geotransform
pip3 install .
~~~

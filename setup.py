#!/usr/bin/env python

from setuptools import setup, find_packages

ENTRY_POINTS = {
    # Entry point used to specify packages containing tutorials accessible
    # from welcome screen. Tutorials are saved Orange Workflows (.ows files).
    'orange.widgets.tutorials': (
        # Syntax: any_text = path.to.package.containing.tutorials
        'exampletutorials = orangecontrib.infrared.tutorials',
    ),

    # Entry point used to specify packages containing widgets.
    'orange.widgets': (
        # Syntax: category name = path.to.package.containing.widgets
        # Widget category specification can be seen in
        #    orangecontrib/example/widgets/__init__.py
        'Infrared = orangecontrib.infrared.widgets',
    ),
}

KEYWORDS = [
    # [PyPi](https://pypi.python.org) packages with keyword "orange3 add-on"
    # can be installed using the Orange Add-on Manager
    'orange3 add-on',
    'spectroscopy',
    'infrared'
]

if __name__ == '__main__':
    setup(
        name="Orange-Infrared",
        description='',
        author='Canadian Light Source, Biolab UL, Soleil, Elettra',
        author_email='marko.toplak@gmail.com',
        version="0.0.7",
        packages=find_packages(),
        install_requires=[
            'Orange3>=3.3.8',
            'scipy>=0.14.0',
            'spectral>=0.18',
            'opusFC>=1.0.0b1',
        ],
        entry_points=ENTRY_POINTS,
        keywords=KEYWORDS,
        namespace_packages=['orangecontrib'],
        include_package_data=True,
        zip_safe=False,
        url="https://github.com/markotoplak/orange-infrared"
    )

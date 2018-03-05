#!/usr/bin/env python

from os import walk, path

import os
from setuptools import setup, find_packages, Command
import subprocess
import sys

PACKAGES = find_packages()

PACKAGE_DATA = {
#    'orangecontrib.example': ['tutorials/*.ows'],
#    'orangecontrib.example.widgets': ['icons/*'],
}

README_FILE = os.path.join(os.path.dirname(__file__), 'README.pypi')
LONG_DESCRIPTION = open(README_FILE).read()

DATA_FILES = [
    # Data files that will be installed outside site-packages folder
]

ENTRY_POINTS = {
    # Entry points that marks this package as an orange add-on. If set, addon will
    # be shown in the add-ons manager even if not published on PyPi.
    'orange3.addon': (
        'spectroscopy = orangecontrib.spectroscopy',
    ),

    # Entry point used to specify packages containing tutorials accessible
    # from welcome screen. Tutorials are saved Orange Workflows (.ows files).
    'orange.widgets.tutorials': (
        # Syntax: any_text = path.to.package.containing.tutorials
        'infraredtutorials = orangecontrib.spectroscopy.tutorials',
    ),

    # Entry point used to specify packages containing widgets.
    'orange.widgets': (
        # Syntax: category name = path.to.package.containing.widgets
        # Widget category specification can be seen in
        #    orangecontrib/example/widgets/__init__.py
        'Spectroscopy = orangecontrib.spectroscopy.widgets',
    ),

    # Register widget help
    "orange.canvas.help": (
        'html-index = orangecontrib.spectroscopy.widgets:WIDGET_HELP_PATH',)

}

KEYWORDS = [
    # [PyPi](https://pypi.python.org) packages with keyword "orange3 add-on"
    # can be installed using the Orange Add-on Manager
    'orange3 add-on',
    'spectroscopy',
    'infrared'
]

class CoverageCommand(Command):
    """A setup.py coverage subcommand developers can run locally."""
    description = "run code coverage"
    user_options = []
    initialize_options = finalize_options = lambda self: None

    def run(self):
        """Check coverage on current workdir"""
        sys.exit(subprocess.call(r'''
        coverage run --source=orangecontrib.spectroscopy -m unittest
        echo; echo
        coverage report --omit="*/tests/*"
        coverage html --omit="*/tests/*" &&
            { echo; echo "See also: file://$(pwd)/htmlcov/index.html"; echo; }
        ''', shell=True, cwd=os.path.dirname(os.path.abspath(__file__))))


TEST_SUITE = "orangecontrib.spectroscopy.tests.suite"


def include_documentation(local_dir, install_dir):
    global DATA_FILES
    if 'bdist_wheel' in sys.argv and not path.exists(local_dir):
        print("Directory '{}' does not exist. "
              "Please build documentation before running bdist_wheel."
              .format(path.abspath(local_dir)))
        sys.exit(0)

    doc_files = []
    for dirpath, dirs, files in walk(local_dir):
        doc_files.append((dirpath.replace(local_dir, install_dir),
                          [path.join(dirpath, f) for f in files]))
    DATA_FILES.extend(doc_files)


if __name__ == '__main__':
    
    cmdclass = {
       'coverage': CoverageCommand,
    }

    include_documentation('doc/build/html', 'help/orange-spectroscopy')

    setup(
        name="Orange-Spectroscopy",
        description='Extends Orange to handle spectral and hyperspectral analysis.',
        long_description=LONG_DESCRIPTION,
        author='Canadian Light Source, Biolab UL, Soleil, Elettra',
        author_email='marko.toplak@gmail.com',
        version="0.3.5",
        packages=PACKAGES,
        package_data=PACKAGE_DATA,
        data_files=DATA_FILES,
        install_requires=[
            'Orange3>=3.7.0',
            'scipy>=0.14.0',
            'spectral>=0.18',
            'opusFC>=1.1.0',
            'serverfiles>=0.2',
            'AnyQt>=0.0.6',
            'pyqtgraph>=0.10.0',
            'colorcet',
            'h5py',
        ],
        entry_points=ENTRY_POINTS,
        keywords=KEYWORDS,
        namespace_packages=['orangecontrib'],
        test_suite=TEST_SUITE,
        include_package_data=True,
        zip_safe=False,
        url="https://github.com/markotoplak/orange-spectroscopy",
        cmdclass=cmdclass,
    )

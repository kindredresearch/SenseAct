from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
import sys, subprocess


if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))


setup(name='senseact',
      packages=[package for package in find_packages()
                if package.startswith('senseact') or package.startswith('test')],
      install_requires=[
          'gym',  # 0.10.5
          'matplotlib',
          'numpy',
          'opencv-python>=3.3.1',
          'psutil',
          'pyserial',
      ],
      description='Kindred SenseAct framework',
      author='Kindred AI',
      url='https://github.com/kindredresearch/SenseAct',
      author_email='',
      version='0.0.1')

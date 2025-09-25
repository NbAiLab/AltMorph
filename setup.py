#!/usr/bin/env python3
"""Setup script for AltMorph package."""

from setuptools import setup, find_packages
import os

# Read the README file
current_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='altmorph',
    version='0.1.0',
    author='Pere',  # Update with your name
    author_email='your.email@example.com',  # Update
    description='Context-aware Norwegian morphological alternative generator',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/altmorph',  # Update
    py_modules=['altmorph'],
    python_requires='>=3.8',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'altmorph=altmorph:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9', 
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    keywords='morphology norwegian nlp linguistics alternatives ordbank pos bert',
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/altmorph/issues',
        'Source': 'https://github.com/yourusername/altmorph',
    },
)

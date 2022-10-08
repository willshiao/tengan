"""
gutils provides a collection of useful functions/classes for data analysis and ML.
"""
from setuptools import setup

setup(
    name='gutils',
    version='0.1.9',
    author='XXXXXX',
    author_email='XX@XX',
    packages=['gutils'],
    scripts=[],
    url='XXXXXX',
    license='LICENSE',
    description='',
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    install_requires=[
        "scikit-learn",
        "numpy",
        "seaborn",
        "pytest",
        "xgboost",
        "scipy",
        'matplotlib'
    ],
)

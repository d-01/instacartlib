from setuptools import setup

setup(
    name='instacartlib',
    version='0.1',
    # * Automatic package discoverty:
    #   find_packages(include=['instacartlib', 'instacartlib.*'])
    packages=['instacartlib', 'instacartlib.feature_extractors'],
    long_description="Library to work with instacart dataset.",
)

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name='orca',
    version='0.1.3',
    description='Korean typo correcter',
    author='MJ Jang',
    install_requires=required,
    packages=find_packages(exclude=['docs', 'tests', 'tmp', 'data', '__pycache__']),
    python_requires='>=3',
    package_data={'orca': ['resources/*']},
    zip_safe=False,
    include_package_data=True
)

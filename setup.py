from setuptools import setup, find_packages

setup(
    name='pyksolver',
    version='0.1',
    packages=find_packages(),
    install_requires=[
    ],
    entry_points={
        'console_scripts': [
            'pyksolver=main:main',
        ],
    },
    url='https://gitlab.com/ayoub_ana/pyksolver',
    license='MIT',
    author='Ayoub Aouina',
    author_email='ayoubaouina@outlook.com',
    description='PyKSolver'
)

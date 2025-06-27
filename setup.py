from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='pyksolver',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'joblib',
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
    description='PyKSolver',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
    ],
)

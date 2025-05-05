from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='HBV_Lab',
    version='0.2.1',
    packages=find_packages(include=['HBV_python', 'HBV_python.*']),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'tqdm',
    ],
    author='Abdalla Mohammed',
    author_email='abdalla.mohammed.ox@gmail.com',
    description='An intuitive, object-oriented and user-friendly Python implementation of a lumped conceptual HBV hydrological model for educational and research purposes.',
    long_description=long_description,
    long_description_content_type='text/markdown',  # Requires README.md in Markdown
    url='https://github.com/abdallaox/HBV_python_implementation',
    license='MIT',  # Explicit license field
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Hydrology',
    ],
    python_requires='>=3.7',
    keywords='hydrology HBV-model rainfall-runoff hydrological-modelling',
)
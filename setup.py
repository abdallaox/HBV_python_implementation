from setuptools import setup, find_packages

setup(
    name='HBV_package',
    version='0.1.0',
    packages=find_packages(include=['HBV_package', 'HBV_package.*']),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'tqdm',
    ],
    author='Abdalla Mohammed',
    author_email='abdalla.mohammed.ox@gmail.com',
    description='HBV is a simple conceptual hydrological model that simulates the main hydrological processes related to snow, soil, groundwater, and routing.',
    long_description='HBV is a simple conceptual hydrological model that simulates the main hydrological processes related to snow, soil, groundwater, and routing. There are many software packages and off-the-shelf products that implement it. I’ve been experimenting with the model lately and—in an endeavour to better understand the logic behind it—I decided to implement my own version in Python, following an intuitive object-oriented programming approach. This can be flexibly used for different modelling tasks, but can also be used in a classroom setup to explain hydrological concepts (processes, calibration, uncertainty analysis, etc.).',
    long_description_content_type='text/plain',  # You can use 'text/markdown' if you use markdown
    url='https://github.com/abdallaox/HBV_python_implementation',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)

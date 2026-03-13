from setuptools import setup, find_packages

setup(
    name="mlip-platform",
    version="0.2.0",
    description="CLI platform for running MLIP-based MD, NEB, and Benchmarking",
    author="Juan M Arce-Ramos",
    author_email="juan.arce@ihpc.a-star.edu.sg",
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "typer[all]",
        "ase",
        "pandas",
        "matplotlib",
        "scipy",
        "asetools",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    entry_points={
        'console_scripts': [
            'md = mlip_platform.cli.commands.md:app',
            'neb = mlip_platform.cli.commands.neb:app',
            'autoneb = mlip_platform.cli.commands.autoneb:app',
            'autoneb-results = mlip_platform.cli.commands.autoneb_results:app',
            'benchmark = mlip_platform.cli.commands.benchmark:app',
            'optimize = mlip_platform.cli.commands.optimize:app',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

from setuptools import setup, find_packages

setup(
    name="mlip-platform",
    version="0.1.0",
    description="CLI platform for running MLIP-based MD, NEB, and Benchmarking",
    author="Your Name",
    author_email="your.email@example.com",
    package_dir={"": "src"},  # <-- This tells setuptools where your code is
    packages=find_packages(where="src"),  # <-- Looks inside /src
    install_requires=[
        "typer[all]",
        "ase"
    ],
    entry_points={
        'console_scripts': [
            'md = mlip_platform.cli.commands.md:app',
            'neb = mlip_platform.cli.commands.neb:app',
            'benchmark = mlip_platform.cli.commands.benchmark:app',
            'optimize = mlip_platform.cli.commands.optimize:app',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

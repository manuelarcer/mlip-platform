from setuptools import setup, find_packages

setup(
    name="mlip_platform_cli",
    version="0.1.0",
    description="CLI for MD, NEB and benchmarking with MACE/SevenNet",
    author="Your Name",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "typer>=0.6.0",
        "ase",
        # Optionally list mace, sevenn if you want pip to pull them in
    ],
    entry_points={
        "console_scripts": [
            "mlip=mlip_platform.cli.main:app",
            "md=mlip_platform.cli.commands.md:app",
            "neb=mlip_platform.cli.commands.neb:app",
            "benchmark=mlip_platform.cli.commands.benchmark:app",
        ],
    },
    python_requires=">=3.7",
)

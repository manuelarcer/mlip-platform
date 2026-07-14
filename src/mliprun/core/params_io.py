"""I/O utilities for parameter files and endpoint results."""
from pathlib import Path


def write_parameters_file(path: Path, title: str, params: dict) -> None:
    """Write a key-value parameter file with a title header.

    Parameters
    ----------
    path : Path
        Output file path.
    title : str
        Title line (e.g. "NEB Run Parameters").
    params : dict
        Ordered mapping of parameter names to values.
        Keys become the left column, values the right column.
    """
    separator = "=" * max(len(title), 25)
    with open(path, "w") as f:
        f.write(f"{title}\n")
        f.write(f"{separator}\n")
        for key, value in params.items():
            f.write(f"{key:<23}{value}\n")


def write_endpoint_results(path: Path, endpoint_results: dict) -> None:
    """Write endpoint optimization results to a text file.

    Parameters
    ----------
    path : Path
        Output file path.
    endpoint_results : dict
        Dictionary with 'initial', 'final', 'reaction_energy', and
        'similarity' keys as returned by CustomNEB.optimize_endpoints().
    """
    with open(path, "w") as f:
        f.write("Endpoint Optimization Results\n")
        f.write("==============================\n\n")

        for label in ("initial", "final"):
            data = endpoint_results[label]
            f.write(f"{label.capitalize()} Structure:\n")
            f.write(f"  Energy before: {data['energy_before']:.6f} eV\n")
            f.write(f"  Energy after:  {data['energy_after']:.6f} eV\n")
            f.write(f"  Energy change: {data['energy_change']:.6f} eV\n")
            f.write(f"  Steps:         {data['steps']}\n")
            f.write(f"  Converged:     {data['converged']}\n\n")

        f.write(f"Reaction energy: {endpoint_results['reaction_energy']:.6f} eV\n\n")

        sim = endpoint_results["similarity"]
        f.write("Similarity Check:\n")
        f.write(f"  Average displacement: {sim['avg_displacement']:.3f} Ang\n")
        f.write(f"  Max displacement:     {sim['max_displacement']:.3f} Ang (atom {sim['max_disp_atom']})\n")
        f.write(f"  Min displacement:     {sim['min_displacement']:.3f} Ang\n")
        f.write(f"  Energy difference:    {sim['energy_diff']:.6f} eV\n")
        f.write(f"  Structures similar:   {sim['is_similar']}\n")
        if sim["warning_reasons"]:
            f.write("  Warning reasons:\n")
            for reason in sim["warning_reasons"]:
                f.write(f"    - {reason}\n")

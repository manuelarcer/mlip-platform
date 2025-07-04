import importlib.util


def get_available_mlip():
    """Check which MLIP is available in the environment."""
    if importlib.util.find_spec("mace") is not None:
        return "mace"
    elif importlib.util.find_spec("sevenn") is not None:
        return "sevennet"
    else:
        raise ImportError("Neither MACE nor SevenNet is installed in this environment.")


def load_calculator(model=None):
    """
    Load the appropriate MLIP calculator depending on the environment.

    Args:
        model (str, optional): Model name or path. Ignored for MACE if not needed.

    Returns:
        ASE calculator instance.
    """
    mlip = get_available_mlip()

    if mlip == "sevennet":
        from sevenn.calculator import SevenNetCalculator
        return SevenNetCalculator(model or "7net-mf-ompa", modal="mpa")

    elif mlip == "mace":
        from mace.calculators import MACECalculator
        return MACECalculator()

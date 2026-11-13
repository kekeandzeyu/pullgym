from pullgym.families.base import Episode, Family, Quantity
from pullgym.families.decay_background import DecayBackground
from pullgym.families.line_fit import LineFit
from pullgym.families.voltmeter_gain import VoltmeterGain

FAMILIES: dict[str, Family] = {f.name: f for f in (LineFit(), DecayBackground(), VoltmeterGain())}


def get_family(name: str) -> Family:
    try:
        return FAMILIES[name]
    except KeyError:
        raise KeyError(f"unknown family {name!r}; known: {', '.join(FAMILIES)}") from None


__all__ = ["Episode", "Family", "Quantity", "FAMILIES", "get_family"]

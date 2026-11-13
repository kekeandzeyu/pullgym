from pullgym.families.base import Episode, Family, Quantity
from pullgym.families.correlated_norm import CorrelatedNorm
from pullgym.families.counting_rate import CountingRate
from pullgym.families.dead_time import DeadTime
from pullgym.families.decay_background import DecayBackground
from pullgym.families.errors_xy import ErrorsXY
from pullgym.families.line_fit import LineFit
from pullgym.families.low_count_rate import LowCountRate
from pullgym.families.no_signal import NoSignal
from pullgym.families.peak_background import PeakBackground
from pullgym.families.underestimated_errors import UnderestimatedErrors
from pullgym.families.voltmeter_gain import VoltmeterGain
from pullgym.families.weighted_mean import WeightedMean

_ALL = (
    # tier 1
    LineFit(),
    WeightedMean(),
    CountingRate(),
    # tier 2
    DecayBackground(),
    PeakBackground(),
    LowCountRate(),
    # tier 3
    VoltmeterGain(),
    UnderestimatedErrors(),
    ErrorsXY(),
    CorrelatedNorm(),
    DeadTime(),
    # tier 4
    NoSignal(),
)
FAMILIES: dict[str, Family] = {f.name: f for f in _ALL}


def get_family(name: str) -> Family:
    try:
        return FAMILIES[name]
    except KeyError:
        raise KeyError(f"unknown family {name!r}; known: {', '.join(FAMILIES)}") from None


__all__ = ["Episode", "Family", "Quantity", "FAMILIES", "get_family"]

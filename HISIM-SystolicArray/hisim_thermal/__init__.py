"""hisim_thermal: steady-state 3D finite-volume thermal solver shared by HISIM-IMC and HISIM-SystolicArray.

Pipeline (plan §3.4): neutral input -> stack -> grid -> region maps -> materials -> power -> operator -> solve -> outputs.
"""
from .config import VERSION, get_config
from .neutral_input import Link, ThermalInput, Tile

__version__ = VERSION
__all__ = ['get_config', 'ThermalInput', 'Tile', 'Link', 'run', '__version__']


def run(ti, cfg=None, out_dir=None, **kw):
    from .driver import run as _run
    return _run(ti, cfg or get_config(), out_dir, **kw)

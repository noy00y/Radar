"""Centralised physical constants and project-wide conventions.

Every module imports constants from here so that there is a single source of
truth. Conventions are documented once here and in the README
("Signal and array conventions").
"""

from __future__ import annotations

#: Speed of light in vacuum [m/s] (exact, SI definition).
SPEED_OF_LIGHT: float = 299_792_458.0

#: Boltzmann constant [J/K] (only used for optional link-budget helpers).
BOLTZMANN: float = 1.380649e-23

# ---------------------------------------------------------------------------
# Conventions (strings are used in configs and metadata so they are explicit)
# ---------------------------------------------------------------------------

#: IF = s_tx(t) * conj(s_rx(t)).  Under this convention a target at range R
#: produces a POSITIVE beat frequency f_b = S*tau and a receding target
#: (dR/dt > 0) produces a POSITIVE slow-time phase rate nu = +2 v_r / lambda.
MIXER_TX_CONJ_RX: str = "tx_times_conj_rx"

#: IF = s_rx(t) * conj(s_tx(t)): the complex conjugate of the above. All
#: frequencies flip sign. Supported so a hardware capture whose I/Q sign is
#: verified to be opposite can be handled without hacks.
MIXER_RX_CONJ_TX: str = "rx_times_conj_tx"

MIXER_CONVENTIONS = (MIXER_TX_CONJ_RX, MIXER_RX_CONJ_TX)

#: Radial velocity sign convention used everywhere in this package:
#: v_r = dR/dt, POSITIVE = RECEDING (range increasing).
RADIAL_VELOCITY_CONVENTION: str = "v_r = dR/dt, positive receding"

#: Space-time snapshot stacking: index = n * M + m (pulse-major, channel fastest),
#: i.e. x = vec(Y_r) with Y_r[m, n]  ->  s = kron(b_time, a_space).
STACKING_ORDER: str = "pulse_major_channel_fastest"

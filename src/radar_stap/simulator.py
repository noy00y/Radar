"""Multichannel FMCW scene simulator with geometrically coherent ground clutter.

Every return (target, discrete reflector, clutter scatterer) is a point
scatterer with a world position and velocity. For every scatterer s, RX m and
chirp n we compute the *exact* bistatic path length

    L[s, m, n] = |p_s(t_n) - p_tx(n)(t_n)| + |p_s(t_n) - p_rx_m(t_n)|

at the ramp start t_n of chirp n (TX chosen by the TDM schedule) and its rate
Ldot. Inside the chirp the delay is tau(t_f) = (L + Ldot * t_f) / c and the IF
sample is generated with :func:`radar_stap.waveforms.if_phase`. Consequences:

* fast-time and slow-time phase come from one delay model (no mixing of
  inconsistent phase models), including range-Doppler coupling;
* range migration across the CPI is modelled exactly (path recomputed per chirp);
* array phase differences come from actual element positions (no far-field or
  narrowband approximation in the *simulator*; the processor's steering vectors
  use the far-field model, and tests quantify the agreement);
* platform motion makes stationary clutter Doppler depend on angle, producing
  the space-time clutter ridge; all channels/chirps of one scatterer share one
  complex reflectivity, so clutter has coherent space-time structure.

Approximations (documented in README): point scatterers, constant velocities
over the CPI, linear delay *within* a chirp, no intrinsic clutter motion, no
multipath/occlusion, no IF filter / HPF / ADC nonlinearity, isotropic
scatterers, element pattern = two-way cos^p of the off-boresight angle.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field

import numpy as np

from .config import SimulationConfig, element_positions_m, sim_derived, wavelength_for
from .constants import MIXER_TX_CONJ_RX, SPEED_OF_LIGHT
from .waveforms import chirp_period, complex_noise, fast_time_axis, if_phase, ramp_start_times

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def rotation_body_to_world(yaw_deg: float, depression_deg: float) -> np.ndarray:
    """R = Rz(yaw) @ Rx(-depression). Body: x array axis, y boresight, z up."""
    y = np.deg2rad(yaw_deg)
    a = -np.deg2rad(depression_deg)
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    return rz @ rx


@dataclass
class RadarGeometry:
    """Platform + array geometry (all metres / seconds, world frame unless noted)."""

    position: np.ndarray  # platform reference point at t = 0 (world)
    velocity: np.ndarray  # platform velocity (world)
    rotation: np.ndarray  # body -> world
    rx_body: np.ndarray  # (M, 3)
    tx_body: np.ndarray  # (Ptx, 3)
    tx_schedule: list[int]
    wavelength: float
    pattern_exponent: float

    @classmethod
    def from_config(cls, cfg: SimulationConfig) -> "RadarGeometry":
        lam = wavelength_for(cfg.radar)
        rx, tx = element_positions_m(cfg.array, lam)
        return cls(
            position=np.asarray(cfg.platform.position_m, float),
            velocity=np.asarray(cfg.platform.velocity_mps, float),
            rotation=rotation_body_to_world(cfg.platform.yaw_deg, cfg.platform.depression_deg),
            rx_body=rx,
            tx_body=tx,
            tx_schedule=list(cfg.array.tx_schedule),
            wavelength=lam,
            pattern_exponent=cfg.platform.element_pattern_exponent,
        )

    def to_body(self, vec_world: np.ndarray) -> np.ndarray:
        return np.asarray(vec_world) @ self.rotation  # R^T v for row vectors

    def line_of_sight(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(range (S,), unit LOS world (S,3)) from the platform reference at t=0."""
        d = np.asarray(positions, float) - self.position
        r = np.linalg.norm(d, axis=-1)
        return r, d / r[..., None]

    def pattern_gain(self, positions: np.ndarray) -> np.ndarray:
        """Two-way amplitude element pattern max(u_y_body, 0)^p."""
        _, u = self.line_of_sight(positions)
        uy = self.to_body(u)[..., 1]
        return np.clip(uy, 0.0, None) ** self.pattern_exponent


# ---------------------------------------------------------------------------
# Scatterers and ground truth
# ---------------------------------------------------------------------------


@dataclass
class ScattererSet:
    kind: str  # "target" | "reflector" | "clutter"
    positions: np.ndarray  # (S, 3) world at t = 0
    velocities: np.ndarray  # (S, 3) world
    amplitudes: np.ndarray  # (S,) complex scattering amplitude A_s
    powers: np.ndarray  # (S,) expected |A_s|^2 (for clairvoyant covariance)
    names: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return int(self.positions.shape[0])

    def subset(self, idx) -> "ScattererSet":
        idx = np.asarray(idx)
        names = [self.names[i] for i in np.atleast_1d(idx)] if self.names else []
        return ScattererSet(
            self.kind, self.positions[idx], self.velocities[idx], self.amplitudes[idx], self.powers[idx], names
        )


@dataclass
class TargetTruth:
    name: str
    kind: str
    range_m: float  # |p_t - p_platform| at t = 0
    radial_velocity_mps: float  # (v_t - v_p) . u_LOS, positive receding
    angle_deg: float  # array cone angle: asin(u_body_x)
    elevation_deg: float  # asin(u_body_z)
    snr_db: float
    position_m: list[float]
    velocity_mps: list[float]

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class Scene:
    geometry: RadarGeometry
    targets: ScattererSet
    reflectors: ScattererSet
    clutter: ScattererSet
    truths: list[TargetTruth]
    seed: int

    def all_sets(self) -> list[ScattererSet]:
        return [self.targets, self.reflectors, self.clutter]


def _empty_set(kind: str) -> ScattererSet:
    z3 = np.zeros((0, 3))
    return ScattererSet(kind, z3, z3.copy(), np.zeros(0, complex), np.zeros(0), [])


def rng_streams(seed: int) -> dict[str, np.random.Generator]:
    """Independent RNG streams so that e.g. changing noise does not move clutter."""
    names = ("geometry", "clutter_reflectivity", "target_phase", "noise", "spare")
    seqs = np.random.SeedSequence(seed).spawn(len(names))
    return {n: np.random.default_rng(s) for n, s in zip(names, seqs)}


def _direction_from_range_angle(geom: RadarGeometry, range_m: float, angle_deg: float, height_m: float) -> np.ndarray:
    """World position at given slant range, array cone angle and world height."""
    ux = np.sin(np.deg2rad(angle_deg))
    q = np.sqrt(max(1.0 - ux**2, 0.0))
    w = (height_m - geom.position[2]) / range_m
    # body (y, z) = q (cos psi, sin psi); world z of direction = q sin(psi - dep)
    dep = -np.arcsin(np.clip(geom.rotation[2, 1], -1, 1))  # recover depression from R
    if abs(w) > q + 1e-12:
        raise ValueError(
            f"cannot place scatterer at range {range_m} m, angle {angle_deg} deg, height {height_m} m"
        )
    psi = dep + np.arcsin(np.clip(w / q, -1, 1))
    d_body = np.array([ux, q * np.cos(psi), q * np.sin(psi)])
    return geom.position + range_m * (geom.rotation @ d_body)


def _target_set(cfg_list, geom: RadarGeometry, noise_power: float, rng: np.random.Generator, kind: str):
    if not cfg_list:
        return _empty_set(kind), []
    pos, vel, amp, pw, names, truths = [], [], [], [], [], []
    for t in cfg_list:
        if t.position_m is not None:
            p = np.asarray(t.position_m, float)
        else:
            p = _direction_from_range_angle(geom, t.range_m, t.angle_deg, t.height_m)
        r, u = geom.line_of_sight(p[None])
        r, u = float(r[0]), u[0]
        if t.velocity_mps is not None:
            v = np.asarray(t.velocity_mps, float)
        elif t.radial_velocity_mps is not None:
            uh = np.array([u[0], u[1], 0.0])
            nh = np.linalg.norm(uh)
            if nh < 1e-6:
                raise ValueError(f"target {t.name}: LOS is vertical, cannot set radial velocity")
            uh /= nh
            c = (t.radial_velocity_mps + geom.velocity @ u) / (uh @ u)
            v = c * uh
        else:
            v = np.zeros(3)
        power = noise_power * 10 ** (t.snr_db / 10)
        phase = rng.uniform(0, 2 * np.pi) if t.phase_deg is None else np.deg2rad(t.phase_deg)
        pos.append(p)
        vel.append(v)
        amp.append(np.sqrt(power) * np.exp(1j * phase))
        pw.append(power)
        names.append(t.name)
        ub = geom.to_body(u)
        truths.append(
            TargetTruth(
                name=t.name,
                kind=kind,
                range_m=r,
                radial_velocity_mps=float((v - geom.velocity) @ u),
                angle_deg=float(np.rad2deg(np.arcsin(np.clip(ub[0], -1, 1)))),
                elevation_deg=float(np.rad2deg(np.arcsin(np.clip(ub[2], -1, 1)))),
                snr_db=t.snr_db,
                position_m=p.tolist(),
                velocity_mps=v.tolist(),
            )
        )
    return (
        ScattererSet(kind, np.array(pos), np.array(vel), np.array(amp), np.array(pw), names),
        truths,
    )


def draw_clutter_reflectivity(powers: np.ndarray, model: str, rng: np.random.Generator) -> np.ndarray:
    n = powers.size
    if model == "rayleigh":
        g = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    elif model == "constant_amplitude":
        g = np.exp(1j * rng.uniform(0, 2 * np.pi, n))
    else:
        raise ValueError(f"unknown reflectivity model {model!r}")
    return np.sqrt(powers) * g


def _clutter_set(cfg: SimulationConfig, geom: RadarGeometry, streams) -> ScattererSet:
    c = cfg.clutter
    if not c.enabled:
        return _empty_set("clutter")
    rng = streams["geometry"]
    n = c.num_scatterers
    r = rng.uniform(c.range_min_m, c.range_max_m, n)
    az = np.deg2rad(rng.uniform(c.azimuth_min_deg, c.azimuth_max_deg, n))
    dz = c.ground_height_m - geom.position[2]
    rho = np.sqrt(np.maximum(r**2 - dz**2, 0.0))
    yaw = np.deg2rad(cfg.platform.yaw_deg)
    dirs = np.stack([np.sin(az), np.cos(az)], axis=1)  # horizontal, relative to heading
    rot2 = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    xy = geom.position[:2] + rho[:, None] * (dirs @ rot2.T)
    pos = np.column_stack([xy, np.full(n, c.ground_height_m)])
    gain = geom.pattern_gain(pos)
    rng_r, _ = geom.line_of_sight(pos)
    w = gain**2 * (rng_r / c.range_min_m) ** (-c.range_power_exponent)
    total = cfg.radar.noise_power * 10 ** (c.cnr_db / 10)
    powers = total * w / w.sum() if w.sum() > 0 else np.zeros(n)
    amps = draw_clutter_reflectivity(powers, c.reflectivity, streams["clutter_reflectivity"])
    return ScattererSet("clutter", pos, np.zeros((n, 3)), amps, powers, [])


def build_scene(cfg: SimulationConfig) -> Scene:
    """Deterministically build the scene geometry and scatterer amplitudes from ``cfg.seed``."""
    geom = RadarGeometry.from_config(cfg)
    streams = rng_streams(cfg.seed)
    targets, t_truth = _target_set(cfg.targets, geom, cfg.radar.noise_power, streams["target_phase"], "target")
    refl, r_truth = _target_set(
        cfg.stationary_reflectors, geom, cfg.radar.noise_power, streams["target_phase"], "reflector"
    )
    clutter = _clutter_set(cfg, geom, streams)
    return Scene(geom, targets, refl, clutter, t_truth + r_truth, cfg.seed)


# ---------------------------------------------------------------------------
# Signal synthesis
# ---------------------------------------------------------------------------


def path_lengths(
    scat: ScattererSet, geom: RadarGeometry, chirp_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Two-way path L[s, m, n] and its rate Ldot[s, m, n] at the ramp start of each chirp."""
    P = len(geom.tx_schedule)
    tx_idx = np.array([geom.tx_schedule[n % P] for n in range(chirp_times.size)])
    t = chirp_times[None, :, None]  # (1, N, 1)
    ps = scat.positions[:, None, :] + scat.velocities[:, None, :] * t  # (S, N, 3)
    plat = geom.position[None, :] + geom.velocity[None, :] * chirp_times[:, None]  # (N, 3)
    tx_w = plat + (geom.tx_body[tx_idx] @ geom.rotation.T)  # (N, 3)
    rx_w = plat[None, :, :] + (geom.rx_body @ geom.rotation.T)[:, None, :]  # (M, N, 3)
    dvel = scat.velocities - geom.velocity  # (S, 3) relative velocity
    d_tx = ps - tx_w[None]  # (S, N, 3)
    r_tx = np.linalg.norm(d_tx, axis=-1)  # (S, N)
    rdot_tx = np.einsum("snk,sk->sn", d_tx, dvel) / r_tx
    d_rx = ps[:, None, :, :] - rx_w[None]  # (S, M, N, 3)
    r_rx = np.linalg.norm(d_rx, axis=-1)  # (S, M, N)
    rdot_rx = np.einsum("smnk,sk->smn", d_rx, dvel) / r_rx
    return r_tx[:, None, :] + r_rx, rdot_tx[:, None, :] + rdot_rx


def mixer_amplitude(amplitudes: np.ndarray, mixer: str) -> np.ndarray:
    """Complex coefficient multiplying exp(j*if_phase): conj(A) for tx*conj(rx)."""
    return np.conj(amplitudes) if mixer == MIXER_TX_CONJ_RX else amplitudes


def unit_raw_responses(
    scat: ScattererSet, cfg: SimulationConfig, geom: RadarGeometry | None = None
) -> np.ndarray:
    """Raw IF response of each scatterer with unit coefficient: shape (S, M, N, K), complex128."""
    geom = geom or RadarGeometry.from_config(cfg)
    radar = cfg.radar
    t_n = ramp_start_times(radar)
    t_f = fast_time_axis(radar)
    L, Ld = path_lengths(scat, geom, t_n)
    tau = (L[..., None] + Ld[..., None] * t_f) / SPEED_OF_LIGHT
    phi = if_phase(tau, t_f, radar.carrier_frequency_hz, radar.chirp_slope_hz_per_s, radar.mixer_convention)
    return np.exp(1j * phi)


def synthesize(
    scat: ScattererSet,
    cfg: SimulationConfig,
    geom: RadarGeometry | None = None,
    coefficients: np.ndarray | None = None,
    chunk: int = 64,
) -> np.ndarray:
    """Coherent sum of scatterer returns, (M, N, K) complex128, processed in chunks."""
    geom = geom or RadarGeometry.from_config(cfg)
    M, N, K = geom.rx_body.shape[0], cfg.radar.num_chirps, cfg.radar.num_adc_samples
    out = np.zeros((M, N, K), complex)
    if scat.size == 0:
        return out
    coef = mixer_amplitude(scat.amplitudes, cfg.radar.mixer_convention) if coefficients is None else coefficients
    for s0 in range(0, scat.size, chunk):
        idx = np.arange(s0, min(s0 + chunk, scat.size))
        resp = unit_raw_responses(scat.subset(idx), cfg, geom)
        out += np.tensordot(coef[idx], resp, axes=1)
    return out


@dataclass
class SimulationResult:
    """Simulated CPI. ``data`` = targets + reflectors + clutter + noise, layout X[rx, chirp, sample]."""

    data: np.ndarray
    components: dict[str, np.ndarray]
    scene: Scene
    config: SimulationConfig
    metadata: dict

    def save_npz(self, path, include_components: bool = True) -> None:
        arrays = {"data": self.data}
        if include_components:
            arrays.update({f"component_{k}": v for k, v in self.components.items()})
        arrays["clutter_positions"] = self.scene.clutter.positions
        arrays["clutter_powers"] = self.scene.clutter.powers
        np.savez_compressed(path, **arrays)


def simulate(cfg: SimulationConfig, scene: Scene | None = None, noise_seed: int | None = None) -> SimulationResult:
    """Simulate one CPI. Components are kept separately for evaluation."""
    scene = scene or build_scene(cfg)
    geom = scene.geometry
    radar = cfg.radar
    M, N, K = geom.rx_body.shape[0], radar.num_chirps, radar.num_adc_samples
    comps = {
        "targets": synthesize(scene.targets, cfg, geom),
        "reflectors": synthesize(scene.reflectors, cfg, geom),
        "clutter": synthesize(scene.clutter, cfg, geom),
    }
    noise_rng = rng_streams(cfg.seed)["noise"] if noise_seed is None else np.random.default_rng(noise_seed)
    noise = complex_noise(noise_rng, (M, N, K), radar.noise_power, dtype=np.complex128)
    if radar.sampling == "real":
        # Real ADC: keep the in-phase channel only. Noise power per real sample = noise_power/2.
        comps = {k: v.real for k, v in comps.items()}
        noise = noise.real
    comps["noise"] = noise
    data = sum(comps.values())
    dtype = np.complex64 if radar.sampling == "complex" else np.float32
    d = sim_derived(cfg)
    meta = {
        "layout": "X[rx_channel, chirp_index, adc_sample_index]",
        "chirp_order": "transmission order; TX of chirp n = tx_schedule[n % len(tx_schedule)]",
        "mixer_convention": radar.mixer_convention,
        "sampling": radar.sampling,
        "seed": cfg.seed,
        "num_clutter_scatterers": scene.clutter.size,
        "derived": d.as_dict(),
        "range_migration_bins_max": _max_range_migration_bins(scene, cfg),
        "truth": [t.as_dict() for t in scene.truths],
    }
    return SimulationResult(
        data=data.astype(dtype),
        components={k: v.astype(dtype) for k, v in comps.items()},
        scene=scene,
        config=cfg,
        metadata=meta,
    )


def _max_range_migration_bins(scene: Scene, cfg: SimulationConfig) -> float:
    """Largest |v_r| * T_CPI / range_bin over targets (diagnostic; migration is modelled exactly)."""
    d = sim_derived(cfg)
    if not scene.truths:
        return 0.0
    vmax = max(abs(t.radial_velocity_mps) for t in scene.truths)
    return float(vmax * d.cpi_duration_s / d.range_bin_spacing_m)


def chirp_timing_summary(cfg: SimulationConfig) -> dict:
    return {"chirp_period_s": chirp_period(cfg.radar), "wavelength_m": wavelength_for(cfg.radar)}

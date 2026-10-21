"""DCA1000EVM raw-ADC decoding with an *explicit*, fully specified binary format.

Nothing about the file layout is guessed. Every field of :class:`AdcFormat`
must be supplied (from your capture configuration / verification) and the
decoder fails with an actionable message otherwise. The frame count may be
derived from the file size only when the *per-frame* layout is fully known and
the size divides exactly (or the trailing partial frame is explicitly allowed).

Supported layouts (each must be selected explicitly)
---------------------------------------------------
* words: 16-bit, ``byte_order`` little/big.
* ``sign_handling``:
    - ``int16_twos_complement``: each word is a two's-complement int16 (device
      already sign-extended the ``adc_bits`` value). Values outside the
      ``adc_bits`` range are counted in the report (a strong hint that the
      packing assumption is wrong).
    - ``sign_extend_adc_bits``: the low ``adc_bits`` of each word hold a two's-
      complement value that must be sign-extended here.
* ``data_type``: ``complex`` or ``real``.
* ``complex_packing`` (complex only):
    - ``iq_interleaved``:     A0 B0 A1 B1 ...
    - ``two_sample_blocks``:  A0 A1 B0 B1 A2 A3 B2 B3 ...  (pairs of samples)
  with (A, B) = (I, Q) for ``iq_order: I_first`` and (Q, I) for ``Q_first``.
  The TI DCA1000/mmWave Studio documentation for xWR16xx/18xx 2-lane LVDS
  captures describes a two-sample block pattern; whether your file is I- or
  Q-first and whether channels are interleaved depends on the device, LVDS
  configuration and the tool that wrote the file -- verify before use.
* ``channel_layout``:
    - ``rx_planar_per_chirp``:       per chirp: all samples of RX0, then RX1, ...
    - ``rx_interleaved_per_sample``: per chirp: sample 0 of RX0..RX(M-1), sample 1 ...
* chirp order: chirps of a frame are consecutive in transmission order; frames
  are consecutive.

Adding a format: add a new value to the relevant ``SUPPORTED_*`` tuple and
implement it in :func:`_words_to_stream` / :func:`_stream_to_chirp` (and the
encoder), with a known-answer test vector in ``tests/test_adc_loader.py``.

Packet-level capture (UDP): :func:`reassemble_udp_packets` reorders DCA1000
raw-mode packets by sequence number and zero-fills gaps, reporting losses. Its
header layout (4-byte sequence number + 6-byte byte count, little-endian) is
the documented DCA1000 raw-mode header; verify against your firmware/tools.
"""

from __future__ import annotations

import struct
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional

import numpy as np

SUPPORTED_DATA_TYPES = ("complex", "real")
SUPPORTED_SIGN_HANDLING = ("int16_twos_complement", "sign_extend_adc_bits")
SUPPORTED_BYTE_ORDERS = ("little", "big")
SUPPORTED_IQ_ORDERS = ("I_first", "Q_first")
SUPPORTED_COMPLEX_PACKING = ("iq_interleaved", "two_sample_blocks")
SUPPORTED_CHANNEL_LAYOUTS = ("rx_planar_per_chirp", "rx_interleaved_per_sample")
SUPPORTED_ADC_BITS = (12, 14, 16)

WORD_BYTES = 2


class AdcFormatError(ValueError):
    """Incomplete, unsupported or inconsistent ADC format specification."""


class CaptureDecodeError(ValueError):
    """The capture data do not match the specified format."""


@dataclass
class AdcFormat:
    """Complete binary-format specification for a DCA1000 raw capture (no defaults on purpose)."""

    data_type: Optional[str] = None
    adc_bits: Optional[int] = None
    sign_handling: Optional[str] = None
    byte_order: Optional[str] = None
    iq_order: Optional[str] = None  # complex only
    complex_packing: Optional[str] = None  # complex only
    channel_layout: Optional[str] = None
    num_rx: Optional[int] = None
    num_adc_samples: Optional[int] = None
    chirps_per_frame: Optional[int] = None
    num_frames: Optional[int] = None  # optional: None -> derive from size (exact division required)

    def required_fields(self) -> list[str]:
        req = ["data_type", "adc_bits", "sign_handling", "byte_order", "channel_layout", "num_rx",
               "num_adc_samples", "chirps_per_frame"]
        if self.data_type == "complex":
            req += ["iq_order", "complex_packing"]
        return req

    def missing_fields(self) -> list[str]:
        return [f for f in self.required_fields() if getattr(self, f) is None]

    def validate(self) -> None:
        missing = self.missing_fields()
        if missing:
            raise AdcFormatError(
                "ADC format incomplete; missing: "
                + ", ".join(missing)
                + ". These must come from your mmWave Studio / DCA1000 CLI / SDK capture settings "
                "(see README 'Hardware validation checklist'); they are never guessed."
            )
        errs = []
        checks = [
            ("data_type", SUPPORTED_DATA_TYPES),
            ("sign_handling", SUPPORTED_SIGN_HANDLING),
            ("byte_order", SUPPORTED_BYTE_ORDERS),
            ("channel_layout", SUPPORTED_CHANNEL_LAYOUTS),
            ("adc_bits", SUPPORTED_ADC_BITS),
        ]
        if self.data_type == "complex":
            checks += [("iq_order", SUPPORTED_IQ_ORDERS), ("complex_packing", SUPPORTED_COMPLEX_PACKING)]
        for name, allowed in checks:
            if getattr(self, name) not in allowed:
                errs.append(f"{name}={getattr(self, name)!r} unsupported (supported: {allowed})")
        for name in ("num_rx", "num_adc_samples", "chirps_per_frame"):
            if getattr(self, name) <= 0:
                errs.append(f"{name} must be > 0")
        if self.num_frames is not None and self.num_frames <= 0:
            errs.append("num_frames must be > 0 when given")
        if self.data_type == "complex" and self.complex_packing == "two_sample_blocks":
            n = self.num_rx * self.num_adc_samples
            if n % 2:
                errs.append("two_sample_blocks packing needs an even number of complex samples per chirp")
        if self.sign_handling == "sign_extend_adc_bits" and self.adc_bits == 16:
            errs.append("sign_extend_adc_bits with adc_bits=16 is equivalent to int16_twos_complement; use that")
        if errs:
            raise AdcFormatError("unsupported/inconsistent ADC format: " + "; ".join(errs))

    @property
    def words_per_chirp(self) -> int:
        per = 2 if self.data_type == "complex" else 1
        return per * self.num_rx * self.num_adc_samples

    @property
    def bytes_per_frame(self) -> int:
        return self.words_per_chirp * self.chirps_per_frame * WORD_BYTES

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AdcFormat":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise AdcFormatError(f"unknown adc_format key(s): {sorted(unknown)}")
        return cls(**d)


@dataclass
class DecodeReport:
    num_frames: int
    bytes_total: int
    bytes_used: int
    bytes_dropped: int
    out_of_range_values: int
    full_scale_fraction: float
    channel_mean: list = field(default_factory=list)  # complex DC per RX (as [re, im])
    channel_rms: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Word conversion
# ---------------------------------------------------------------------------


def _read_words(buf: bytes | np.ndarray, byte_order: str) -> np.ndarray:
    dt = np.dtype("<u2" if byte_order == "little" else ">u2")
    return np.frombuffer(buf, dtype=dt).astype(np.uint16)


def words_to_values(words: np.ndarray, fmt: AdcFormat) -> tuple[np.ndarray, int]:
    """uint16 words -> signed integer values (int32), and count of values outside the adc_bits range."""
    w = np.asarray(words, np.uint16)
    bits = fmt.adc_bits
    if fmt.sign_handling == "int16_twos_complement":
        v = w.view(np.int16).astype(np.int32)
    elif fmt.sign_handling == "sign_extend_adc_bits":
        mask = (1 << bits) - 1
        v = (w.astype(np.int32) & mask)
        v = np.where(v >= (1 << (bits - 1)), v - (1 << bits), v)
        upper = (w.astype(np.int32) >> bits) != 0
        return v.astype(np.int32), int(np.count_nonzero(upper))  # non-zero upper bits are suspicious
    else:  # pragma: no cover - validated
        raise AdcFormatError(fmt.sign_handling)
    lim = 1 << (bits - 1)
    oor = int(np.count_nonzero((v < -lim) | (v > lim - 1)))
    return v, oor


def values_to_words(values: np.ndarray, fmt: AdcFormat) -> np.ndarray:
    """Signed integer values -> uint16 words (inverse of :func:`words_to_values`)."""
    v = np.asarray(values, np.int64)
    lim = 1 << (fmt.adc_bits - 1)
    if np.any(v < -lim) or np.any(v > lim - 1):
        raise ValueError(f"values exceed the {fmt.adc_bits}-bit range")
    if fmt.sign_handling == "int16_twos_complement":
        return v.astype(np.int16).view(np.uint16)
    return (v & ((1 << fmt.adc_bits) - 1)).astype(np.uint16)


def _words_to_stream(vals: np.ndarray, fmt: AdcFormat) -> np.ndarray:
    """(..., words_per_chirp) values -> (..., num_rx*num_adc_samples) complex or real stream."""
    if fmt.data_type == "real":
        return vals.astype(np.float32)
    if fmt.complex_packing == "iq_interleaved":
        a, b = vals[..., 0::2], vals[..., 1::2]
    else:  # two_sample_blocks
        blk = vals.reshape(*vals.shape[:-1], -1, 4)
        a = blk[..., 0:2].reshape(*vals.shape[:-1], -1)
        b = blk[..., 2:4].reshape(*vals.shape[:-1], -1)
    i, q = (a, b) if fmt.iq_order == "I_first" else (b, a)
    return (i.astype(np.float32) + 1j * q.astype(np.float32)).astype(np.complex64)


def _stream_to_chirp(stream: np.ndarray, fmt: AdcFormat) -> np.ndarray:
    """(..., M*K) -> (..., M, K)."""
    M, K = fmt.num_rx, fmt.num_adc_samples
    if fmt.channel_layout == "rx_planar_per_chirp":
        return stream.reshape(*stream.shape[:-1], M, K)
    return np.swapaxes(stream.reshape(*stream.shape[:-1], K, M), -1, -2)


# ---------------------------------------------------------------------------
# Decode / encode
# ---------------------------------------------------------------------------


def decode_bytes(buf: bytes, fmt: AdcFormat, allow_partial_final_frame: bool = False) -> tuple[np.ndarray, DecodeReport]:
    """Decode a raw byte stream -> cube (frames, chirps, rx, samples) + report."""
    fmt.validate()
    n = len(buf)
    fb = fmt.bytes_per_frame
    warnings = []
    if n % WORD_BYTES:
        raise CaptureDecodeError(f"capture size {n} bytes is not a whole number of 16-bit words")
    if fmt.num_frames is not None:
        need = fmt.num_frames * fb
        if n < need:
            raise CaptureDecodeError(
                f"capture has {n} bytes but {fmt.num_frames} frames x {fb} bytes = {need} are required "
                "(incomplete capture, packet loss, or wrong format)"
            )
        if n > need:
            warnings.append(f"{n - need} trailing bytes beyond {fmt.num_frames} frames ignored")
        frames = fmt.num_frames
    else:
        frames, rem = divmod(n, fb)
        if rem and not allow_partial_final_frame:
            raise CaptureDecodeError(
                f"capture size {n} bytes is not a multiple of the frame size {fb} bytes "
                f"({frames} full frames + {rem} bytes). The format settings may be wrong or the capture is "
                "truncated; set num_frames explicitly or allow_partial_final_frame after checking."
            )
        if rem:
            warnings.append(f"dropped partial final frame ({rem} bytes)")
        if frames == 0:
            raise CaptureDecodeError(f"capture ({n} bytes) is smaller than one frame ({fb} bytes)")
    used = frames * fb
    words = _read_words(buf[:used], fmt.byte_order)
    vals, oor = words_to_values(words, fmt)
    if oor:
        warnings.append(
            f"{oor} values outside the {fmt.adc_bits}-bit range / with unexpected upper bits: "
            "check sign_handling, adc_bits and packing"
        )
    vals = vals.reshape(frames, fmt.chirps_per_frame, fmt.words_per_chirp)
    cube = _stream_to_chirp(_words_to_stream(vals, fmt), fmt)  # (F, Nc, M, K)
    lim = (1 << (fmt.adc_bits - 1)) - 1
    fs_frac = float(np.mean(np.abs(vals) >= lim))
    if fs_frac > 1e-3:
        warnings.append(f"{fs_frac:.2%} of values at ADC full scale (clipping / saturation)")
    mean = cube.mean(axis=(0, 1, 3))
    rms = np.sqrt((np.abs(cube) ** 2).mean(axis=(0, 1, 3)))
    rep = DecodeReport(
        num_frames=int(frames),
        bytes_total=int(n),
        bytes_used=int(used),
        bytes_dropped=int(n - used),
        out_of_range_values=oor,
        full_scale_fraction=fs_frac,
        channel_mean=[[float(np.real(m)), float(np.imag(m))] for m in np.atleast_1d(mean)],
        channel_rms=[float(r) for r in np.atleast_1d(rms)],
        warnings=warnings,
    )
    return cube, rep


def encode_cube(cube: np.ndarray, fmt: AdcFormat) -> bytes:
    """Inverse of :func:`decode_bytes` for integer-valued cubes (frames, chirps, rx, samples)."""
    fmt.validate()
    cube = np.asarray(cube)
    F, Nc, M, K = cube.shape
    if (Nc, M, K) != (fmt.chirps_per_frame, fmt.num_rx, fmt.num_adc_samples):
        raise ValueError("cube shape does not match format")
    if fmt.channel_layout == "rx_planar_per_chirp":
        stream = cube.reshape(F, Nc, M * K)
    else:
        stream = np.swapaxes(cube, -1, -2).reshape(F, Nc, K * M)
    if fmt.data_type == "real":
        vals = np.rint(np.real(stream)).astype(np.int64)
    else:
        i = np.rint(stream.real).astype(np.int64)
        q = np.rint(stream.imag).astype(np.int64)
        a, b = (i, q) if fmt.iq_order == "I_first" else (q, i)
        if fmt.complex_packing == "iq_interleaved":
            vals = np.stack([a, b], axis=-1).reshape(F, Nc, -1)
        else:
            a2 = a.reshape(F, Nc, -1, 2)
            b2 = b.reshape(F, Nc, -1, 2)
            vals = np.concatenate([a2, b2], axis=-1).reshape(F, Nc, -1)
    words = values_to_words(vals.ravel(), fmt)
    dt = np.dtype("<u2" if fmt.byte_order == "little" else ">u2")
    return words.astype(dt).tobytes()


def load_capture_file(path: str | Path, fmt: AdcFormat, allow_partial_final_frame: bool = False,
                      max_frames: int | None = None) -> tuple[np.ndarray, DecodeReport]:
    """Decode a raw (header-free) DCA1000 ``.bin`` capture."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    fmt.validate()
    size = p.stat().st_size
    nbytes = size if max_frames is None else min(size, max_frames * fmt.bytes_per_frame)
    with open(p, "rb") as fh:
        buf = fh.read(nbytes)
    if max_frames is not None and nbytes < size:
        sub = AdcFormat(**{**fmt.as_dict(), "num_frames": nbytes // fmt.bytes_per_frame})
        return decode_bytes(buf, sub, allow_partial_final_frame)
    return decode_bytes(buf, fmt, allow_partial_final_frame)


def cube_to_cpis(cube: np.ndarray) -> np.ndarray:
    """(frames, chirps, rx, samples) -> (frames, rx, chirps, samples), i.e. X[rx, chirp, sample] per frame."""
    return np.ascontiguousarray(np.transpose(cube, (0, 2, 1, 3)))


# ---------------------------------------------------------------------------
# UDP packet reassembly (DCA1000 raw mode)
# ---------------------------------------------------------------------------

UDP_HEADER = struct.Struct("<I6s")  # seq (uint32 LE), byte count (uint48 LE)


def make_udp_packet(seq: int, byte_offset: int, payload: bytes) -> bytes:
    """Build a raw-mode packet (used for synthetic tests): header + payload."""
    return UDP_HEADER.pack(seq, byte_offset.to_bytes(6, "little")) + payload


def reassemble_udp_packets(packets: list[bytes]) -> tuple[bytes, dict]:
    """Reorder packets by sequence number; zero-fill missing byte ranges using the byte-count field.

    The byte-count field is interpreted as the number of payload bytes sent
    *before* this packet. Returns (stream, report) with lost/duplicate counts.
    """
    parsed = {}
    dup = 0
    for pkt in packets:
        if len(pkt) < UDP_HEADER.size:
            raise CaptureDecodeError("packet shorter than header")
        seq, cnt = UDP_HEADER.unpack_from(pkt)
        off = int.from_bytes(cnt, "little")
        if seq in parsed:
            dup += 1
            continue
        parsed[seq] = (off, pkt[UDP_HEADER.size:])
    if not parsed:
        return b"", {"packets": 0, "lost_packets": 0, "lost_bytes": 0, "duplicates": dup}
    seqs = sorted(parsed)
    out = bytearray()
    lost_bytes = 0
    for s in seqs:
        off, payload = parsed[s]
        if off < len(out):
            raise CaptureDecodeError(f"packet {s}: byte offset {off} overlaps already assembled data")
        if off > len(out):
            lost_bytes += off - len(out)
            out.extend(b"\x00" * (off - len(out)))
        out.extend(payload)
    lost_packets = (seqs[-1] - seqs[0] + 1) - len(seqs)
    return bytes(out), {
        "packets": len(seqs),
        "first_seq": seqs[0],
        "last_seq": seqs[-1],
        "lost_packets": lost_packets,
        "lost_bytes": lost_bytes,
        "duplicates": dup,
        "out_of_order": [p for p in packets if p] != sorted(packets, key=lambda p: UDP_HEADER.unpack_from(p)[0]),
    }

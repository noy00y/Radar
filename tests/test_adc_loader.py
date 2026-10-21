"""DCA1000 decoder tests with hand-made known-answer vectors (no hardware required)."""

import itertools

import numpy as np
import pytest

from radar_stap.adc_loader import (
    AdcFormat,
    AdcFormatError,
    CaptureDecodeError,
    decode_bytes,
    encode_cube,
    load_capture_file,
    make_udp_packet,
    reassemble_udp_packets,
    words_to_values,
)
from radar_stap.mmwave_cfg import MmwaveCfgError, capture_settings_from_cli, parse_mmwave_cfg


def fmt(**kw):
    base = dict(
        data_type="complex", adc_bits=16, sign_handling="int16_twos_complement", byte_order="little",
        iq_order="I_first", complex_packing="iq_interleaved", channel_layout="rx_planar_per_chirp",
        num_rx=2, num_adc_samples=2, chirps_per_frame=1, num_frames=None,
    )
    base.update(kw)
    return AdcFormat(**base)


def words(*w, order="<"):
    return np.array(w, dtype=np.int64).astype(np.int16).astype(order + "i2").tobytes()


def test_known_vector_iq_interleaved_planar():
    cube, rep = decode_bytes(words(1, 2, 3, 4, 5, 6, 7, 8), fmt())
    assert cube.shape == (1, 1, 2, 2)  # frames, chirps, rx, samples
    assert np.array_equal(cube[0, 0, 0], [1 + 2j, 3 + 4j])
    assert np.array_equal(cube[0, 0, 1], [5 + 6j, 7 + 8j])
    assert rep.num_frames == 1 and rep.bytes_dropped == 0


def test_known_vector_two_sample_blocks_and_q_first():
    w = words(1, 2, 10, 20, 3, 4, 30, 40)
    c, _ = decode_bytes(w, fmt(num_rx=1, num_adc_samples=4, complex_packing="two_sample_blocks"))
    assert np.array_equal(c[0, 0, 0], [1 + 10j, 2 + 20j, 3 + 30j, 4 + 40j])
    c, _ = decode_bytes(w, fmt(num_rx=1, num_adc_samples=4, complex_packing="two_sample_blocks", iq_order="Q_first"))
    assert np.array_equal(c[0, 0, 0], [10 + 1j, 20 + 2j, 30 + 3j, 40 + 4j])


def test_channel_interleaving_order():
    # per sample: RX0 then RX1  ->  s0: (1+2j)(3+4j), s1: (5+6j)(7+8j)
    c, _ = decode_bytes(words(1, 2, 3, 4, 5, 6, 7, 8), fmt(channel_layout="rx_interleaved_per_sample"))
    assert np.array_equal(c[0, 0, 0], [1 + 2j, 5 + 6j])
    assert np.array_equal(c[0, 0, 1], [3 + 4j, 7 + 8j])


def test_signed_conversion_and_sign_extension():
    f16 = fmt(data_type="real", num_rx=1, num_adc_samples=4, iq_order=None, complex_packing=None)
    c, _ = decode_bytes(words(-1, -32768, 32767, 0), f16)
    assert np.array_equal(c[0, 0, 0], [-1, -32768, 32767, 0])
    f12 = fmt(data_type="real", adc_bits=12, sign_handling="sign_extend_adc_bits", num_rx=1, num_adc_samples=4,
              iq_order=None, complex_packing=None)
    raw = np.array([0x0FFF, 0x0800, 0x07FF, 0x0001], dtype="<u2").tobytes()
    c, rep = decode_bytes(raw, f12)
    assert np.array_equal(c[0, 0, 0], [-1, -2048, 2047, 1])
    assert rep.out_of_range_values == 0
    raw_bad = np.array([0xF001, 0, 0, 0], dtype="<u2").tobytes()  # upper bits set: suspicious
    _, rep = decode_bytes(raw_bad, f12)
    assert rep.out_of_range_values == 1 and rep.warnings


def test_out_of_range_values_flag_wrong_bit_width():
    f = fmt(data_type="real", adc_bits=12, num_rx=1, num_adc_samples=2, iq_order=None, complex_packing=None)
    _, rep = decode_bytes(words(5000, -3), f)
    assert rep.out_of_range_values == 1
    assert any("outside" in w for w in rep.warnings)


def test_big_endian():
    b = np.array([1, -2, 3, -4], dtype=">i2").tobytes()
    c, _ = decode_bytes(b, fmt(byte_order="big", num_rx=1))
    assert np.array_equal(c[0, 0, 0], [1 - 2j, 3 - 4j])


def test_incomplete_and_invalid_lengths():
    f = fmt(chirps_per_frame=2)  # 2 chirps x 2 rx x 2 samples x (I,Q) x 2 bytes = 32 bytes per frame
    with pytest.raises(CaptureDecodeError, match="not a multiple"):
        decode_bytes(words(*range(12)), f)
    with pytest.raises(CaptureDecodeError, match="whole number of 16-bit words"):
        decode_bytes(b"\x00" * 17, f)
    with pytest.raises(CaptureDecodeError, match="required"):
        decode_bytes(words(*range(8)), fmt(chirps_per_frame=2, num_frames=2))
    with pytest.raises(CaptureDecodeError, match="smaller than one frame"):
        decode_bytes(words(*range(4)), f, allow_partial_final_frame=True)
    c, rep = decode_bytes(words(*range(24)), f, allow_partial_final_frame=True)
    assert c.shape[0] == 1 and rep.bytes_dropped == 16 and rep.warnings


def test_missing_and_unsupported_settings_rejected():
    with pytest.raises(AdcFormatError, match="iq_order"):
        decode_bytes(words(1, 2), fmt(iq_order=None))
    with pytest.raises(AdcFormatError, match="missing"):
        AdcFormat(data_type="complex").validate()
    with pytest.raises(AdcFormatError, match="unsupported"):
        fmt(complex_packing="lvds4").validate()
    with pytest.raises(AdcFormatError, match="unsupported"):
        fmt(adc_bits=10).validate()
    with pytest.raises(AdcFormatError, match="even number"):
        fmt(num_rx=1, num_adc_samples=3, complex_packing="two_sample_blocks").validate()
    with pytest.raises(AdcFormatError, match="unknown"):
        AdcFormat.from_dict({"data_typ": "complex"})


@pytest.mark.parametrize(
    "packing,iq,layout,order,sign",
    list(itertools.product(("iq_interleaved", "two_sample_blocks"), ("I_first", "Q_first"),
                           ("rx_planar_per_chirp", "rx_interleaved_per_sample"), ("little", "big"),
                           ("int16_twos_complement", "sign_extend_adc_bits"))),
)
def test_encode_decode_roundtrip(packing, iq, layout, order, sign):
    f = fmt(complex_packing=packing, iq_order=iq, channel_layout=layout, byte_order=order, sign_handling=sign,
            adc_bits=14, num_rx=3, num_adc_samples=8, chirps_per_frame=4, num_frames=2)
    rng = np.random.default_rng(0)
    cube = rng.integers(-8192, 8192, (2, 4, 3, 8)) + 1j * rng.integers(-8192, 8192, (2, 4, 3, 8))
    out, rep = decode_bytes(encode_cube(cube, f), f)
    assert np.array_equal(out, cube)
    assert rep.out_of_range_values == 0


def test_real_roundtrip_and_file_loading(tmp_path):
    f = fmt(data_type="real", iq_order=None, complex_packing=None, num_rx=4, num_adc_samples=16, chirps_per_frame=3)
    cube = np.random.default_rng(1).integers(-100, 100, (5, 3, 4, 16))
    p = tmp_path / "adc.bin"
    p.write_bytes(encode_cube(cube, f))
    out, rep = load_capture_file(p, f)
    assert np.array_equal(out.real, cube) and rep.num_frames == 5
    out, rep = load_capture_file(p, f, max_frames=2)
    assert out.shape[0] == 2


def test_udp_reassembly_reorders_and_zero_fills_losses():
    stream = bytes(range(200)) * 3  # 600 bytes
    chunk = 100
    pkts = [make_udp_packet(i + 1, i * chunk, stream[i * chunk:(i + 1) * chunk]) for i in range(6)]
    shuffled = [pkts[3], pkts[0], pkts[5], pkts[1], pkts[4]]  # packet seq 3 (index 2) lost
    out, rep = reassemble_udp_packets(shuffled)
    assert rep["lost_packets"] == 1 and rep["lost_bytes"] == chunk
    assert out[:200] == stream[:200] and out[300:] == stream[300:]
    assert out[200:300] == b"\x00" * chunk
    out2, rep2 = reassemble_udp_packets(pkts + [pkts[2]])
    assert out2 == stream and rep2["duplicates"] == 1 and rep2["lost_packets"] == 0


def test_words_to_values_int16():
    v, oor = words_to_values(np.array([0xFFFF, 0x8000], np.uint16), fmt())
    assert list(v) == [-1, -32768] and oor == 0


CFG_TEXT = """
% example SDK-style configuration
sensorStop
channelCfg 15 5 0
adcCfg 2 1
adcbufCfg -1 0 1 1 1
profileCfg 0 77 7 6 57.14 0 0 70 1 256 5209 0 0 30
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 1 1 0 0 0 0 0 4
frameCfg 0 1 32 0 100 1 0
"""


def test_mmwave_cfg_parsing():
    s = capture_settings_from_cli(parse_mmwave_cfg(CFG_TEXT))
    w = s["waveform"]
    assert w["carrier_frequency_hz"] == pytest.approx(77e9)
    assert w["chirp_slope_hz_per_s"] == pytest.approx(70e12)
    assert w["idle_time_s"] == pytest.approx(7e-6)
    assert w["ramp_end_time_s"] == pytest.approx(57.14e-6)
    assert w["adc_sample_rate_hz"] == pytest.approx(5.209e6)
    assert s["num_adc_samples"] == 256 and s["num_rx"] == 4
    assert s["tx_schedule"] == [0, 2] and s["chirps_per_frame"] == 64
    assert s["adc_bits"] == 16 and s["data_type"] == "complex"
    assert "NOT necessarily" in s["hints"]["adcbuf"]["note"]


def test_mmwave_cfg_rejects_simultaneous_tx_and_missing_chirps():
    with pytest.raises(MmwaveCfgError, match="simultaneous"):
        capture_settings_from_cli(parse_mmwave_cfg(CFG_TEXT.replace("chirpCfg 1 1 0 0 0 0 0 4", "chirpCfg 1 1 0 0 0 0 0 5")))
    with pytest.raises(MmwaveCfgError, match="no chirpCfg"):
        capture_settings_from_cli(parse_mmwave_cfg(CFG_TEXT.replace("chirpCfg 1 1 0 0 0 0 0 4\n", "")))

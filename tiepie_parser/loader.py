import configparser
import warnings
from os import PathLike
from pathlib import Path

import numpy as np

from .tpo import parse_tpo

__all__ = [
    "load_tpidx",
    "load_tpo",
]


def load_tpo(
    tpo_path: str | PathLike,
) -> tuple[None, None, None, None, dict] | tuple[np.ndarray, np.datetime64, float, float, dict]:
    tpo_path = Path(tpo_path)
    with tpo_path.open("rb") as fp:
        raw_tpo = parse_tpo(fp)
    src = raw_tpo["RIFF", "SRC "]
    if "DATA" not in src:
        return None, None, None, None, raw_tpo
    untyped_data = src.pop("DATA")
    raw_dty = src.pop("DATY")
    if raw_dty != b"\x05 ":
        raise NotImplementedError("Only FLOAT32 data is supported for now")
    dty = {
        b"\x05 ": np.float32,
    }[raw_dty]
    data = untyped_data.view(dty)
    count = src.pop("DASI")
    if count != len(data):
        wmsg = f"Expected {count} data points, but found {len(data)} in {tpo_path}"
        warnings.warn(wmsg, stacklevel=2)
    start_time = src.pop("TIME")
    sample_rate = src.pop("SAFR")
    sample_offset = src.pop("SVAL", 0.0)
    return data, start_time, sample_rate, sample_offset, raw_tpo


def load_tpidx(
    tpidx_path: str | PathLike,
) -> dict[str, tuple[np.ndarray, np.datetime64, float, tuple[float, ...], dict]]:
    tpidx_path = Path(tpidx_path)
    parser = configparser.ConfigParser()
    parser.read(tpidx_path)
    tpos = {}
    for k, d in parser.items():
        if k == "DEFAULT":
            continue
        dd = dict(d.items())
        fname = dd.pop("filename")
        fcount = int(dd.pop("filecount"))
        # dsize = int(dd.pop("datasize"))
        # samplefrequency = float(dd.pop("samplefrequency"))
        # print(k, fname, fcount, dd)
        children = {child: int(child.stem.removeprefix(fname)) for child in tpidx_path.parent.glob(f"{fname}*.tpo")}
        assert len(children) == fcount, f"{len(children)} == {fcount}"
        assert set(range(fcount)) == set(children.values())
        datas, start_times, sample_rates, sample_offsets, raw_tpos = zip(
            *[load_tpo(c) for c, _ in sorted(children.items(), key=lambda t: t[1])],
            strict=True,
        )
        assert len(set(start_times)) == 1
        assert len(set(sample_rates)) == 1
        (start_time,) = set(start_times)
        (sample_rate,) = set(sample_rates)
        # FIXME: how to use `sample_offsets` during concatenation?
        if len(datas) == 1:
            (data,) = datas
        else:
            data = np.concatenate(datas, axis=0)
        # TODO: should it raise an Exception or skip it?
        if data is None:
            continue
        tpos[k] = data, start_time, sample_rate, sample_offsets, raw_tpos
    return tpos

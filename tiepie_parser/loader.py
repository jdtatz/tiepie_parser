import configparser
import warnings
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path

import numpy as np

from .tpo import parse_tpo

__all__ = [
    "DataCollection",
    "load_tpidx",
    "load_tpo",
]


@dataclass
class DataCollection:
    values: np.ndarray
    start: np.datetime64
    sample_rate: float
    _sample_offsets: tuple[float, ...] = field(repr=False)
    raw_tpos: tuple[dict, ...] | None = None

    @property
    @warnings.deprecated("sample offsets are now verified and used internally")
    def sample_offsets(self):
        return self._sample_offsets

    @warnings.deprecated("load_tpidx now returns a DataCollection, not a tuple")
    def __iter__(self):
        return iter((self.values, self.start, self.sample_rate, self._sample_offsets, self.raw_tpos))

    @warnings.deprecated("load_tpidx now returns a DataCollection, not a tuple")
    def __getitem__(self, index):
        return (self.values, self.start, self.sample_rate, self._sample_offsets, self.raw_tpos)[index]


def _load_tpo(
    tpo_path: str | PathLike, *, pad_missing: bool = True
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
    if count > len(data):
        wmsg = f"Missing data, expected {count} data points, but found {len(data)} in {tpo_path}"
        warnings.warn(wmsg, stacklevel=3)
        if pad_missing:
            data = np.pad(data, (0, count - len(data)), constant_values=np.nan)
    elif count < len(data):
        wmsg = f"Excess data, expected {count} data points, but found {len(data)} in {tpo_path}"
        warnings.warn(wmsg, stacklevel=3)
    start_time = src.pop("TIME")
    sample_rate = src.pop("SAFR")
    sample_offset = src.pop("SVAL", 0.0)
    return data, start_time, sample_rate, sample_offset, raw_tpo


def load_tpo(
    tpo_path: str | PathLike, *, pad_missing: bool = True
) -> tuple[None, None, None, None, dict] | tuple[np.ndarray, np.datetime64, float, float, dict]:
    return _load_tpo(tpo_path, pad_missing=pad_missing)


def load_tpidx(tpidx_path: str | PathLike, *, pad_missing: bool = True, full: bool = True) -> dict[str, DataCollection]:
    tpidx_path = Path(tpidx_path)
    parser = configparser.ConfigParser()
    with tpidx_path.open() as fp:
        parser.read_file(fp, str(tpidx_path))
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
            *[_load_tpo(c, pad_missing=pad_missing) for c, _ in sorted(children.items(), key=lambda t: t[1])],
            strict=True,
        )
        assert len(set(start_times)) == 1
        assert len(set(sample_rates)) == 1
        (start_time,) = set(start_times)
        (sample_rate,) = set(sample_rates)
        if sample_offsets[0] is not None and sample_offsets[0] != 0:
            msg = f"The first sample offset {sample_offsets[0]} is not 0 for {k!r}"
            warnings.warn(msg, stacklevel=2)
        if sample_rate is not None and np.any(
            np.diff([sample_rate * o for o in sample_offsets]) != [len(d) for d in datas[:-1]]
        ):
            # TODO: improve the message
            msg = f"data doesn't match sample offsets, concatenation will be incorrect, for {k!r}"
            warnings.warn(msg, stacklevel=2)
        if len(datas) == 1:
            (data,) = datas
        else:
            data = np.concatenate(datas, axis=0)
        # TODO: should it raise an Exception or skip it?
        if data is None:
            continue
        tpos[k] = DataCollection(data, start_time, sample_rate, sample_offsets, raw_tpos if full else None)
    return tpos

import struct
import uuid
import warnings
from functools import singledispatch
from typing import Literal

import numpy as np
from typing_extensions import Reader

from .riff import DataChunk, FormChunk, ListChunk, RawChunk, parse_riff

__all__ = [
    "parse_tpo",
]

type _Scalar = int | float | np.datetime64 | bytes | str


@singledispatch
def _parse_tpo_subchunk(
    chunk,
    *,
    warn=False,
) -> tuple[str | tuple[Literal["RIFF"], str], dict | np.ndarray[tuple[int], np.uint8] | _Scalar]:
    raise TypeError


@_parse_tpo_subchunk.register
def _(chunk: FormChunk, *, warn: bool = False) -> tuple[tuple[Literal["RIFF"], str], dict]:
    form = {}
    for subchunk in chunk.value:
        k, v = _parse_tpo_subchunk(subchunk, warn=warn)
        if k in form:
            raise ValueError(f"Duplicate subchunk id {k} in form {FormChunk}")
        form[k] = v
    return ("RIFF", chunk.form_type), form


@_parse_tpo_subchunk.register
def _(chunk: ListChunk, *, warn: bool = False):  # -> tuple[tuple[Literal["LIST"], str], list]:
    raise NotImplementedError(f"'LIST' chunks are not yet implemented {chunk}")


@_parse_tpo_subchunk.register
def _(chunk: DataChunk, *, warn: bool = False) -> tuple[str, np.ndarray[tuple[int], np.uint8]]:
    return chunk.chunk_id, chunk.value


@_parse_tpo_subchunk.register
def _(chunk: RawChunk, *, warn: bool = False) -> tuple[str, _Scalar]:
    match chunk.chunk_id:
        case "NUM " | "OUT#" | "DASI" | "DOMN" | "COLM" | "RESO":
            data = int.from_bytes(chunk.value, byteorder="little", signed=False)
        case "TIME":
            assert len(chunk.value) == 8
            # https://www.tiepie.com/en/multi-channel/exporting-data#foot_1
            (ts,) = struct.unpack("<d", chunk.value)
            if ts <= 50_000:
                days_to_ns = 24 * 60 * 60 * 1_000_000_000
                data = np.timedelta64(int(ts * days_to_ns), "ns") + np.datetime64("1899-12-30")
            else:
                data = ts
        case "SAFR" | "SVAL":
            assert len(chunk.value) == 8
            (data,) = struct.unpack("<d", chunk.value)
        case "NAME":
            _data = chunk.value
            try:
                data = _data.decode("utf-16-le")
            except UnicodeDecodeError:
                data = _data
        case "CLID":
            assert len(chunk.value) == 16
            # FIXME: is this actually a UUID?
            data = uuid.UUID(int=int.from_bytes(chunk.value, "little"))
        case "BNDS":
            assert len(chunk.value) == 16
            data = struct.unpack("<dd", chunk.value)
        case _:
            data = chunk.value
            if warn:
                msg = f"unimplemented chunk id: {chunk.chunk_id}, size: {len(chunk.value)}, data: {data[:128]}"
                warnings.warn(msg, stacklevel=2)
    return chunk.chunk_id, data


def parse_tpo(fp: Reader[bytes], *, warn_unimplemented: bool = False) -> dict:
    riff, _size = parse_riff(fp)
    if riff.form_type != "MIMO":
        raise ValueError(f"Invalid RIFF type, expected 'MIMO', but found {riff.form_type!r}")

    _k, chunks = _parse_tpo_subchunk(riff, warn=warn_unimplemented)
    assert _k == ("RIFF", "MIMO")
    assert isinstance(chunks, dict)
    return chunks

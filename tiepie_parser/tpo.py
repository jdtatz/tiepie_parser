import struct
import uuid
import warnings

import numpy as np
from typing_extensions import Reader

__all__ = [
    "parse_tpo",
]


def parse_tpo_subchunk(fp: Reader[bytes]):
    subchunk_id, subchunk_size = struct.unpack("<4sI", fp.read(8))
    if subchunk_id.isascii():
        subchunk_id = subchunk_id.decode("ascii")
    match subchunk_id:
        case "RIFF":
            (form_type,) = struct.unpack("<4s", fp.read(4))
            if form_type.isascii():
                form_type = form_type.decode("ascii")

            form_chunks = {}
            offset = 12
            while offset < subchunk_size:
                key, size, data = parse_tpo_subchunk(fp)
                offset += size + 8
                assert key not in form_chunks
                form_chunks[key] = data
            return (subchunk_id, form_type), subchunk_size, form_chunks
        case "LIST":
            (list_type,) = struct.unpack("<4s", fp.read(4))
            if list_type.isascii():
                list_type = list_type.decode("ascii")
            raise NotImplementedError((subchunk_id, subchunk_size, list_type))
            # FIXME: untested, but should be correct
            list_chunks = []
            offset = 8
            while offset < subchunk_size:
                key, size, data = parse_tpo_subchunk(fp)
                offset += size + 8
                list_chunks.append((key, data))
            return (subchunk_id, list_type), subchunk_size, list_chunks
        case "NUM " | "OUT#" | "DASI" | "DOMN" | "COLM" | "RESO":
            unpack_fmt = {
                1: "<B",
                2: "<H",
                4: "<I",
                8: "<Q",
            }[subchunk_size]
            (data,) = struct.unpack(unpack_fmt, fp.read(subchunk_size))
        case "TIME":
            assert subchunk_size == 8
            # https://www.tiepie.com/en/multi-channel/exporting-data#foot_1
            (ts,) = struct.unpack("<d", fp.read(8))
            if ts <= 50_000:
                days_to_ns = 24 * 60 * 60 * 1_000_000_000
                data = np.timedelta64(int(ts * days_to_ns), "ns") + np.datetime64("1899-12-30")
            else:
                data = ts
        case "SAFR" | "SVAL":
            assert subchunk_size == 8
            (data,) = struct.unpack("<d", fp.read(8))
        case "NAME":
            _data = fp.read(subchunk_size)
            try:
                data = _data.decode("utf-16-le")
            except UnicodeDecodeError:
                data = _data
        case "CLID":
            assert subchunk_size == 16
            # FIXME: is this actually a UUID?
            data = uuid.UUID(int=int.from_bytes(fp.read(16), "little"))
        case "BNDS":
            assert subchunk_size == 16
            data = struct.unpack("<dd", fp.read(subchunk_size))
        case "DATA":
            # data = np.frombuffer(fp, dtype=np.uint8, count=subchunk_size)
            # data = np.frombuffer(fp.read(subchunk_size), dtype=np.uint8)
            data = np.fromfile(fp, dtype=np.uint8, count=subchunk_size)
        case _:
            data = fp.read(subchunk_size)
            msg = f"unimplemented chunk id: {subchunk_id}, size: {subchunk_size}, data: {data[:128]}"
            warnings.warn(msg, stacklevel=2)
    return subchunk_id, subchunk_size, data


def parse_tpo(fp: Reader[bytes]) -> dict:
    riff, fsize, fformat = struct.unpack("<4sI4s", fp.read(12))
    if riff != b"RIFF":
        raise ValueError(f"Invalid file signature, expected b'RIFF', but found {riff!r}")
    elif fformat != b"MIMO":
        raise ValueError(f"Invalid RIFF type, expected b'MIMO', but found {fformat!r}")

    chunks = {}
    offset = 12
    while offset < fsize:
        key, size, data = parse_tpo_subchunk(fp)
        offset += size + 8
        assert key not in chunks
        chunks[key] = data
    return chunks

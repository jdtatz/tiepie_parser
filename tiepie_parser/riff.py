import io
import struct
import warnings
from dataclasses import dataclass
from typing import TypeGuard

import numpy as np
from typing_extensions import Reader

__all__ = [
    "Chunk",
    "DataChunk",
    "FormChunk",
    "ListChunk",
    "RawChunk",
    "parse_riff",
]


@dataclass
class RawChunk:
    chunk_id: str
    value: bytes


@dataclass
class DataChunk:
    chunk_id: str
    value: np.ndarray[tuple[int], np.uint8]


type Chunk = RawChunk | DataChunk | FormChunk | ListChunk


@dataclass
class FormChunk:
    form_type: str
    value: list[Chunk]

    def __iter__(self):
        return iter(self.value)


@dataclass
class ListChunk:
    list_type: str
    value: list[Chunk]

    def __iter__(self):
        return iter(self.value)


def _is_real_file(fp: Reader[bytes]) -> TypeGuard[io.IOBase]:
    if not isinstance(fp, io.IOBase):  #  not hasattr(fp, "fileno"):
        return False
    try:
        _ = fp.fileno()
        return True
    except OSError:
        return False


def _parse_form_list_chunks(fp: Reader[bytes], subchunk_size: int) -> tuple[str, list[Chunk]]:
    (kind,) = struct.unpack("<4s", fp.read(4))
    if kind.isascii():
        kind = kind.decode("ascii")

    subchunks = []
    offset = 4
    while offset < subchunk_size:
        subchunk, size = parse_riff_subchunk(fp)
        offset += size
        subchunks.append(subchunk)
    assert offset == subchunk_size, f"{offset} != {subchunk_size}"
    return kind, subchunks


def parse_riff_subchunk(fp: Reader[bytes], *, _check_signature: bool = False) -> tuple[Chunk, int]:
    header_size = 8
    subchunk_id, subchunk_size = struct.unpack("<4sI", fp.read(8))
    if subchunk_id.isascii():
        subchunk_id = subchunk_id.decode("ascii")
    if _check_signature and subchunk_id != "RIFF":
        msg = f"Invalid file signature, expected 'RIFF', but found {subchunk_id!r}"
        raise ValueError(msg)

    match subchunk_id:
        case "RIFF":
            form_type, form_chunks = _parse_form_list_chunks(fp, subchunk_size)
            chunk = FormChunk(form_type, form_chunks)
        case "LIST":
            list_type, list_chunks = _parse_form_list_chunks(fp, subchunk_size)
            chunk = ListChunk(list_type, list_chunks)
        case "data" | "DATA":
            if _is_real_file(fp):
                data = np.fromfile(fp, dtype=np.uint8, count=subchunk_size)
            else:
                # data = np.frombuffer(fp, dtype=np.uint8, count=subchunk_size)
                data = np.frombuffer(fp.read(subchunk_size), dtype=np.uint8)
            chunk = DataChunk(subchunk_id, data)
        case _:
            data = fp.read(subchunk_size)
            chunk = RawChunk(subchunk_id, data)
    if subchunk_size % 2 == 1:
        pad = fp.read(1)
        if pad != b"\x00":
            # NOTE: not inlined into the string to ensure the repr is consistent
            zero = b"\x00"
            msg = f"padding byte, {pad}, is not {zero}, for subchunk {subchunk_id!r}"
            warnings.warn(msg, stacklevel=2)
        subchunk_size += 1
    return chunk, header_size + subchunk_size


def parse_riff(fp: Reader[bytes]) -> tuple[FormChunk, int]:
    return parse_riff_subchunk(fp, _check_signature=True)

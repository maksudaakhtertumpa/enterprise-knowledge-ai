"""Original-file storage on local disk.

Paths are built only from server-generated IDs — never from the client's
filename — so path traversal is impossible by construction. The interface
(save/open/delete by key) is what an S3 implementation will provide in Phase 8.
"""

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

_KEY = re.compile(r"^[0-9a-f-]{36}/[0-9a-f-]{36}\.(pdf|docx|txt)$")
_COPY_BLOCK = 1024 * 1024


class FileTooLarge(Exception):
    pass


@dataclass(frozen=True, slots=True)
class StoredFile:
    key: str
    size_bytes: int
    sha256: str


class LocalFileStorage:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def path_for(self, key: str) -> Path:
        if not _KEY.match(key):
            raise ValueError("invalid storage key")
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root):  # defence in depth
            raise ValueError("invalid storage key")
        return path

    def save(
        self, source: BinaryIO, *, workspace_id: str, document_id: str, ext: str, max_bytes: int
    ) -> StoredFile:
        """Stream to disk while hashing, aborting as soon as max_bytes is exceeded."""
        key = f"{workspace_id}/{document_id}.{ext}"
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")

        digest = hashlib.sha256()
        size = 0
        try:
            with tmp.open("wb") as out:
                while block := source.read(_COPY_BLOCK):
                    size += len(block)
                    if size > max_bytes:
                        raise FileTooLarge
                    digest.update(block)
                    out.write(block)
            os.replace(tmp, path)  # atomic: readers never see a half-written file
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return StoredFile(key=key, size_bytes=size, sha256=digest.hexdigest())

    def delete(self, key: str) -> None:
        self.path_for(key).unlink(missing_ok=True)

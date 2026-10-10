"""Upload validation: filename hygiene, type allow-list and content sniffing.

Nothing the client sends is trusted: the Content-Type header is ignored for
decisions, the extension only selects which content check to run, and the
bytes themselves must match the claimed type.
"""

import re
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

from app.db.models import FileType
from app.services.ingestion.types import ExtractionError

ALLOWED_EXTENSIONS: dict[str, FileType] = {
    ".pdf": FileType.PDF,
    ".docx": FileType.DOCX,
    ".txt": FileType.TXT,
}

_UNSAFE_CHARS = re.compile(r'[\x00-\x1f\x7f<>:"/\\|?*]')
_MAX_ZIP_ENTRIES = 10_000


class UploadRejected(Exception):
    def __init__(self, code: str, message: str, status_code: int = 415) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def sanitize_filename(raw: str | None) -> str:
    """Display-safe filename. Never used to build a filesystem path."""
    if not raw:
        raise UploadRejected("missing_filename", "The upload has no filename.", 400)
    # Drop any directory part, whichever OS style the client used.
    name = PureWindowsPath(PurePosixPath(raw).name).name
    name = unicodedata.normalize("NFC", name)
    name = _UNSAFE_CHARS.sub("_", name).strip(" .")
    if not name:
        raise UploadRejected("invalid_filename", "The filename is not valid.", 400)
    if len(name) > 255:
        stem, dot, ext = name.rpartition(".")
        name = (stem[: 250 - len(ext)] + dot + ext) if dot else name[:255]
    return name


def file_type_for(filename: str) -> FileType:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise UploadRejected("unsupported_type", f"Unsupported file type. Allowed: {allowed}.")
    return ALLOWED_EXTENSIONS[ext]


def verify_content(path: Path, file_type: FileType, *, max_docx_uncompressed: int) -> None:
    """Check the stored bytes really are the claimed type."""
    with path.open("rb") as fh:
        head = fh.read(8192)

    if file_type is FileType.PDF:
        # The spec allows junk before the header; real readers scan the first 1 KB.
        if b"%PDF-" not in head[:1024]:
            raise UploadRejected("content_mismatch", "The file is not a valid PDF.")
    elif file_type is FileType.DOCX:
        if not head.startswith(b"PK"):
            raise UploadRejected("content_mismatch", "The file is not a valid DOCX.")
        try:
            check_zip_safety(path, max_uncompressed_bytes=max_docx_uncompressed)
        except ExtractionError as exc:
            raise UploadRejected(exc.code, exc.message) from exc
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
        if "word/document.xml" not in names or "[Content_Types].xml" not in names:
            raise UploadRejected("content_mismatch", "The file is not a valid DOCX.")
    elif file_type is FileType.TXT and b"\x00" in head:
        # NUL bytes don't occur in text (UTF-16 files also land here; ask for UTF-8).
        raise UploadRejected(
            "content_mismatch", "The file looks binary. Text files must be UTF-8 or similar."
        )


def check_zip_safety(path: Path, *, max_uncompressed_bytes: int) -> None:
    """Reject zip bombs before any XML parser touches the archive."""
    try:
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
    except (zipfile.BadZipFile, OSError) as exc:
        raise ExtractionError("corrupt_file", "The archive is corrupt.") from exc
    if len(infos) > _MAX_ZIP_ENTRIES:
        raise ExtractionError("archive_too_complex", "The document has too many parts.")
    total = sum(info.file_size for info in infos)
    if total > max_uncompressed_bytes:
        raise ExtractionError("archive_too_large", "The document expands to an unsafe size.")

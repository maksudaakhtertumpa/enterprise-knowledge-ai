import io
import uuid

import pytest

from app.db.models import FileType
from app.services.ingestion.validation import (
    UploadRejected,
    file_type_for,
    sanitize_filename,
    verify_content,
)
from app.services.storage import FileTooLarge, LocalFileStorage
from tests.factories import make_docx, make_pdf, write


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("report.pdf", "report.pdf"),
        ("../../etc/passwd.txt", "passwd.txt"),
        ("C:\\Users\\me\\secret.docx", "secret.docx"),
        ("na<me>:?.txt", "na_me___.txt"),
        ("tab\there.pdf", "tab_here.pdf"),
        ("  spaced.pdf  ", "spaced.pdf"),
    ],
)
def test_sanitize_filename(raw, expected):
    assert sanitize_filename(raw) == expected


def test_overlong_filename_keeps_extension():
    name = sanitize_filename("a" * 400 + ".pdf")
    assert len(name) <= 255 and name.endswith(".pdf")


@pytest.mark.parametrize("raw", [None, "", "..", "/"])
def test_unusable_filenames_rejected(raw):
    with pytest.raises(UploadRejected):
        sanitize_filename(raw)


@pytest.mark.parametrize(
    ("name", "expected"),
    [("a.PDF", FileType.PDF), ("b.docx", FileType.DOCX), ("c.Txt", FileType.TXT)],
)
def test_allowed_extensions(name, expected):
    assert file_type_for(name) is expected


@pytest.mark.parametrize("name", ["x.exe", "x.doc", "x.pdf.exe", "noext", "x.html"])
def test_disallowed_extensions(name):
    with pytest.raises(UploadRejected) as err:
        file_type_for(name)
    assert err.value.code == "unsupported_type"


@pytest.mark.parametrize(
    ("content", "file_type"),
    [
        (b"just text", FileType.PDF),  # text renamed .pdf
        (b"%PDF-1.7", FileType.DOCX),  # pdf renamed .docx
        (b"MZ\x90\x00\x03\x00", FileType.TXT),  # Windows executable renamed .txt
        ("utf16".encode("utf-16"), FileType.TXT),
    ],
)
def test_content_must_match_claimed_type(tmp_path, content, file_type):
    path = write(tmp_path, "f", content)
    with pytest.raises(UploadRejected) as err:
        verify_content(path, file_type, max_docx_uncompressed=10**8)
    assert err.value.code == "content_mismatch"


def test_plain_zip_is_not_a_docx(tmp_path):
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("hello.txt", "hi")
    with pytest.raises(UploadRejected):
        verify_content(
            write(tmp_path, "z", buf.getvalue()), FileType.DOCX, max_docx_uncompressed=10**8
        )


def test_real_files_pass_content_check(tmp_path):
    verify_content(
        write(tmp_path, "a", make_pdf([["Hi."]])), FileType.PDF, max_docx_uncompressed=10**8
    )
    verify_content(
        write(tmp_path, "b", make_docx([("p", "Hi.")])), FileType.DOCX, max_docx_uncompressed=10**8
    )
    verify_content(write(tmp_path, "c", b"Hello"), FileType.TXT, max_docx_uncompressed=10**8)


# ------------------------------------------------------------------ storage
def ids():
    return str(uuid.uuid4()), str(uuid.uuid4())


def test_storage_saves_and_hashes(tmp_path):
    storage = LocalFileStorage(tmp_path)
    ws, doc = ids()
    stored = storage.save(
        io.BytesIO(b"hello"), workspace_id=ws, document_id=doc, ext="txt", max_bytes=100
    )
    assert stored.size_bytes == 5
    assert stored.sha256 == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert storage.path_for(stored.key).read_bytes() == b"hello"


def test_storage_aborts_oversize_and_leaves_nothing(tmp_path):
    storage = LocalFileStorage(tmp_path)
    ws, doc = ids()
    with pytest.raises(FileTooLarge):
        storage.save(
            io.BytesIO(b"x" * 101), workspace_id=ws, document_id=doc, ext="txt", max_bytes=100
        )
    assert not any(p.is_file() for p in tmp_path.rglob("*"))


@pytest.mark.parametrize(
    "key",
    ["../x.txt", "a/b.txt", "/etc/passwd", f"{uuid.uuid4()}/../../{uuid.uuid4()}.txt"],
)
def test_storage_rejects_malformed_keys(tmp_path, key):
    with pytest.raises(ValueError):
        LocalFileStorage(tmp_path).path_for(key)


def test_storage_delete_is_idempotent(tmp_path):
    storage = LocalFileStorage(tmp_path)
    ws, doc = ids()
    stored = storage.save(
        io.BytesIO(b"x"), workspace_id=ws, document_id=doc, ext="txt", max_bytes=10
    )
    storage.delete(stored.key)
    storage.delete(stored.key)
    assert not storage.path_for(stored.key).exists()

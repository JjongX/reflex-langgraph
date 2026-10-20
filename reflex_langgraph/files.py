"""
The files submodule keeps the files users attach to their messages.

A file is uploaded once, gets an id, and belongs to one conversation. Messages keep only
its id and name, never its bytes: just before each model call the agent's middleware turns
the newest message's files into content the model can read (see
:class:`~reflex_langgraph.agents.AttachmentMiddleware`), and tools read any of the
conversation's files by id with :func:`attached_file`.

.. code-block:: python

    app = create_server(graph, files=LocalFileStore("data/files"))   # uploads on
    app = create_server(graph, files=S3FileStore("my-bucket"))       # or in S3

Files can be kept anywhere: :class:`FileStore` does the checking and bookkeeping, and a
subclass only moves bytes (:class:`LocalFileStore`, :class:`S3FileStore`, or your own).

By default the formats accepted (:data:`FORMATS`) are images (PNG, JPEG, WebP, GIF), PDF
and text (TXT, Markdown, CSV, TSV, JSON). An app adds its own, e.g. YAML or compressed
CSV, as :class:`FileFormat` entries: ``LocalFileStore(root, formats=[*FORMATS,
...])``. A file's first bytes must match its format. Text formats can be shown to the
model; ``data`` formats are described to it and read by tools.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import unicodedata
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Callable, Dict, List, Optional, Sequence, Tuple, Union

#: Where the server puts its file store in a run's ``config["configurable"]``.
FILES_KEY = "rlg_files"
#: Where a user message lists its files, in ``additional_kwargs``.
ATTACHMENTS_KEY = "rlg_files"

@dataclass(frozen=True)
class FileFormat:
    """
    A kind of file users may attach.

    A format only says what is accepted and how the model is shown it. Reading the file
    (unzipping a ``.csv.gz``, parsing a YAML file) is the job of the app's tools, which get the
    file with :func:`attached_file`.

    Attributes
    ----------
    extensions : tuple of str
        Its extensions, lower case with the dot; two-part ones (``.csv.gz``) work.
    type : str
        The type it is stored and sent as, e.g. ``text/yaml``.
    kind : {'image', 'pdf', 'text', 'data'}
        How a model takes it: as content (``image``, ``pdf``, for models that read them),
        as text when short (``text``), or only described, for tools to read (``data``).
    check : str or callable
        What its first bytes must be: a name in :data:`CHECKS` (e.g. ``"gzip"``,
        ``"text"``), or ``check(head: bytes) -> bool``. Several formats may share an
        extension; the first whose check passes is used.

    Examples
    --------

    >>> from reflex_langgraph.files import FORMATS, FileFormat, LocalFileStore
    >>> yaml = FileFormat((".yaml", ".yml"), "text/yaml", "text", "text")
    >>> store = LocalFileStore("/tmp/rlg-files-doctest", formats=[*FORMATS, yaml])
    >>> ".yml" in store.accepted()["text/yaml"]
    True
    """

    extensions: Tuple[str, ...]
    type: str
    kind: str = "data"
    check: Union[str, Callable[[bytes], bool]] = "any"


def _is_text(head: bytes) -> bool:
    if b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError as exc:  # fine only if the sample cut a character in two
        return exc.reason == "unexpected end of data"
    return True


#: Named checks on a file's first bytes, for :attr:`FileFormat.check`.
CHECKS: Dict[str, Callable[[bytes], bool]] = {
    "png": lambda h: h.startswith(b"\x89PNG\r\n\x1a\n"),
    "jpeg": lambda h: h.startswith(b"\xff\xd8\xff"),
    "gif": lambda h: h.startswith((b"GIF87a", b"GIF89a")),
    "webp": lambda h: h[:4] == b"RIFF" and h[8:12] == b"WEBP",
    "pdf": lambda h: h.startswith(b"%PDF-"),
    "text": _is_text,
    "gzip": lambda h: h.startswith(b"\x1f\x8b"),
    "zip": lambda h: h.startswith(b"PK\x03\x04"),
    "hdf5": lambda h: h.startswith(b"\x89HDF\r\n\x1a\n"),
    "parquet": lambda h: h.startswith(b"PAR1"),
    "arrow": lambda h: h.startswith(b"ARROW1"),
    "any": lambda h: True,
}

#: What the package accepts by default: images, PDF and text. An app adds its own formats
#: with ``LocalFileStore(root, formats=[*FORMATS, ...])``.
FORMATS: List[FileFormat] = [
    FileFormat((".png",), "image/png", "image", "png"),
    FileFormat((".jpg", ".jpeg"), "image/jpeg", "image", "jpeg"),
    FileFormat((".webp",), "image/webp", "image", "webp"),
    FileFormat((".gif",), "image/gif", "image", "gif"),
    FileFormat((".pdf",), "application/pdf", "pdf", "pdf"),
    FileFormat((".txt",), "text/plain", "text", "text"),
    FileFormat((".md",), "text/markdown", "text", "text"),
    FileFormat((".csv",), "text/csv", "text", "text"),
    FileFormat((".tsv", ".tab"), "text/tab-separated-values", "text", "text"),
    FileFormat((".json",), "application/json", "text", "text"),
]

_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_CHUNK_BYTES = 1 << 20


class FileError(ValueError):
    """Raised when a file is refused (an unsupported or mismatched type, or an unknown id)."""


class FileTooLarge(FileError):
    """Raised when a file is larger than allowed."""


def accepted_types(formats: Sequence[FileFormat]) -> Dict[str, List[str]]:
    """
    Return the extensions of each type, as a file picker's ``accept`` wants them.

    Parameters
    ----------
    formats : sequence of FileFormat
        The formats.

    Returns
    -------
    dict
        Type to its extensions.

    Examples
    --------

    >>> from reflex_langgraph.files import FORMATS, accepted_types
    >>> accepted_types(FORMATS)['image/jpeg']
    ['.jpg', '.jpeg']
    """
    out: Dict[str, List[str]] = {}
    for fmt in formats:
        for ext in fmt.extensions:
            if ext not in out.setdefault(fmt.type, []):
                out[fmt.type].append(ext)
    return out


def match_format(name: str, head: bytes, formats: Sequence[FileFormat]) -> FileFormat:
    """
    Return the format of a file, from its name and first bytes.

    The longest extension the name ends with decides the candidates (so ``.csv.gz``
    wins over ``.gz``); the first of them whose check passes is the format.

    Parameters
    ----------
    name : str
        The file's name.
    head : bytes
        Its first bytes (4 KB is enough).
    formats : sequence of FileFormat
        The formats accepted.

    Returns
    -------
    FileFormat
        Its format.

    Raises
    ------
    FileError
        If no format has its extension, or its contents match none of them.

    Examples
    --------

    >>> from reflex_langgraph.files import FORMATS, match_format
    >>> match_format('notes.TXT', b'hello', FORMATS).type
    'text/plain'
    """
    lower = name.lower()
    ext = max((e for f in formats for e in f.extensions if lower.endswith(e)), key=len,
              default="")
    if not ext:
        raise FileError(f"{name}: this file type is not accepted")
    candidates = [f for f in formats if ext in f.extensions]
    for fmt in candidates:
        check = CHECKS[fmt.check] if isinstance(fmt.check, str) else fmt.check
        if check(head):
            return fmt
    raise FileError(f"{name}: its contents do not match its type ({candidates[0].type})")


def file_kind(file_type: str) -> str:
    """
    Return how a model takes a file of one of the default types.

    Files keep their kind when they are stored (:attr:`FileRecord.kind`); this answers for
    the default :data:`FORMATS` only.

    Parameters
    ----------
    file_type : str
        The file's type, e.g. ``application/pdf``.

    Returns
    -------
    str
        ``image``, ``pdf``, ``text``, or ``data`` for types it does not know.

    Examples
    --------

    >>> from reflex_langgraph.files import file_kind
    >>> file_kind('image/png'), file_kind('text/csv'), file_kind('application/zip')
    ('image', 'text', 'data')
    """
    return next((f.kind for f in FORMATS if f.type == file_type), "data")


def clean_name(name: str) -> str:
    """
    Return a safe display name for an uploaded file.

    Only the last path part is kept, without control characters, at most 200 characters.

    Parameters
    ----------
    name : str
        The name the browser sent.

    Returns
    -------
    str
        The name to show.

    Examples
    --------

    >>> from reflex_langgraph.files import clean_name
    >>> clean_name('../../etc/report\\x00.pdf')
    'report.pdf'
    """
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(ch for ch in name if unicodedata.category(ch)[0] != "C").strip()
    return (name or "file")[-200:]


@dataclass
class FileRecord:
    """
    A file attached in a conversation.

    Attributes
    ----------
    id : str
        Its id, e.g. ``f_3b9c...``.
    thread_id : str
        The conversation it belongs to.
    name : str
        Its name, as uploaded.
    type : str
        Its type, e.g. ``application/pdf``.
    size : int
        Its size in bytes.
    created_at : str
        When it was uploaded (ISO 8601, UTC).
    kind : str
        How a model takes it (see :attr:`FileFormat.kind`).
    path : str, optional
        Where it is stored, when the store keeps it on disk.
    """

    id: str
    thread_id: str
    name: str
    type: str
    size: int
    created_at: str
    kind: str = "data"
    path: Optional[str] = None

    def ref(self) -> dict:
        """
        Return what a message keeps of the file.

        Returns
        -------
        dict
            ``id``, ``name``, ``type``, ``kind`` and ``size``: no bytes, no path.
        """
        return {"id": self.id, "name": self.name, "type": self.type, "kind": self.kind,
                "size": self.size}


class FileStore:
    """
    Keeps the files attached in conversations; subclass it for a place to keep them.

    This class does the work every store shares: checking a file's name, size and contents
    against the accepted formats, giving it an id, and keeping its name and type beside it.
    A subclass only moves bytes, by implementing :meth:`write_file`, :meth:`write_bytes`,
    :meth:`read_bytes`, :meth:`list_keys` and :meth:`remove_keys` (and optionally
    :meth:`download` and :meth:`local_path`). Keys look like ``<thread_id>/<file_id>.bin``.
    The package ships :class:`LocalFileStore` and :class:`S3FileStore`.

    .. code-block:: python

        class MyCloudStore(FileStore):
            def write_file(self, key, source): ...   # upload a local file
            def write_bytes(self, key, data): ...    # store a small object
            def read_bytes(self, key): ...           # its bytes, or None when missing
            def list_keys(self, prefix): ...         # keys starting with prefix
            def remove_keys(self, keys): ...         # delete them

    Parameters
    ----------
    formats : sequence of FileFormat, optional
        The formats accepted. By default, :data:`FORMATS` (images, PDF, text); an app adds
        its own with ``formats=[*FORMATS, ...]``.
    cache_dir : str or pathlib.Path, optional
        Where tools get local copies of files kept elsewhere (see :meth:`local_path`). By
        default, a folder in the system's temporary directory.
    """

    def __init__(self, formats: Optional[Sequence[FileFormat]] = None,
                 cache_dir: Union[str, Path, None] = None):
        self.formats = list(FORMATS if formats is None else formats)
        self.cache_dir = Path(cache_dir) if cache_dir else \
            Path(tempfile.gettempdir()) / "reflex-langgraph-files"

    # --- what a subclass implements ------------------------------------------------------

    def write_file(self, key: str, source: Path) -> None:
        """
        Store a local file under a key; ``source`` may be moved or deleted afterwards.

        Parameters
        ----------
        key : str
            Where to keep it.
        source : pathlib.Path
            The local file.
        """
        raise NotImplementedError

    def write_bytes(self, key: str, data: bytes) -> None:
        """
        Store a small object (a file's name and type) under a key.

        Parameters
        ----------
        key : str
            Where to keep it.
        data : bytes
            Its contents.
        """
        raise NotImplementedError

    def read_bytes(self, key: str) -> Optional[bytes]:
        """
        Return what is stored under a key.

        Parameters
        ----------
        key : str
            The key.

        Returns
        -------
        bytes or None
            Its contents; ``None`` when there is nothing under it.
        """
        raise NotImplementedError

    def list_keys(self, prefix: str) -> List[str]:
        """
        Return the keys that start with a prefix.

        Parameters
        ----------
        prefix : str
            E.g. ``"<thread_id>/"``.

        Returns
        -------
        list of str
            The keys.
        """
        raise NotImplementedError

    def remove_keys(self, keys: Sequence[str]) -> None:
        """
        Delete what is stored under these keys; missing ones are ignored.

        Parameters
        ----------
        keys : sequence of str
            The keys.
        """
        raise NotImplementedError

    def download(self, key: str, dest: Path) -> None:
        """
        Copy what is stored under a key to a local file.

        By default it reads the whole object; override it to stream large files.

        Parameters
        ----------
        key : str
            The key.
        dest : pathlib.Path
            The local file to write.
        """
        data = self.read_bytes(key)
        if data is None:
            raise FileError(f"nothing stored under {key}")
        dest.write_bytes(data)

    def staging_dir(self) -> Path:
        """
        Return where an upload is written before it is checked and stored.

        Returns
        -------
        pathlib.Path
            A local folder (by default, under :attr:`cache_dir`).
        """
        folder = self.cache_dir / ".staging"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    # --- what the server and tools use ---------------------------------------------------

    @staticmethod
    def _key(thread_id: str, file_id: str, suffix: str) -> str:
        if not (_ID.match(thread_id or "") and _ID.match(file_id or "")):
            raise FileError("invalid conversation or file id")
        return f"{thread_id}/{file_id}{suffix}"

    def accepted(self) -> Dict[str, List[str]]:
        """
        Return the formats it accepts, for the UI's file picker.

        Returns
        -------
        dict
            Type to its extensions (see :func:`accepted_types`).
        """
        return accepted_types(self.formats)

    def save(self, thread_id: str, name: str, data: BinaryIO, max_bytes: int) -> FileRecord:
        """
        Store a file for a conversation.

        Parameters
        ----------
        thread_id : str
            The conversation.
        name : str
            The file's name, as uploaded.
        data : file object
            Its contents, read in chunks.
        max_bytes : int
            The largest size allowed.

        Returns
        -------
        FileRecord
            The stored file.

        Raises
        ------
        FileError
            If its type is not accepted or does not match its contents.
        FileTooLarge
            If it is larger than ``max_bytes``.
        """
        name = clean_name(name)
        if not any(name.lower().endswith(e) for f in self.formats for e in f.extensions):
            raise FileError(f"{name}: this file type is not accepted")  # before storing it
        file_id = "f_" + uuid.uuid4().hex[:16]
        key = self._key(thread_id, file_id, ".bin")
        fd, staged = tempfile.mkstemp(dir=self.staging_dir(), suffix=".part")
        part, size, head = Path(staged), 0, b""
        try:
            with os.fdopen(fd, "wb") as out:
                while chunk := data.read(_CHUNK_BYTES):
                    size += len(chunk)
                    if size > max_bytes:
                        raise FileTooLarge(f"{name} is larger than {max_bytes // (1 << 20)} MB")
                    if len(head) < 4096:
                        head += chunk[:4096 - len(head)]
                    out.write(chunk)
            fmt = match_format(name, head, self.formats)
            self.write_file(key, part)
        finally:
            part.unlink(missing_ok=True)
        record = FileRecord(id=file_id, thread_id=thread_id, name=name, type=fmt.type,
                            size=size, created_at=datetime.now(timezone.utc).isoformat(),
                            kind=fmt.kind)
        meta = {k: v for k, v in asdict(record).items() if k != "path"}
        self.write_bytes(self._key(thread_id, file_id, ".json"), json.dumps(meta).encode())
        return record

    def get(self, thread_id: str, file_id: str) -> Optional[FileRecord]:
        """
        Return a conversation's file.

        Parameters
        ----------
        thread_id : str
            The conversation.
        file_id : str
            The file.

        Returns
        -------
        FileRecord or None
            The file; ``None`` when unknown in this conversation.
        """
        try:
            meta = self.read_bytes(self._key(thread_id, file_id, ".json"))
        except FileError:
            return None
        return FileRecord(**json.loads(meta)) if meta else None

    def read(self, record: FileRecord) -> bytes:
        """
        Return a file's contents.

        Parameters
        ----------
        record : FileRecord
            The file.

        Returns
        -------
        bytes
            Its contents.
        """
        data = self.read_bytes(self._key(record.thread_id, record.id, ".bin"))
        if data is None:
            raise FileError(f"{record.name}: its contents are missing")
        return data

    def local_path(self, record: FileRecord) -> Path:
        """
        Return a local file with a file's contents, for tools that read from disk.

        A store that keeps files elsewhere downloads it once into :attr:`cache_dir`.

        Parameters
        ----------
        record : FileRecord
            The file.

        Returns
        -------
        pathlib.Path
            The local file; it keeps the original name's extension.
        """
        folder = self.cache_dir / record.thread_id
        dest = folder / f"{record.id}__{record.name}"
        if not dest.is_file():
            folder.mkdir(parents=True, exist_ok=True)
            fd, partial = tempfile.mkstemp(dir=folder, suffix=".part")
            os.close(fd)
            try:
                self.download(self._key(record.thread_id, record.id, ".bin"), Path(partial))
                os.replace(partial, dest)
            finally:
                Path(partial).unlink(missing_ok=True)
        return dest

    def list(self, thread_id: str) -> List[FileRecord]:
        """
        Return a conversation's files, oldest first.

        Parameters
        ----------
        thread_id : str
            The conversation.

        Returns
        -------
        list of FileRecord
            Its files.
        """
        if not _ID.match(thread_id or ""):
            raise FileError("invalid conversation id")
        ids = [k.rsplit("/", 1)[-1][:-len(".json")]
               for k in self.list_keys(f"{thread_id}/") if k.endswith(".json")]
        records = [self.get(thread_id, i) for i in ids]
        return sorted((r for r in records if r), key=lambda r: r.created_at)

    def delete(self, thread_id: str, file_id: str) -> None:
        """
        Remove a file; nothing happens when it is unknown.

        Parameters
        ----------
        thread_id : str
            The conversation.
        file_id : str
            The file.
        """
        try:
            keys = [self._key(thread_id, file_id, s) for s in (".bin", ".json")]
        except FileError:
            return
        self.remove_keys(keys)
        for copy in (self.cache_dir / thread_id).glob(f"{file_id}__*"):
            copy.unlink(missing_ok=True)

    def delete_thread(self, thread_id: str) -> None:
        """
        Remove every file of a conversation.

        Parameters
        ----------
        thread_id : str
            The conversation.
        """
        if not _ID.match(thread_id or ""):
            raise FileError("invalid conversation id")
        self.remove_keys(self.list_keys(f"{thread_id}/"))
        shutil.rmtree(self.cache_dir / thread_id, ignore_errors=True)


class LocalFileStore(FileStore):
    """
    Keeps attached files in a folder, one subfolder per conversation.

    Each file is saved under a random id, with its name and type beside it in a small JSON
    file; nothing is served from the folder. Tools read the files where they are.

    Parameters
    ----------
    root : str or pathlib.Path
        The folder; it is created if needed.
    formats : sequence of FileFormat, optional
        The formats accepted (see :class:`FileStore`).

    Examples
    --------

    >>> import io, tempfile
    >>> from reflex_langgraph.files import LocalFileStore
    >>> store = LocalFileStore(tempfile.mkdtemp())
    >>> record = store.save('t1', 'notes.txt', io.BytesIO(b'hello'), max_bytes=1000)
    >>> record.type, record.size, store.read(record)
    ('text/plain', 5, b'hello')
    >>> [r.name for r in store.list('t1')]
    ['notes.txt']
    """

    def __init__(self, root: Union[str, Path], formats: Optional[Sequence[FileFormat]] = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        super().__init__(formats, cache_dir=self.root / ".cache")

    def staging_dir(self) -> Path:
        folder = self.root / ".staging"  # same disk as the files: a rename stores them
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _path(self, key: str) -> Path:
        return self.root / key

    def write_file(self, key: str, source: Path) -> None:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, dest)

    def write_bytes(self, key: str, data: bytes) -> None:
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    def read_bytes(self, key: str) -> Optional[bytes]:
        path = self._path(key)
        return path.read_bytes() if path.is_file() else None

    def list_keys(self, prefix: str) -> List[str]:
        folder = self._path(prefix)
        if not folder.is_dir():
            return []
        return [f"{prefix}{p.name}" for p in folder.iterdir() if p.is_file()]

    def remove_keys(self, keys: Sequence[str]) -> None:
        folders = set()
        for key in keys:
            self._path(key).unlink(missing_ok=True)
            folders.add(self._path(key).parent)
        for folder in folders:
            try:
                folder.rmdir()  # only when empty
            except OSError:
                pass

    def local_path(self, record: FileRecord) -> Path:
        return self._path(self._key(record.thread_id, record.id, ".bin"))


class S3FileStore(FileStore):
    """
    Keeps attached files in an S3 bucket (or any S3-compatible store, e.g. MinIO).

    Objects go under ``<prefix><thread_id>/<file_id>.bin``, with the name and type in a
    ``.json`` beside each. Tools get a local copy, downloaded once into ``cache_dir``.
    Needs ``pip install "reflex-langgraph[s3]"`` (boto3), unless you pass a client.

    .. code-block:: python

        files = S3FileStore("my-bucket", prefix="chat-files/", region_name="ap-northeast-2")
        app = create_server(graph, files=files)

    Parameters
    ----------
    bucket : str
        The bucket.
    prefix : str, default: ''
        Prepended to every key, e.g. ``"chat-files/"``.
    formats : sequence of FileFormat, optional
        The formats accepted (see :class:`FileStore`).
    cache_dir : str or pathlib.Path, optional
        Where local copies for tools go.
    client : object, optional
        A boto3 S3 client. By default, ``boto3.client("s3", **client_options)``; the
        credentials come from the usual AWS settings (environment, profile, instance role).
    **client_options
        Passed to ``boto3.client``, e.g. ``region_name``, ``endpoint_url``.
    """

    def __init__(self, bucket: str, prefix: str = "", *,
                 formats: Optional[Sequence[FileFormat]] = None,
                 cache_dir: Union[str, Path, None] = None, client=None, **client_options):
        super().__init__(formats, cache_dir)
        self.bucket, self.prefix = bucket, prefix
        if client is None:
            try:
                import boto3
            except ImportError:
                raise ImportError('S3FileStore needs: pip install "reflex-langgraph[s3]"') from None
            client = boto3.client("s3", **client_options)
        self.client = client

    @staticmethod
    def _missing(exc: Exception) -> bool:
        code = (getattr(exc, "response", None) or {}).get("Error", {}).get("Code")
        return code in ("NoSuchKey", "404", "NotFound")

    def write_file(self, key: str, source: Path) -> None:
        self.client.upload_file(str(source), self.bucket, self.prefix + key)

    def write_bytes(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self.prefix + key, Body=data)

    def read_bytes(self, key: str) -> Optional[bytes]:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=self.prefix + key)["Body"].read()
        except Exception as exc:  # noqa: BLE001 - boto3 raises ClientError subclasses
            if self._missing(exc):
                return None
            raise

    def download(self, key: str, dest: Path) -> None:
        try:
            self.client.download_file(self.bucket, self.prefix + key, str(dest))
        except Exception as exc:  # noqa: BLE001
            if self._missing(exc):
                raise FileError(f"nothing stored under {key}") from None
            raise

    def list_keys(self, prefix: str) -> List[str]:
        pages = self.client.get_paginator("list_objects_v2").paginate(
            Bucket=self.bucket, Prefix=self.prefix + prefix)
        return [obj["Key"][len(self.prefix):] for page in pages
                for obj in page.get("Contents", [])]

    def remove_keys(self, keys: Sequence[str]) -> None:
        keys = list(keys)
        for i in range(0, len(keys), 1000):  # S3 deletes at most 1000 per request
            self.client.delete_objects(Bucket=self.bucket, Delete={
                "Objects": [{"Key": self.prefix + k} for k in keys[i:i + 1000]],
                "Quiet": True})


def attached_file(config: dict, file_id: str) -> FileRecord:
    """
    Return a file of the current conversation, for a tool.

    A tool takes ``config: RunnableConfig`` and an id the agent passes it; the file must
    belong to the conversation the tool runs in.

    .. code-block:: python

        @registry.tool
        def count_rows(file_id: str, config: RunnableConfig) -> str:
            \"\"\"Count the rows of an attached CSV file.\"\"\"
            record = attached_file(config, file_id)
            with open(record.path) as f:
                return str(sum(1 for _ in f) - 1)

    Parameters
    ----------
    config : dict
        The tool's run config.
    file_id : str
        The file's id, as the message's file label shows it.

    Returns
    -------
    FileRecord
        The file; ``path`` is a local file with its contents, wherever the store keeps it
        (a store such as S3 downloads it first).

    Raises
    ------
    FileError
        If uploads are off on this server, or the conversation has no such file.
    """
    configurable = (config or {}).get("configurable") or {}
    store, thread_id = configurable.get(FILES_KEY), configurable.get("thread_id")
    if store is None:
        raise FileError("this server does not keep attached files")
    record = store.get(thread_id, file_id)
    if record is None:
        raise FileError(f"no file {file_id!r} in this conversation")
    record.path = str(store.local_path(record))  # downloaded first if kept elsewhere
    return record

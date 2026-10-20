"""Files attached to messages: the store, the upload routes, what the model is shown, files
handed to workers, and tools reading them by id."""

import io

import pytest
from langchain_core.runnables import RunnableConfig

from reflex_langgraph import (FileError, FileStore, LocalFileStore, S3FileStore, ToolRegistry,
                              attached_file, build_graph, create_server, load_config)
from reflex_langgraph.files import FORMATS, FileFormat
from reflex_langgraph.agents import AttachmentMiddleware
from reflex_langgraph.files import FileTooLarge

from conftest import client_for, sse_events
from scripted import ScriptedChatModel, call

pytestmark = pytest.mark.anyio

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
PDF = b"%PDF-1.4\n%fake but well-formed enough\n"
CSV = b"region,units\nNorth,10\nSouth,7\n"
MB = 1 << 20


# --- the store -----------------------------------------------------------------------------

def test_the_store_keeps_checks_and_removes_files(tmp_path):
    store = LocalFileStore(tmp_path)
    png = store.save("t1", "chart.png", io.BytesIO(PNG), max_bytes=MB)
    csv = store.save("t1", "../../etc/sales.csv", io.BytesIO(CSV), max_bytes=MB)
    assert (png.type, csv.name, csv.type) == ("image/png", "sales.csv", "text/csv")
    assert store.read(store.get("t1", png.id)) == PNG
    assert [r.name for r in store.list("t1")] == ["chart.png", "sales.csv"]
    assert store.get("t2", png.id) is None                    # another conversation's id

    for name, data, error in [("tool.exe", b"MZ", FileError),               # type not accepted
                              ("fake.png", b"not an image", FileError),     # contents mismatch
                              ("big.csv", b"x" * (MB + 1), FileTooLarge)]:
        with pytest.raises(error):
            store.save("t1", name, io.BytesIO(data), max_bytes=MB)
    with pytest.raises(FileError):
        store.save("../escape", "a.txt", io.BytesIO(b"x"), max_bytes=MB)
    assert len(store.list("t1")) == 2 and not list((tmp_path / "t1").glob("*.part"))

    store.delete("t1", png.id)
    assert [r.name for r in store.list("t1")] == ["sales.csv"]
    store.delete_thread("t1")
    assert store.list("t1") == [] and not (tmp_path / "t1").exists()


# Formats an app adds.
APP_FORMATS = [*FORMATS,
               FileFormat((".yaml",), "text/yaml", "text", "text"),
               FileFormat((".csv.gz",), "application/gzip", "data", "gzip"),
               FileFormat((".ts",), "text/typescript", "text", "text"),
               FileFormat((".ts",), "video/mp2t", "data",
                          lambda head: head.startswith(b"\x47"))]


def test_the_package_accepts_only_general_formats_unless_the_app_adds_some(tmp_path):
    plain = LocalFileStore(tmp_path / "plain")
    with pytest.raises(FileError, match="not accepted"):
        plain.save("t", "config.yaml", io.BytesIO(b"name: demo\n"), max_bytes=MB)
    assert "text/yaml" not in plain.accepted() and plain.accepted()["text/csv"] == [".csv"]

    store = LocalFileStore(tmp_path / "app", formats=APP_FORMATS)
    save = lambda name, data: store.save("t", name, io.BytesIO(data), max_bytes=MB)
    assert (save("config.yaml", b"name: demo\n").type, save("table.CSV.gz", b"\x1f\x8b\x08").kind) \
        == ("text/yaml", "data")                           # two-part extension wins over .gz
    assert save("app.ts", b"const x: number = 1;\n").type == "text/typescript"
    assert save("clip.ts", b"\x47\x40\x00\x10\x00").type == "video/mp2t"
    with pytest.raises(FileError, match="do not match"):
        save("table.csv.gz", b"name,value\n")
    assert store.accepted()["application/gzip"] == [".csv.gz"]


async def test_app_formats_reach_the_ui_and_the_model(tmp_path):
    model = ScriptedChatModel(script=["Ok."])
    graph = build_graph(load_config({"model": "x:y", "agents": [{"name": "a"}]}), models={"a": model})
    app = create_server(graph, files=LocalFileStore(tmp_path / "files", formats=APP_FORMATS))
    async with client_for(app) as c:
        assert (await c.get("/health")).json()["accept"]["text/yaml"] == [".yaml"]
        tid = (await c.post("/threads")).json()["thread_id"]
        cfg = await upload(c, tid, "config.yaml", b"name: demo\nreplicas: 3\n")
        gz = await upload(c, tid, "table.csv.gz", b"\x1f\x8b\x08\x00")
        assert (cfg["kind"], gz["kind"]) == ("text", "data")
        await sse_events(c, f"/threads/{tid}/messages",
                         {"content": "Check", "files": [cfg["file_id"], gz["file_id"]]})
    content = model.inputs[0][-1].content
    assert [b["type"] for b in content] == ["text", "text"] and "replicas: 3" in content[1]["text"]
    assert "tools can read it by its id" in content[0]["text"]


# --- the server and what the model sees -------------------------------------------------------

def app_with(script, tmp_path, *, reads=("image", "pdf"), **server_kw):
    model = {"id": "x:y", "reads": list(reads)}
    model_ = ScriptedChatModel(script=script)
    graph = build_graph(load_config({"model": model, "agents": [{"name": "a"}]}),
                        models={"a": model_})
    return create_server(graph, files=LocalFileStore(tmp_path / "files"), **server_kw), model_


async def upload(c, tid, name, data):
    resp = await c.post(f"/threads/{tid}/files", files={"file": (name, data)})
    assert resp.status_code == 200, resp.text
    return resp.json()


async def test_uploads_are_off_unless_the_server_keeps_files(tmp_path):
    graph = build_graph(load_config({"model": "x:y", "agents": [{"name": "a"}]}),
                        models={"a": ScriptedChatModel(script=[])})
    async with client_for(create_server(graph)) as c:
        assert (await c.get("/health")).json()["files"] is False
        tid = (await c.post("/threads")).json()["thread_id"]
        assert (await c.post(f"/threads/{tid}/files", files={"file": ("a.txt", b"x")})).status_code == 404


async def test_the_model_reads_the_files_of_the_message_it_answers(tmp_path):
    app, model = app_with(["Read them.", "Next."], tmp_path, max_file_mb=1)
    async with client_for(app) as c:
        health = (await c.get("/health")).json()
        assert (health["files"], health["max_file_mb"], health["max_files"]) == (True, 1, 30)
        tid = (await c.post("/threads")).json()["thread_id"]
        files = [await upload(c, tid, n, d) for n, d in
                 [("chart.png", PNG), ("report.pdf", PDF), ("sales.csv", CSV)]]
        assert (await c.post(f"/threads/{tid}/files",
                             files={"file": ("big.txt", b"x" * (MB + 1))})).status_code == 413
        assert (await c.post(f"/threads/{tid}/files",
                             files={"file": ("a.exe", b"MZ")})).status_code == 415

        events = await sse_events(c, f"/threads/{tid}/messages",
                                  {"content": "", "files": [f["file_id"] for f in files]})
        assert events[-1][1]["status"] == "completed"
        content = model.inputs[0][-1].content
        assert [b["type"] for b in content] == ["text", "image", "file", "text"]
        assert "file id " + files[0]["file_id"] in content[0]["text"]
        assert content[2]["mime_type"] == "application/pdf" and "North,10" in content[3]["text"]

        await sse_events(c, f"/threads/{tid}/messages", {"content": "And now?"})
        earlier = model.inputs[1][0].content            # the first message, a turn later
        assert [b["type"] for b in earlier] == ["text"] and "report.pdf" in earlier[0]["text"]

        history = (await c.get(f"/threads/{tid}")).json()
        assert [f["name"] for f in history["messages"][0]["files"]] == [
            "chart.png", "report.pdf", "sales.csv"]
        assert (await c.get("/threads")).json()[0]["title"] == "chart.png"   # no text: a file

        for body, code in [({"content": ""}, 422),
                           ({"content": "x", "files": ["f_unknown"]}, 422)]:
            assert (await c.post(f"/threads/{tid}/messages", json=body)).status_code == code


async def test_files_a_model_cannot_take_are_described_instead(tmp_path):
    app, model = app_with(["Ok."], tmp_path, reads=())
    async with client_for(app) as c:
        tid = (await c.post("/threads")).json()["thread_id"]
        png = await upload(c, tid, "chart.png", PNG)
        await sse_events(c, f"/threads/{tid}/messages", {"content": "Look", "files": [png["file_id"]]})
    content = model.inputs[0][-1].content
    assert [b["type"] for b in content] == ["text"]
    assert "cannot read image files directly" in content[0]["text"]

    store = LocalFileStore(tmp_path / "limits")
    big = store.save("t", "big.png", io.BytesIO(PNG + b"\x00" * 4096), max_bytes=MB).ref()
    long_text = store.save("t", "long.txt", io.BytesIO(b"y" * 200), max_bytes=MB).ref()
    from langchain_core.messages import HumanMessage
    mw = AttachmentMiddleware(["image"], max_inline_mb=0.001, max_text_chars=100)
    blocks = mw._content(HumanMessage("see"), [big, long_text], True, store, "t")
    assert [b["type"] for b in blocks] == ["text"]
    assert "Too large to send" in blocks[0]["text"] and "Too long to include" in blocks[0]["text"]


async def test_try_again_keeps_the_files_and_delete_removes_them(tmp_path):
    app, model = app_with(["First.", "Second."], tmp_path)
    async with client_for(app) as c:
        tid = (await c.post("/threads")).json()["thread_id"]
        png = await upload(c, tid, "chart.png", PNG)
        await sse_events(c, f"/threads/{tid}/messages", {"content": "Hi", "files": [png["file_id"]]})
        await sse_events(c, f"/threads/{tid}/regenerate", {})
        assert [b["type"] for b in model.inputs[1][-1].content] == ["text", "image"]
        assert (tmp_path / "files" / tid).is_dir()
        await c.delete(f"/threads/{tid}")
    assert not (tmp_path / "files" / tid).exists()


# --- workers and tools ------------------------------------------------------------------------

TEAM = {"model": {"id": "x:y", "reads": ["image", "pdf"]}, "entry": "boss", "agents": [
    {"name": "boss", "workers": ["reader"]},
    {"name": "reader", "description": "Reads documents.", "tools": ["count_rows"]}]}


async def test_a_worker_gets_the_files_it_is_handed_and_tools_read_them_by_id(tmp_path):
    registry, seen = ToolRegistry(), []

    @registry.tool
    def count_rows(file_id: str, config: RunnableConfig) -> str:
        """Count the data rows of an attached CSV file."""
        try:
            record = attached_file(config, file_id)
        except FileError as exc:
            return f"error: {exc}"
        with open(record.path) as f:
            seen.append(record.name)
            return str(sum(1 for _ in f) - 1)

    boss = ScriptedChatModel(script=["placeholder"])
    reader = ScriptedChatModel(script=[])
    graph = build_graph(load_config(TEAM), registry, models={"boss": boss, "reader": reader})
    app = create_server(graph, files=LocalFileStore(tmp_path / "files"))
    async with client_for(app) as c:
        tid = (await c.post("/threads")).json()["thread_id"]
        pdf = await upload(c, tid, "report.pdf", PDF)
        csv = await upload(c, tid, "sales.csv", CSV)
        boss.script[:] = [call("ask_reader", task="Summarize and count.",
                               files=[pdf["file_id"], csv["file_id"]]), "Done."]
        boss.model_post_init(None)
        reader.script[:] = [call("count_rows", file_id=csv["file_id"]),
                            call("count_rows", file_id="f_nope"), "2 rows; the PDF is a report."]
        reader.model_post_init(None)
        events = await sse_events(c, f"/threads/{tid}/messages",
                                  {"content": "Read these", "files": [pdf["file_id"], csv["file_id"]]})
    assert events[-1][1]["status"] == "completed"
    task = reader.inputs[0][-1].content                 # the worker's task, with the files
    assert [b["type"] for b in task] == ["text", "file", "text"]
    assert task[0]["text"].startswith("Summarize and count.")
    assert seen == ["sales.csv"]
    outputs = [d["output"] for e, d in events if e == "tool_result" and d["tool"] == "count_rows"]
    assert outputs == ["2", "error: no file 'f_nope' in this conversation"]


def test_attached_file_needs_a_server_with_files():
    with pytest.raises(FileError, match="does not keep"):
        attached_file({"configurable": {"thread_id": "t"}}, "f_1")


# --- other places to keep files: S3, or your own --------------------------------------------

class FakeS3:
    """Just the boto3 S3 client calls S3FileStore makes, kept in a dict."""

    class Missing(Exception):
        response = {"Error": {"Code": "NoSuchKey"}}

    def __init__(self):
        self.objects = {}

    def upload_file(self, filename, bucket, key):
        self.objects[(bucket, key)] = open(filename, "rb").read()

    def put_object(self, Bucket, Key, Body):
        self.objects[(Bucket, Key)] = Body

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise self.Missing()
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def download_file(self, bucket, key, filename):
        open(filename, "wb").write(self.get_object(bucket, key)["Body"].read())

    def get_paginator(self, name):
        objects = self.objects

        class Pages:
            def paginate(self, Bucket, Prefix):
                return [{"Contents": [{"Key": k} for b, k in sorted(objects)
                                      if b == Bucket and k.startswith(Prefix)]}]
        return Pages()

    def delete_objects(self, Bucket, Delete):
        for o in Delete["Objects"]:
            self.objects.pop((Bucket, o["Key"]), None)


class DictStore(FileStore):
    """A whole custom store: five methods (how an app adds GCS, a file server...)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.data = {}

    def write_file(self, key, source):
        self.data[key] = source.read_bytes()

    def write_bytes(self, key, data):
        self.data[key] = data

    def read_bytes(self, key):
        return self.data.get(key)

    def list_keys(self, prefix):
        return [k for k in self.data if k.startswith(prefix)]

    def remove_keys(self, keys):
        for k in keys:
            self.data.pop(k, None)


@pytest.mark.parametrize("make", [
    lambda tmp: S3FileStore("bucket", prefix="chat/", client=FakeS3(), cache_dir=tmp / "cache"),
    lambda tmp: DictStore(cache_dir=tmp / "cache"),
])
def test_stores_elsewhere_keep_check_copy_and_remove_files(make, tmp_path):
    store = make(tmp_path)
    png = store.save("t1", "chart.png", io.BytesIO(PNG), max_bytes=MB)
    store.save("t1", "sales.csv", io.BytesIO(CSV), max_bytes=MB)
    with pytest.raises(FileError):
        store.save("t1", "fake.png", io.BytesIO(b"not an image"), max_bytes=MB)
    assert [r.name for r in store.list("t1")] == ["chart.png", "sales.csv"]
    assert store.read(store.get("t1", png.id)) == PNG and store.get("t2", png.id) is None
    copy = store.local_path(png)                          # for tools that read from disk
    assert copy.read_bytes() == PNG and copy.name.endswith("chart.png")
    if isinstance(store, S3FileStore):
        assert ("bucket", f"chat/t1/{png.id}.bin") in store.client.objects
    store.delete("t1", png.id)
    assert [r.name for r in store.list("t1")] == ["sales.csv"] and not copy.exists()
    store.delete_thread("t1")
    assert store.list("t1") == [] and not list(store.staging_dir().iterdir())


async def test_tools_get_a_local_copy_of_a_file_kept_in_s3(tmp_path):
    registry, seen = ToolRegistry(), []

    @registry.tool
    def count_rows(file_id: str, config: RunnableConfig) -> str:
        """Count the data rows of an attached CSV file."""
        record = attached_file(config, file_id)
        seen.append(record.path)
        with open(record.path) as f:
            return str(sum(1 for _ in f) - 1)

    model = ScriptedChatModel(script=[])
    graph = build_graph(load_config({"model": "x:y", "agents": [{"name": "a", "tools": ["count_rows"]}]}),
                        registry, models={"a": model})
    s3 = S3FileStore("bucket", client=FakeS3(), cache_dir=tmp_path / "cache")
    async with client_for(create_server(graph, files=s3)) as c:
        tid = (await c.post("/threads")).json()["thread_id"]
        csv = await upload(c, tid, "sales.csv", CSV)
        model.script[:] = [call("count_rows", file_id=csv["file_id"]), "Two rows."]
        model.model_post_init(None)
        events = await sse_events(c, f"/threads/{tid}/messages",
                                  {"content": "Count", "files": [csv["file_id"]]})
        assert [d["output"] for e, d in events if e == "tool_result"] == ["2"]
        assert seen[0].startswith(str(tmp_path / "cache"))
        await c.delete(f"/threads/{tid}")
    assert not [k for k in s3.client.objects if tid in k[1]]

import urllib.request

from tools.download_colmap_vocab_tree import (
    DEFAULT_RELEASE,
    DownloadResult,
    FAISS_FORMAT_NOTE,
    LEGACY_FORMAT_WARNING,
    download_file,
    download_vocab_trees,
    format_byte_count,
    main,
    selected_variants,
    vocab_tree_url,
)


class FakeResponse:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, _chunk_size):
        if not self.chunks:
            return b""
        return self.chunks.pop(0)


def test_vocab_tree_url_uses_official_release_asset_name() -> None:
    assert vocab_tree_url("256k") == (
        "https://github.com/colmap/colmap/releases/download/"
        f"{DEFAULT_RELEASE}/vocab_tree_faiss_flickr100K_words256K.bin"
    )


def test_vocab_tree_url_can_request_legacy_asset_name() -> None:
    assert vocab_tree_url("256k", tree_format="legacy") == (
        "https://github.com/colmap/colmap/releases/download/"
        f"{DEFAULT_RELEASE}/vocab_tree_flickr100K_words256K.bin"
    )


def test_selected_variants_expands_all_in_stable_order() -> None:
    assert selected_variants("all") == ("32k", "256k", "1m")


def test_format_byte_count_reports_readable_units() -> None:
    assert format_byte_count(512) == "512 bytes"
    assert format_byte_count(1024 * 1024) == "1.0 MiB"


def test_download_file_writes_to_destination(tmp_path) -> None:
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return FakeResponse([b"abc", b"def"])

    destination = tmp_path / "tree.bin"

    bytes_written = download_file(
        "https://example.test/tree.bin",
        destination,
        timeout_seconds=7,
        urlopen=fake_urlopen,
    )

    assert bytes_written == 6
    assert destination.read_bytes() == b"abcdef"
    request, timeout = requests[0]
    assert isinstance(request, urllib.request.Request)
    assert request.full_url == "https://example.test/tree.bin"
    assert timeout == 7
    assert not (tmp_path / "tree.bin.part").exists()


def test_download_file_reports_progress(tmp_path) -> None:
    messages = []

    def fake_urlopen(_request, timeout):
        assert timeout == 30.0
        return FakeResponse([b"abc", b"def"])

    destination = tmp_path / "tree.bin"

    download_file(
        "https://example.test/tree.bin",
        destination,
        chunk_size=3,
        progress_interval_bytes=3,
        progress=messages.append,
        urlopen=fake_urlopen,
    )

    assert messages == [
        "opening: https://example.test/tree.bin",
        f"downloaded 3 bytes to {tmp_path / 'tree.bin.part'}",
        f"downloaded 6 bytes to {tmp_path / 'tree.bin.part'}",
        f"finished: {destination} (6 bytes)",
    ]


def test_download_vocab_trees_skips_existing_file(tmp_path) -> None:
    destination = tmp_path / "vocab_tree_faiss_flickr100K_words32K.bin"
    destination.write_bytes(b"existing")

    results = download_vocab_trees(
        variant="32k",
        destination_dir=tmp_path,
    )

    assert results == (
        DownloadResult(
            variant="32k",
            url=vocab_tree_url("32k"),
            path=destination,
            status="exists",
        ),
    )
    assert destination.read_bytes() == b"existing"


def test_download_vocab_trees_dry_run_does_not_create_files(tmp_path) -> None:
    messages = []

    results = download_vocab_trees(
        variant="1m",
        destination_dir=tmp_path,
        dry_run=True,
        progress=messages.append,
    )

    assert results[0].status == "planned"
    assert results[0].path.name == "vocab_tree_faiss_flickr100K_words1M.bin"
    assert not results[0].path.exists()
    assert messages == [
        (
            "planned download: "
            f"{vocab_tree_url('1m')} -> "
            f"{tmp_path / 'vocab_tree_faiss_flickr100K_words1M.bin'}"
        ),
    ]


def test_download_vocab_tree_cli_dry_run_prints_url(tmp_path, capsys) -> None:
    assert (
        main(
            [
                "--variant",
                "32k",
                "--destination-dir",
                str(tmp_path),
                "--dry-run",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert FAISS_FORMAT_NOTE in output
    assert "planned download:" in output
    assert "planned:" in output
    assert "vocab_tree_faiss_flickr100K_words32K.bin" in output


def test_download_vocab_tree_cli_legacy_format_prints_warning(tmp_path, capsys) -> None:
    assert (
        main(
            [
                "--variant",
                "32k",
                "--format",
                "legacy",
                "--destination-dir",
                str(tmp_path),
                "--dry-run",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert LEGACY_FORMAT_WARNING in output
    assert "vocab_tree_flickr100K_words32K.bin" in output

import importlib
import os
from pathlib import Path

import pytest

pytest.importorskip("cv2")
pytest.importorskip("pycolmap")

legacy_pano_reconstruction = importlib.import_module(
    "colmap.legacy_pano_reconstruction"
)


def test_configure_ssl_cert_file_uses_explicit_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cert_file = tmp_path / "ca-certificates.crt"
    cert_file.write_text("test bundle", encoding="utf-8")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)

    legacy_pano_reconstruction.configure_ssl_cert_file(cert_file)

    assert os.environ["SSL_CERT_FILE"] == str(cert_file)


def test_configure_ssl_cert_file_respects_existing_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SSL_CERT_FILE", "/custom/ca.pem")

    legacy_pano_reconstruction.configure_ssl_cert_file()

    assert os.environ["SSL_CERT_FILE"] == "/custom/ca.pem"


def test_create_vocab_tree_pairing_options_sets_local_tree(tmp_path: Path) -> None:
    vocab_tree = tmp_path / "vocab_tree.bin"
    vocab_tree.write_bytes(b"tree")

    options = legacy_pano_reconstruction.create_vocab_tree_pairing_options(
        vocab_tree
    )

    assert Path(options.vocab_tree_path) == vocab_tree


def test_create_vocab_tree_pairing_options_rejects_missing_tree(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="Vocabulary tree not found"):
        legacy_pano_reconstruction.create_vocab_tree_pairing_options(
            tmp_path / "missing.bin"
        )

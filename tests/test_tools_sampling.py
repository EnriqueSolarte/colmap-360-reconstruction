import pytest

from tools.sampling_dir import main as sampling_main


def test_sampling_cli_starts_from_initial_index(tmp_path, capsys) -> None:
    source_dir = tmp_path / "source"
    dest_dir = tmp_path / "dest"
    source_dir.mkdir()

    for name in ("000.jpg", "001.jpg", "002.jpg", "003.jpg", "004.jpg"):
        (source_dir / name).write_text(name)

    exit_code = sampling_main(
        [
            "--source",
            str(source_dir),
            "--destination",
            str(dest_dir),
            "--skip",
            "1",
            "--idx",
            "1",
        ]
    )

    assert exit_code == 0
    assert sorted(path.name for path in dest_dir.iterdir()) == ["001.jpg", "003.jpg"]
    assert not (dest_dir / "001.jpg").is_symlink()
    assert (dest_dir / "001.jpg").stat().st_dev == (
        source_dir / "001.jpg"
    ).stat().st_dev
    assert (dest_dir / "001.jpg").stat().st_ino == (
        source_dir / "001.jpg"
    ).stat().st_ino

    output = capsys.readouterr().out
    assert "idx=1" in output
    assert "hard link" in output


def test_sampling_cli_can_create_symlinks(tmp_path) -> None:
    source_dir = tmp_path / "source"
    dest_dir = tmp_path / "dest"
    source_dir.mkdir()
    (source_dir / "000.jpg").write_text("000")

    exit_code = sampling_main(
        [
            "--source",
            str(source_dir),
            "--destination",
            str(dest_dir),
            "--symlink",
        ]
    )

    assert exit_code == 0
    assert (dest_dir / "000.jpg").is_symlink()
    assert (dest_dir / "000.jpg").resolve() == (source_dir / "000.jpg").resolve()


def test_sampling_cli_does_not_fallback_on_link_error(tmp_path) -> None:
    source_dir = tmp_path / "source"
    dest_dir = tmp_path / "dest"
    source_dir.mkdir()
    dest_dir.mkdir()
    (source_dir / "000.jpg").write_text("source")
    (dest_dir / "000.jpg").write_text("existing")

    with pytest.raises(FileExistsError):
        sampling_main(
            [
                "--source",
                str(source_dir),
                "--destination",
                str(dest_dir),
            ]
        )

    assert (dest_dir / "000.jpg").read_text() == "existing"


def test_sampling_cli_rejects_negative_initial_index(tmp_path) -> None:
    source_dir = tmp_path / "source"
    dest_dir = tmp_path / "dest"
    source_dir.mkdir()
    (source_dir / "000.jpg").write_text("000")

    with pytest.raises(SystemExit, match="--idx must be >= 0"):
        sampling_main(
            [
                "--source",
                str(source_dir),
                "--destination",
                str(dest_dir),
                "--idx",
                "-1",
            ]
        )

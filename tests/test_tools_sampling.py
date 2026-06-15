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
    assert (dest_dir / "001.jpg").resolve() == (source_dir / "001.jpg").resolve()

    output = capsys.readouterr().out
    assert "idx=1" in output


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

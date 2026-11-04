"""Archive fixtures contain artificial bytes, not real credit records."""

import json
from zipfile import ZipFile

import pytest

from credit_risk.prepare import fetch_data


def archive(tmp_path, member="default of credit card clients.xls"):
    path = tmp_path / "fixture.zip"
    with ZipFile(path, "w") as z:
        z.writestr(member, b"artificial source bytes for integrity testing")
    return path.as_uri()


def test_download_caches_source_and_detects_later_xls_tampering(tmp_path):
    url = archive(tmp_path)
    raw = tmp_path / "raw"
    path = fetch_data(raw, url)
    assert path.read_bytes() == b"artificial source bytes for integrity testing"
    metadata = json.loads((raw / "source.json").read_text())
    assert metadata["source_url"] == url
    assert len(metadata["xls_sha256"]) == 64
    assert fetch_data(raw, url) == path
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        fetch_data(raw, url)


def test_unexpected_archive_content_is_not_extracted(tmp_path):
    url = archive(tmp_path, "../unexpected.xls")
    with pytest.raises(ValueError, match="archive"):
        fetch_data(tmp_path / "raw", url)
    assert not (tmp_path / "unexpected.xls").exists()


def test_invalid_download_fails_without_metadata(tmp_path):
    path = tmp_path / "bad.zip"
    path.write_bytes(b"this is not a zip archive")
    with pytest.raises(ValueError, match="archive"):
        fetch_data(tmp_path / "raw", path.as_uri())
    assert not (tmp_path / "raw" / "source.json").exists()


def test_cached_archive_tampering_is_detected(tmp_path):
    url = archive(tmp_path)
    raw = tmp_path / "raw"
    fetch_data(raw, url)
    (raw / "source.zip").write_bytes(b"modified archive")
    with pytest.raises(ValueError, match="checksum"):
        fetch_data(raw, url)

"""Tests for TLS self-signed cert generation, SAN matching, key matching, and expiry renewal."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from tabgpu.tls import (
    TLSError,
    create_ssl_context,
    ensure_tls_cert,
    generate_self_signed_cert,
    validate_cert_and_key,
)


def test_cert_generation_and_reuse(tmp_path: Path) -> None:
    cert_dir = tmp_path / "certs"
    first = ensure_tls_cert(cert_dir, ["192.168.1.23", "myhost.local"])
    assert first.regenerated is True
    assert first.cert_path.is_file()
    assert first.key_path.is_file()
    assert "192.168.1.23" in first.san_hosts
    assert "myhost.local" in first.san_hosts

    # SSLContext loads generated cert/key cleanly
    ctx = create_ssl_context(first.cert_path, first.key_path)
    assert ctx is not None

    # Reusing with matching SANs does not regenerate
    second = ensure_tls_cert(cert_dir, ["192.168.1.23"])
    assert second.regenerated is False
    assert second.fingerprint_sha256 == first.fingerprint_sha256


def test_cert_regenerates_on_san_mismatch(tmp_path: Path) -> None:
    cert_dir = tmp_path / "certs"
    first = ensure_tls_cert(cert_dir, ["192.168.1.23"])
    second = ensure_tls_cert(cert_dir, ["10.0.0.55"])
    assert second.regenerated is True
    assert second.fingerprint_sha256 != first.fingerprint_sha256
    assert "10.0.0.55" in second.san_hosts


def test_cert_regenerates_when_expiring_or_expired(tmp_path: Path) -> None:
    cert_dir = tmp_path / "certs"
    base_now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    first = ensure_tls_cert(
        cert_dir,
        ["127.0.0.1"],
        valid_days=30,
        min_remaining_days=7,
        now=base_now,
    )
    assert first.regenerated is True

    # 25 days later -> only 5 days remaining (< 7) -> must regenerate
    later = base_now + timedelta(days=25)
    ok, reason = validate_cert_and_key(
        first.cert_path,
        first.key_path,
        expected_hosts=["127.0.0.1"],
        min_remaining_days=7,
        now=later,
    )
    assert ok is False
    assert "expires" in reason

    renewed = ensure_tls_cert(
        cert_dir,
        ["127.0.0.1"],
        valid_days=30,
        min_remaining_days=7,
        now=later,
    )
    assert renewed.regenerated is True
    assert renewed.fingerprint_sha256 != first.fingerprint_sha256


def test_cert_regenerates_on_key_mismatch(tmp_path: Path) -> None:
    cert_dir = tmp_path / "certs"
    first = ensure_tls_cert(cert_dir, ["127.0.0.1"])

    # Replace private key with a different unrelated key
    other_key = ec.generate_private_key(ec.SECP256R1())
    first.key_path.write_bytes(
        other_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    ok, reason = validate_cert_and_key(first.cert_path, first.key_path, ["127.0.0.1"])
    assert ok is False
    assert "match" in reason

    repaired = ensure_tls_cert(cert_dir, ["127.0.0.1"])
    assert repaired.regenerated is True


def test_user_supplied_cert_and_key_validation(tmp_path: Path) -> None:
    user_cert = tmp_path / "user.pem"
    user_key = tmp_path / "user-key.pem"
    generate_self_signed_cert(user_cert, user_key, ["127.0.0.1"])

    mat = ensure_tls_cert(
        tmp_path / "cache",
        ["127.0.0.1"],
        cert_path=user_cert,
        key_path=user_key,
    )
    assert mat.regenerated is False
    assert mat.cert_path == user_cert

    # Missing key pair rejected
    with pytest.raises(TLSError):
        ensure_tls_cert(tmp_path / "cache", ["127.0.0.1"], cert_path=user_cert, key_path=None)

    # Mismatched user key rejected (and never overwritten)
    bad_key = tmp_path / "other.key"
    other = ec.generate_private_key(ec.SECP256R1())
    bad_key.write_bytes(
        other.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    with pytest.raises(TLSError):
        ensure_tls_cert(
            tmp_path / "cache",
            ["127.0.0.1"],
            cert_path=user_cert,
            key_path=bad_key,
        )
    # Verify cert was still intact
    assert x509.load_pem_x509_certificate(user_cert.read_bytes()) is not None

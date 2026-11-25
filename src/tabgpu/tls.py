"""Local HTTPS certificate generation, SAN/expiry validation, and SSL context creation."""

from __future__ import annotations

import ipaddress
import os
import secrets
import ssl
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


class TLSError(RuntimeError):
    """Raised when TLS certificate or key validation fails."""


@dataclass(frozen=True)
class TLSMaterial:
    cert_path: Path
    key_path: Path
    fingerprint_sha256: str
    not_valid_after: datetime
    san_hosts: tuple[str, ...]
    regenerated: bool


def _normalize_hosts(hosts: Iterable[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for raw in hosts:
        h = raw.strip()
        if not h or h == "0.0.0.0" or h == "::":
            continue
        try:
            normalized = str(ipaddress.ip_address(h))
        except ValueError:
            normalized = h.lower().rstrip(".")
        if normalized not in seen:
            seen.append(normalized)
    if not seen:
        seen.append("127.0.0.1")
    return tuple(seen)


def _build_san_entries(hosts: Iterable[str]) -> list[x509.GeneralName]:
    entries: list[x509.GeneralName] = []
    for host in _normalize_hosts(hosts):
        try:
            ip_obj = ipaddress.ip_address(host)
        except ValueError:
            entries.append(x509.DNSName(host))
        else:
            entries.append(x509.IPAddress(ip_obj))
    return entries


def _extract_san_hosts(cert: x509.Certificate) -> set[str]:
    try:
        ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
    except x509.ExtensionNotFound:
        return set()
    found: set[str] = set()
    for ip_val in ext.value.get_values_for_type(x509.IPAddress):
        found.add(str(ip_val))
    for dns_val in ext.value.get_values_for_type(x509.DNSName):
        found.add(dns_val.lower().rstrip("."))
    return found


def cert_fingerprint_sha256(cert: x509.Certificate) -> str:
    raw_hex = cert.fingerprint(hashes.SHA256()).hex().upper()
    return ":".join(raw_hex[i : i + 2] for i in range(0, len(raw_hex), 2))


def validate_cert_and_key(
    cert_path: Path,
    key_path: Path,
    expected_hosts: Iterable[str] | None = None,
    *,
    min_remaining_days: int = 7,
    now: datetime | None = None,
) -> tuple[bool, str]:
    """Validate cert/key existence, key match, validity window, and SAN coverage."""
    if not cert_path.is_file() or not key_path.is_file():
        return False, "certificate or key file missing"

    try:
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
    except Exception as exc:
        return False, f"failed to parse certificate or key: {exc}"

    cert_pub = cert.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    key_pub = key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    if cert_pub != key_pub:
        return False, "private key does not match certificate public key"

    current = now or datetime.now(timezone.utc)
    if current < cert.not_valid_before_utc:
        return False, "certificate is not yet valid"
    if current >= cert.not_valid_after_utc:
        return False, "certificate is expired"
    if cert.not_valid_after_utc - current < timedelta(days=min_remaining_days):
        return False, f"certificate expires in fewer than {min_remaining_days} days"

    if expected_hosts is not None:
        required = set(_normalize_hosts(expected_hosts))
        present = _extract_san_hosts(cert)
        missing = required - present
        if missing:
            return False, f"certificate SAN missing required hosts: {sorted(missing)}"

    return True, "ok"


def _atomic_write_pem(path: Path, data: bytes, *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp")
    try:
        with tmp.open("wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if private and os.name != "nt":
            os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def generate_self_signed_cert(
    cert_path: Path,
    key_path: Path,
    hosts: Iterable[str],
    *,
    valid_days: int = 30,
    now: datetime | None = None,
) -> TLSMaterial:
    """Generate a self-signed ECDSA P-256 TLS server cert with exact IP/DNS SANs."""
    normalized = _normalize_hosts(hosts)
    san_entries = _build_san_entries(normalized)
    current = now or datetime.now(timezone.utc)

    private_key = ec.generate_private_key(ec.SECP256R1())
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "tabgpu local LAN"),
            x509.NameAttribute(NameOID.COMMON_NAME, normalized[0]),
        ]
    )

    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(current - timedelta(minutes=5))
        .not_valid_after(current + timedelta(days=valid_days))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(private_key.public_key()),
            critical=False,
        )
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
    )

    cert = builder.sign(private_key=private_key, algorithm=hashes.SHA256())

    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    _atomic_write_pem(key_path, key_pem, private=True)
    _atomic_write_pem(cert_path, cert_pem, private=False)

    return TLSMaterial(
        cert_path=cert_path,
        key_path=key_path,
        fingerprint_sha256=cert_fingerprint_sha256(cert),
        not_valid_after=cert.not_valid_after_utc,
        san_hosts=normalized,
        regenerated=True,
    )


def ensure_tls_cert(
    cert_dir: Path,
    hosts: Iterable[str],
    *,
    cert_path: Path | None = None,
    key_path: Path | None = None,
    valid_days: int = 30,
    min_remaining_days: int = 7,
    now: datetime | None = None,
) -> TLSMaterial:
    """Validate user-supplied cert/key or auto-generate/reuse a SAN-matched local cert."""
    normalized = _normalize_hosts(hosts)

    if cert_path is not None or key_path is not None:
        if cert_path is None or key_path is None:
            raise TLSError("Both --cert and --key must be provided together.")
        ok, reason = validate_cert_and_key(
            cert_path,
            key_path,
            expected_hosts=None,
            min_remaining_days=0,
            now=now,
        )
        if not ok:
            raise TLSError(f"Invalid user-supplied TLS certificate/key: {reason}")
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        return TLSMaterial(
            cert_path=cert_path,
            key_path=key_path,
            fingerprint_sha256=cert_fingerprint_sha256(cert),
            not_valid_after=cert.not_valid_after_utc,
            san_hosts=tuple(sorted(_extract_san_hosts(cert))),
            regenerated=False,
        )

    managed_cert = cert_dir / "server.pem"
    managed_key = cert_dir / "server-key.pem"
    ok, _reason = validate_cert_and_key(
        managed_cert,
        managed_key,
        expected_hosts=normalized,
        min_remaining_days=min_remaining_days,
        now=now,
    )
    if ok:
        cert = x509.load_pem_x509_certificate(managed_cert.read_bytes())
        return TLSMaterial(
            cert_path=managed_cert,
            key_path=managed_key,
            fingerprint_sha256=cert_fingerprint_sha256(cert),
            not_valid_after=cert.not_valid_after_utc,
            san_hosts=tuple(sorted(_extract_san_hosts(cert))),
            regenerated=False,
        )

    return generate_self_signed_cert(
        managed_cert,
        managed_key,
        normalized,
        valid_days=valid_days,
        now=now,
    )


def create_ssl_context(cert_path: Path, key_path: Path) -> ssl.SSLContext:
    """Create a hardened TLS 1.2+ server SSLContext."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
    return ctx

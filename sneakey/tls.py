"""A self-signed HTTPS certificate for the phone page, made locally with the openssl command-line tool.

Phones only allow microphone access on secure pages (HTTPS or localhost), and the phone reaches the laptop over
the local network, so the phone server needs HTTPS. The certificate is self-signed: the phone shows a warning
once, which you accept. It is regenerated whenever the laptop's network addresses change.
"""
import json
import shutil
import socket
import subprocess

from . import config

TLS_DIR = config.DATA_DIR / "tls"


def lan_addresses():
    """The laptop's IPv4 address on the network with the default route (the one a phone on the same Wi-Fi
    reaches). Connecting a UDP socket only picks a route; no packet is sent."""
    addresses = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            addresses.append(s.getsockname()[0])
    except OSError:
        pass
    if addresses and not addresses[0].startswith("127."):
        return addresses          # the default-route address; others are often virtual (Docker, VPN) and unreachable
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in addresses and not ip.startswith("127."):
                addresses.append(ip)
    except OSError:
        pass
    return [ip for ip in addresses if not ip.startswith("127.")]


def ensure_certificate(addresses):
    """Returns (cert path, key path), creating or refreshing them for these addresses."""
    if not shutil.which("openssl"):
        raise RuntimeError("The phone server needs the `openssl` command to create its HTTPS certificate.")
    TLS_DIR.mkdir(parents=True, exist_ok=True)
    cert, key, info = TLS_DIR / "cert.pem", TLS_DIR / "key.pem", TLS_DIR / "addresses.json"
    wanted = sorted(set(addresses) | {"127.0.0.1"})
    if cert.exists() and key.exists() and info.exists() and json.loads(info.read_text()) == wanted:
        return cert, key
    san = ",".join([f"IP:{ip}" for ip in wanted] + ["DNS:localhost"])
    subprocess.run(["openssl", "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1",
                    "-nodes", "-days", "30", "-subj", "/CN=SneaKey phone (local)", "-addext", f"subjectAltName={san}",
                    "-keyout", str(key), "-out", str(cert)], check=True, capture_output=True)
    key.chmod(0o600)
    info.write_text(json.dumps(wanted))
    return cert, key

"""Start SneaKey:  python run.py   then open http://127.0.0.1:8000

The app listens on localhost only: recordings and processing never leave this machine.

    python run.py --phone     also serve the phone recording page over HTTPS on the local network
                              (https://<this laptop's address>:8443/p/<code>), so a phone on the same Wi-Fi can be
                              the microphone. That port serves ONLY the phone page and its endpoints, behind a
                              random pairing code; the rest of the app stays on localhost.
"""
import argparse
import asyncio
import os

import uvicorn

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--phone", action="store_true", help="also serve the phone recording page on the local network")
parser.add_argument("--phone-port", type=int, default=8443)
args = parser.parse_args()

servers = []
if args.phone:
    os.environ["SNEAKEY_PHONE_PORT"] = str(args.phone_port)
    from sneakey import tls
    addresses = tls.lan_addresses()
    if not addresses:
        raise SystemExit("No network address found: connect the laptop to the same Wi-Fi (or hotspot) as the phone.")
    from sneakey import config
    config.PHONE_ADDRESSES = addresses          # config was already imported (by tls), so set it directly
    cert, key = tls.ensure_certificate(addresses)

from sneakey.api import app  # noqa: E402  (after the environment above is set)

servers.append(uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=args.port)))
print(f"SneaKey on http://127.0.0.1:{args.port}")
if args.phone:
    servers.append(uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=args.phone_port,
                                                 ssl_certfile=str(cert), ssl_keyfile=str(key))))
    print(f"Phone page on https://{addresses[0]}:{args.phone_port}/p/<code>  (the Record page shows the full link)")


async def main():
    await asyncio.gather(*(server.serve() for server in servers))

asyncio.run(main())

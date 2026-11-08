"""Start SneaKey:  python run.py   then open http://127.0.0.1:8000

Binds to localhost only by default; recordings and processing never leave this machine."""
import argparse

import uvicorn

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--host", default="127.0.0.1", help="keep the default unless you know why you need otherwise")
args = parser.parse_args()
print(f"SneaKey on http://{args.host}:{args.port}")
uvicorn.run("sneakey.api:app", host=args.host, port=args.port)

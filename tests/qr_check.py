"""
Verify the pure-JS QR encoder (web/js/qr.js): render a range of strings to PNGs via Node, then decode them with
zbarimg and/or OpenCV and check they round-trip. Needs node and (zbarimg or python cv2).

    python tests/qr_check.py
"""
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NODE = r'''
import { qrMatrix } from "%s";
const m = qrMatrix(process.argv[process.argv.length - 1]);
process.stdout.write(JSON.stringify(m));
''' % (ROOT / "web/js/qr.js").as_uri()


def render(matrix, path, quiet=4, scale=10):
    import numpy as np
    import cv2
    n = len(matrix)
    img = np.ones(((n + 2 * quiet) * scale, (n + 2 * quiet) * scale), np.uint8) * 255
    for r in range(n):
        for c in range(n):
            if matrix[r][c]:
                img[(r + quiet) * scale:(r + quiet + 1) * scale, (c + quiet) * scale:(c + quiet + 1) * scale] = 0
    cv2.imwrite(str(path), img)
    return img.shape[0]


def decode(path):
    results = set()
    try:
        out = subprocess.run(["zbarimg", "-q", "--raw", str(path)], capture_output=True, text=True, timeout=20)
        if out.returncode == 0 and out.stdout.strip():
            results.add(out.stdout.strip())
    except FileNotFoundError:
        pass
    try:
        import cv2
        text, _, _ = cv2.QRCodeDetector().detectAndDecode(cv2.imread(str(path)))
        if text:
            results.add(text)
    except ImportError:
        pass
    return results


def have_decoder():
    import shutil
    if shutil.which("zbarimg"):
        return True
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:
        return False


def main():
    cases = [
        "https://10.126.29.211:8443/p/Y2CVCJCS",
        "https://192.168.1.237:8443/p/ABCD2345",
        "https://10.0.0.5:8443/p/" + "W" * 8,
        "HELLO",
        "https://255.255.255.255:65535/p/" + "Z" * 8,                       # longest realistic pairing URL
        "https://10.0.0.1:8443/p/ABCD2345&note=" + "x" * 40,              # version 5-6
    ]
    if not have_decoder():
        print("No decoder available (install zbar or python-opencv) - cannot verify.")
        sys.exit(0)
    failures = 0
    for text in cases:
        result = subprocess.run(["node", "--input-type=module", "-e", NODE, "--", text], capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"FAIL  encoder errored for {text!r}: {result.stderr.strip()[:200]}")
            failures += 1
            continue
        import json
        matrix = json.loads(result.stdout)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = Path(f.name)
        dim = render(matrix, path)
        decoded = decode(path)
        path.unlink()
        ok = text in decoded
        print(("PASS  " if ok else "FAIL  ") + f"{len(text):3} bytes, {len(matrix)} modules, {dim}px -> {'scanned' if ok else f'decoded {decoded or 'nothing'}'}  {text[:48]}")
        failures += not ok
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

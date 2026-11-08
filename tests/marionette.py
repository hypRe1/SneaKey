"""Minimal Marionette client (no geckodriver/selenium needed) for driving headless Firefox in e2e tests."""
import json, os, shutil, socket, subprocess, tempfile, time

class Firefox:
    def __init__(self, port=2828, extra_prefs=None, capabilities=None):
        self.profile = tempfile.mkdtemp(prefix="sneakey-ff-")
        prefs = {
            "media.navigator.streams.fake": True,
            "media.navigator.permission.disabled": True,
            "media.getusermedia.audio.capture.enabled": True,
            "marionette.port": port,
            "datareporting.policy.dataSubmissionEnabled": False,
            "browser.shell.checkDefaultBrowser": False,
        }
        prefs.update(extra_prefs or {})
        with open(os.path.join(self.profile, "user.js"), "w") as f:
            for k, v in prefs.items():
                f.write(f"user_pref({json.dumps(k)}, {json.dumps(v)});\n")
        env = dict(os.environ, MOZ_HEADLESS="1")
        self.proc = subprocess.Popen(
            ["firefox", "--headless", "--marionette", "--no-remote", "--profile", self.profile],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.sock = None
        for _ in range(100):
            try:
                self.sock = socket.create_connection(("127.0.0.1", port), timeout=120); break
            except OSError:
                time.sleep(0.2)
        if not self.sock: raise RuntimeError("could not connect to Marionette")
        self.buf = b""; self.id = 0
        self._read()  # hello
        self.cmd("WebDriver:NewSession", {"capabilities": capabilities or {}})
        self.cmd("WebDriver:SetTimeouts", {"script": 600000})

    def _read(self):
        while b":" not in self.buf: self.buf += self.sock.recv(65536)
        n, rest = self.buf.split(b":", 1)
        n = int(n)
        while len(rest) < n: rest += self.sock.recv(1 << 20)
        msg, self.buf = rest[:n], rest[n:]
        return json.loads(msg)

    def cmd(self, name, params=None):
        self.id += 1
        body = json.dumps([0, self.id, name, params or {}]).encode()
        self.sock.sendall(str(len(body)).encode() + b":" + body)
        while True:
            r = self._read()
            if r[0] == 1 and r[1] == self.id:
                if r[2]: raise RuntimeError(f"{name}: {r[2]}")
                return r[3]

    def get(self, url): return self.cmd("WebDriver:Navigate", {"url": url})

    def run(self, script, *args):
        """Execute an async script; the last argument is the completion callback."""
        r = self.cmd("WebDriver:ExecuteAsyncScript", {"script": script, "args": list(args), "scriptTimeout": 120000} if False else {"script": script, "args": list(args)})
        return r.get("value") if isinstance(r, dict) else r

    def close(self):
        try: self.cmd("Marionette:Quit")
        except Exception: pass
        try: self.proc.wait(10)
        except Exception: self.proc.kill()
        shutil.rmtree(self.profile, ignore_errors=True)

"""
Local web app: choose a video in the page -> it is translated on THIS computer.

  python app.py
  then open the link it prints (it contains a secret token).

SECURITY, in plain words:
  * The server listens on 127.0.0.1 only. Other computers cannot reach it.
  * Every request needs a secret token (kept in a cookie). Other websites open in
    your browser cannot use the server.
  * The Host and Origin headers are checked (blocks "DNS rebinding" tricks).
  * The video is saved to a private temporary folder and deleted as soon as it
    has been listened to. Nothing else is written to disk; you download the
    subtitles yourself with a button.
  * pipeline.py blocks every network connection that is not to this computer.
"""
import pipeline   # FIRST: switches on the offline / local-only locks

import argparse
import atexit
import hmac
import json
import re
import secrets
import shutil
import signal
import socketserver
import sys
import tempfile
import threading
import traceback
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

BASE = Path(__file__).resolve().parent
ALLOWED_EXT = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi",
               ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
ACTIVE = {"uploading", "transcribing", "translating"}

# ---------------------------------------------------------------------------
# The one job that can run at a time (this is a personal tool)
# ---------------------------------------------------------------------------
JOB_LOCK = threading.Lock()
CANCEL = threading.Event()
JOB = {}


def reset_job():
    JOB.clear()
    JOB.update(state="idle", fraction=0.0, message="", warning="", error="",
               detected="", detected_name="", confidence=0.0,
               target="", target_name="", segments=[])


reset_job()


def update_job(**kw):
    with JOB_LOCK:
        JOB.update(kw)


def run_job(path: Path, source, target, whisper_name, translator_dir):
    def progress(stage, frac, msg):
        if stage == "transcribing":
            update_job(state=stage, fraction=0.7 * frac, message=f"Step 1 of 2 - {msg}")
        else:
            update_job(state=stage, fraction=0.7 + 0.3 * frac, message=f"Step 2 of 2 - {msg}")

    try:
        whisper_dir = pipeline.MODELS_DIR / f"whisper-{whisper_name}"
        segments, lang, conf = pipeline.transcribe(path, whisper_dir, source, progress, CANCEL)
        path.unlink(missing_ok=True)          # the video is not needed any more
        lang_name = pipeline.WHISPER_NAMES.get(lang, lang)
        warning = ""
        if source is None and conf < 0.5:
            warning = (f"Language detection was unsure ({conf:.0%}). If the result looks wrong, "
                       f"choose the spoken language yourself and run again.")
        update_job(detected=lang, detected_name=lang_name, confidence=conf, warning=warning)

        pipeline.translate(segments, lang, target, translator_dir, progress, CANCEL)
        update_job(state="done", fraction=1.0, segments=segments,
                   message=f"Done: {lang_name} -> {pipeline.NLLB_NAMES.get(target, target)}")
    except pipeline.Cancelled:
        update_job(state="cancelled", message="Cancelled.")
    except Exception as e:
        traceback.print_exc()
        update_job(state="error", error=f"{type(e).__name__}: {e}")
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# The server
# ---------------------------------------------------------------------------
class LocalServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def server_bind(self):
        # The normal version does a name lookup on the network; we do not need that.
        socketserver.TCPServer.server_bind(self)
        self.server_name = "127.0.0.1"
        self.server_port = self.server_address[1]


class Handler(BaseHTTPRequestHandler):
    server_version = "LocalTranslator"
    sys_version = ""
    timeout = 120

    def log_message(self, fmt, *args):      # keep the terminal quiet
        pass

    # ---- small helpers -----------------------------------------------------
    def _send(self, status, body=b"", ctype="text/plain; charset=utf-8", headers=None, nonce=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if nonce:
            # Only this page's own script and style (marked with the nonce) may run. No eval,
            # no inline handlers, no outside files. The page may only talk to this server.
            csp = ("default-src 'none'; script-src 'nonce-%s'; style-src 'nonce-%s'; "
                   "img-src data: blob:; media-src blob:; "
                   "connect-src 'self'; form-action 'none'; base-uri 'none'; frame-ancestors 'none'"
                   % (nonce, nonce))
        else:
            csp = "default-src 'none'; frame-ancestors 'none'"
        self.send_header("Content-Security-Policy", csp)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status, obj):
        self._send(status, json.dumps(obj), "application/json; charset=utf-8")

    def _host_ok(self):
        return self.headers.get("Host", "") in self.server.allowed_hosts

    def _origin_ok(self):
        origin = self.headers.get("Origin")
        return origin is None or origin in self.server.allowed_origins

    def _cookie_ok(self):
        try:
            c = SimpleCookie(self.headers.get("Cookie", ""))
        except CookieError:
            return False
        m = c.get("sid")
        return bool(m) and hmac.compare_digest(m.value, self.server.token)

    # ---- GET ---------------------------------------------------------------
    def do_GET(self):
        if not self._host_ok():
            return self._send(403, "Forbidden (bad Host).")
        url = urlparse(self.path)
        q = parse_qs(url.query)

        # First visit: the link from the terminal carries the token -> store it in a cookie
        if url.path == "/" and "token" in q:
            if hmac.compare_digest(q["token"][0], self.server.token):
                cookie = f"sid={self.server.token}; HttpOnly; SameSite=Strict; Path=/"
                return self._send(302, "", headers={"Location": "/", "Set-Cookie": cookie})
            return self._send(403, "Wrong token.")

        if not self._cookie_ok():
            return self._send(403, "Open the link printed in the terminal where you started app.py.")

        if url.path == "/":
            nonce = secrets.token_urlsafe(16)
            html = (BASE / "player.html").read_text(encoding="utf-8").replace("__NONCE__", nonce)
            return self._send(200, html, "text/html; charset=utf-8", nonce=nonce)

        if url.path == "/api/config":
            langs = [{"whisper": w, "nllb": n, "name": name} for w, n, name in pipeline.LANGUAGES]
            return self._json(200, {
                "languages": sorted(langs, key=lambda x: x["name"]),
                "whisper": pipeline.find_whisper_models(),
                "translator": bool(pipeline.find_translator()),
                "max_gb": self.server.max_bytes / 1024 ** 3,
            })

        if url.path == "/api/status":
            with JOB_LOCK:
                keys = ("state", "fraction", "message", "warning", "error",
                        "detected_name", "confidence", "target_name")
                return self._json(200, {k: JOB[k] for k in keys})

        if url.path == "/api/result":
            with JOB_LOCK:
                if JOB["state"] != "done":
                    return self._json(409, {"error": "No finished result."})
                return self._json(200, {"segments": JOB["segments"],
                                        "detected_name": JOB["detected_name"],
                                        "target_name": JOB["target_name"]})

        if url.path == "/api/srt":
            kind = q.get("kind", ["text"])[0]
            with JOB_LOCK:
                if JOB["state"] != "done" or kind not in ("text", "src", "both"):
                    return self._json(409, {"error": "No finished result."})
                srt = pipeline.to_srt(JOB["segments"], kind)
                tag = {"text": pipeline.short_code(JOB["target"]),
                       "src": JOB["detected"], "both": f"{JOB['detected']}-{pipeline.short_code(JOB['target'])}"}[kind]
            return self._send(200, srt, "application/x-subrip; charset=utf-8",
                              headers={"Content-Disposition": f'attachment; filename="subtitles.{tag}.srt"'})

        return self._send(404, "Not found.")

    # ---- POST --------------------------------------------------------------
    def do_POST(self):
        if not (self._host_ok() and self._origin_ok() and self._cookie_ok()):
            return self._send(403, "Forbidden.")
        url = urlparse(self.path)

        if url.path == "/api/cancel":
            CANCEL.set()
            return self._json(200, {"ok": True})

        if url.path == "/api/start":
            return self._start(parse_qs(url.query))

        return self._send(404, "Not found.")

    def _start(self, q):
        def refuse(code, msg):
            self.close_connection = True
            self._json(code, {"error": msg})

        # --- check the settings BEFORE accepting the (big) upload
        try:
            target = pipeline.resolve_language(q.get("target", [""])[0])
            src_arg = q.get("source", ["auto"])[0]
            source = None if src_arg in ("", "auto") else src_arg
            if source is not None and source not in pipeline.WHISPER_NAMES:
                raise pipeline.UnsupportedLanguage(f"Unknown spoken language: {source}")
        except pipeline.UnsupportedLanguage as e:
            return refuse(400, str(e))

        whisper_name = q.get("whisper", [""])[0]
        if whisper_name not in pipeline.find_whisper_models():
            return refuse(400, "That speech model is not downloaded. Run download_models.py once.")
        translator_dir = pipeline.find_translator()
        if translator_dir is None:
            return refuse(400, "The translation model is not downloaded. Run download_models.py once.")

        ext = Path(q.get("name", [""])[0]).suffix.lower()
        if ext not in ALLOWED_EXT:
            return refuse(400, "This file type is not supported.")

        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            return refuse(411, "Content-Length is required.")
        if length <= 0:
            return refuse(400, "The file is empty.")
        if length > self.server.max_bytes:
            return refuse(413, f"File is larger than the {self.server.max_bytes / 1024 ** 3:.0f} GB limit.")
        if shutil.disk_usage(self.server.tmpdir).free < length * 1.1:
            return refuse(507, "Not enough free disk space for a temporary copy.")

        with JOB_LOCK:
            if JOB["state"] in ACTIVE:
                return refuse(409, "Another file is still being processed.")
            CANCEL.clear()
            JOB.update(state="uploading", fraction=0.0, message="Receiving the file ...",
                       warning="", error="", segments=[], detected="", detected_name="",
                       confidence=0.0, target=target,
                       target_name=pipeline.NLLB_NAMES.get(target, target))

        # --- receive the file into the private temporary folder
        path = Path(self.server.tmpdir) / f"upload{ext}"
        try:
            remaining = length
            with open(path, "wb") as f:
                while remaining:
                    if CANCEL.is_set():
                        raise pipeline.Cancelled()
                    chunk = self.rfile.read(min(1 << 20, remaining))
                    if not chunk:
                        raise ConnectionError("Upload was interrupted.")
                    f.write(chunk)
                    remaining -= len(chunk)
        except pipeline.Cancelled:
            path.unlink(missing_ok=True)
            self.close_connection = True
            update_job(state="cancelled", message="Cancelled.")
            return
        except Exception as e:
            path.unlink(missing_ok=True)
            self.close_connection = True
            update_job(state="error", error=f"Upload failed: {e}")
            return

        update_job(state="transcribing", message="Step 1 of 2 - starting ...")
        threading.Thread(target=run_job, args=(path, source, target, whisper_name, translator_dir),
                         daemon=True).start()
        self._send(202, json.dumps({"ok": True}), "application/json; charset=utf-8")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Private local video translator (web page).")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-gb", type=float, default=8.0, help="Largest file accepted (default 8 GB)")
    args = ap.parse_args()

    if not pipeline.find_whisper_models() or pipeline.find_translator() is None:
        print("Models not found in ./models . Run 'python download_models.py' once (with internet).")
        sys.exit(1)

    tmpdir = tempfile.mkdtemp(prefix="local-translator-")      # private (owner only)
    atexit.register(shutil.rmtree, tmpdir, ignore_errors=True)
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))

    token = secrets.token_urlsafe(32)
    server = LocalServer(("127.0.0.1", args.port), Handler)    # 127.0.0.1 = this computer only
    port = server.server_address[1]
    server.token = token
    server.tmpdir = tmpdir
    server.max_bytes = int(args.max_gb * 1024 ** 3)
    server.allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    server.allowed_origins = {f"http://{h}" for h in server.allowed_hosts}

    print("\nLocal video translator is running (this computer only).")
    print(f"Open this link in your browser:\n\n  http://127.0.0.1:{port}/?token={token}\n")
    print("Keep this window open. Press Ctrl+C to stop; temporary files are deleted.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        CANCEL.set()
        server.server_close()


if __name__ == "__main__":
    main()

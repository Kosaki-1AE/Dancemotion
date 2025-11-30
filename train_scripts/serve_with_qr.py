# serve_with_qr.py — local HTTP + (optional) Cloudflare Tunnel QR (+ v-JEPA /analyze)
import argparse
import http.server
import json
import os
import re
import shutil
import socket
import socketserver
import subprocess
import sys
import threading
import time
from io import BytesIO
from urllib.parse import urlparse

from PIL import Image

# ===== v-JEPA loader (lazy) =====
try:
    from .vjepa_infer import VJEPA
    VJEPA_AVAILABLE = True
except Exception:
    VJEPA_AVAILABLE = False
    VJEPA = None

_VJEPA = None
def _get_vjepa():
    global _VJEPA
    if _VJEPA is None:
        assert VJEPA_AVAILABLE, "vjepa_infer.py が見つからない/依存が足りません"
        os.makedirs("weights", exist_ok=True)
        _VJEPA = VJEPA("weights/vjepa_vitl16.pt")  # 置き場所は適宜
    return _VJEPA

# ===== QR =====
try:
    import qrcode
except ImportError:
    print("pip install qrcode[pil]")
    sys.exit(1)

def print_qr(url, label="Open this URL"):
    print(f"\n{label}: {url}\n")
    qr = qrcode.QRCode(border=1)
    qr.add_data(url); qr.make(fit=True); qr.print_ascii(invert=True)

# ===== HTTP Handler =====
class Handler(http.server.SimpleHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type,Authorization")
        self.end_headers()

    def _ok_json(self, payload, code=200):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        # 1) /analyze → 画像バイナリを受け取り v-JEPA へ
        if self.path == "/analyze":
            try:
                length = int(self.headers.get("Content-Length", "0") or 0)
                raw = self.rfile.read(length) if length > 0 else b""
                if not raw:
                    return self._ok_json({"ok": False, "error": "empty body"}, 400)
                try:
                    img = Image.open(BytesIO(raw)).convert("RGB")
                except Exception as e:
                    return self._ok_json({"ok": False, "error": f"invalid image: {e}"}, 400)

                try:
                    vj = _get_vjepa()
                except AssertionError as e:
                    return self._ok_json({"ok": False, "error": str(e)}, 500)

                try:
                    res = vj.embed(img)  # 例: {"score": float, "dim": int, ...}
                except Exception as e:
                    return self._ok_json({"ok": False, "error": f"vjepa error: {e}"}, 500)

                return self._ok_json({"ok": True, "vjepa": res})
            except Exception as e:
                return self._ok_json({"ok": False, "error": f"server error: {e}"}, 500)

        # 2) 既存: /event /snapshot など JSON ログ（互換）
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            payload = {"_raw": "<non-json>"}
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        line = {"ts": ts, "path": self.path, "data": payload}
        print(json.dumps(line, ensure_ascii=False), flush=True)
        self._ok_json({"ok": True})

# ===== Server bootstrap / Cloudflare tunnel =====
class ReusableThreadingTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

def run_http(bind, port, root):
    os.chdir(root)
    Handler.extensions_map.update({".html": "text/html"})
    with ReusableThreadingTCPServer((bind, port), Handler) as httpd:
        print(f"\nServing '{os.path.abspath(root)}' at {bind}:{port}  (Ctrl+C to stop)\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass

def start_cloudflared(cf_path, local_url):
    if not shutil.which(cf_path):
        raise FileNotFoundError(f"cloudflared not found: {cf_path}")
    proc = subprocess.Popen(
        [cf_path, "tunnel", "--url", local_url],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, bufsize=1
    )
    url = None
    url_pat = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com", re.I)
    for line in proc.stdout:
        print(line.rstrip())
        m = url_pat.search(line)
        if m:
            url = m.group(0)
            break
    return proc, url

def get_wsl_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = socket.gethostbyname(socket.gethostname())
    finally:
        s.close()
    return ip

def get_windows_ip():
    try:
        pwsh = "powershell.exe"
        cmd = [
            pwsh, "-NoProfile", "-Command",
            "(Get-NetIPAddress -AddressFamily IPv4 | "
            "Where-Object { $_.IPAddress -notmatch '169\\.254' -and $_.InterfaceAlias -match 'Wi-Fi|Ethernet' } | "
            "Select-Object -First 1 -ExpandProperty IPAddress)"
        ]
        out = subprocess.check_output(cmd, universal_newlines=True, stderr=subprocess.DEVNULL).strip()
        return out or None
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--root", default=".")
    ap.add_argument("--index", default="index.html")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--tunnel", action="store_true", help="Start Cloudflare quick tunnel and show its URL/QR")
    ap.add_argument("--cloudflared", default="cloudflared", help="Path to cloudflared")
    ap.add_argument("--prefer-windows-ip", dest="prefer_windows_ip", action="store_true",
                    help="QRにWindowsのLAN IPを優先表示（--tunnel未使用時）")
    args = ap.parse_args()

    t = threading.Thread(target=run_http, args=(args.bind, args.port, args.root), daemon=True)
    t.start()
    time.sleep(0.5)

    local_url = f"http://localhost:{args.port}/{args.index}"

    if args.tunnel:
        print("\n=== Starting Cloudflare quick tunnel ===\n")
        try:
            proc, cf_url = start_cloudflared(args.cloudflared, local_url)
        except FileNotFoundError as e:
            print(f"[ERROR] {e}\nwinget install Cloudflare.cloudflared で入れてね。")
            cf_url = None
            proc = None

        if cf_url:
            print_qr(cf_url, label="Public (HTTPS) URL")
        else:
            print("Failed to obtain trycloudflare URL. Falling back to local URL.")
            ip = (get_windows_ip() or get_wsl_ip())
            print_qr(f"http://{ip}:{args.port}/{args.index}", label="Local URL")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            if proc and proc.poll() is None:
                proc.terminate()
            print("\nBye")
    else:
        ip = (get_windows_ip() if args.prefer_windows_ip else get_wsl_ip()) or "127.0.0.1"
        print_qr(f"http://{ip}:{args.port}/{args.index}", label="Local URL")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nBye")

if __name__ == "__main__":
    main()

"""Access from other devices (phones) on the local network.

Browsers only allow the microphone on https:// (or localhost), and WSL2 is not reachable from the LAN, so
`windows/lan_proxy.py` runs on the Windows side: it terminates TLS (self-signed certificate) on
0.0.0.0:8443 and forwards to the server's LAN port, 127.0.0.1:8001 (a second listening socket of this same
server). Everything that arrives on the LAN port needs the access key, kept in `.lan_key`:
  * browsers: open https://<pc-ip>:8443/?key=<key> once (the QR code on the desktop page); the server swaps
    the key for an HttpOnly cookie and redirects to the clean URL
  * API clients: `Authorization: Bearer <key>` (or `?key=`)
The localhost port (8000) stays open without a key, as before. MING_API_KEY additionally requires a key
on every port (then that value is the key).

`MING_PROXY_AUTH`: if set, a request whose `X-Proxy-Auth` header matches that value is accepted without
a key, on the ports that otherwise need one. Meant for the home reverse proxy (NPM + Authelia). The phone
socket (8001) is normally served by `windows/lan_proxy.py`, a TLS-terminating byte pipe, so the server only
ever sees it as 127.0.0.1: the proxy rewrites each connection's head with an `X-Forwarded-For` naming its
real TCP peer (dropping client-sent copies of it and of X-Real-IP) and passes the rest, including the
`X-Proxy-Auth` header, through unchanged. Trusting the proxy also stops an auth-gated frontend from
re-challenging on every request — the app answers the proxied request instead of 401, so the login flow
terminates.
`MING_PROXY_TRUSTED` (comma-separated IPs and CIDRs): when set, the `X-Proxy-Auth` header is only honored
if the request comes from one of those sources (the `X-Forwarded-For` peer on the phone socket, the direct
TCP peer otherwise) - this is what stops other LAN machines from sending the header straight to the phone
socket and bypassing the key. Unset: honored from any source (the 0.3.12 behavior). Requests without a
valid header always still need the key, as before. The proxy stamps `X-Forwarded-For` on a connection's
first request only: that request's source verdict (stamped XFF peer, or the direct TCP peer when no
XFF) is remembered per client (ip, port) for a short while, and the connection's later keep-alive
requests - which carry no header - inherit it.
"""
from __future__ import annotations

import hmac
import ipaddress
import json
import os
import secrets
import time
from pathlib import Path
from urllib.parse import parse_qsl, urlencode

DATA = Path(os.environ.get("MING_DATA_DIR") or Path(__file__).resolve().parent)
KEY_FILE = DATA / ".lan_key"
INFO_FILE = DATA / ".lan_info.json"  # written by windows/lan_proxy.py: {"ips": [...], "port": 8443, "pid": ..}
LAN_PORT = int(os.environ.get("MING_LAN_PORT", "8001"))
COOKIE = "ming_key"
_SOURCE_TTL = 300.0  # how long a per-source-port source verdict is kept (keep-alive connections)
_SOURCE_MAX = 1024   # bound on remembered source ports (phones use a few; evict the oldest)


def _trusted_sources() -> list | None:
    """The networks MING_PROXY_TRUSTED allows the X-Proxy-Auth header from (None when unset: any source)."""
    raw = os.environ.get("MING_PROXY_TRUSTED", "").strip()
    if not raw:
        return None
    nets = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            try:
                nets.append(ipaddress.ip_network(part, strict=False))
            except ValueError:
                print(f"WARNING: ignoring unparseable MING_PROXY_TRUSTED entry: {part}")
    return nets


def _source_trusted(scope) -> bool:
    """Whether this request's source (the X-Forwarded-For peer on the phone socket, the direct TCP peer
    otherwise) is in MING_PROXY_TRUSTED; a missing or unparseable source is never trusted."""
    peer = (scope.get("client") or (None, None))[0]
    if peer in ("127.0.0.1", "::1"):
        xff = next((v for k, v in scope.get("headers", []) if k == b"x-forwarded-for"), b"")
        ip = xff.decode(errors="ignore").split(",")[0].strip()
    else:
        ip = peer or ""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return any(addr in net for net in _trusted_sources() or [])


def access_key() -> str:
    if os.environ.get("MING_API_KEY"):
        return os.environ["MING_API_KEY"]
    try:
        k = KEY_FILE.read_text().strip()
        if k:
            return k
    except FileNotFoundError:
        pass
    k = secrets.token_urlsafe(12)
    KEY_FILE.write_text(k + "\n")
    os.chmod(KEY_FILE, 0o600)
    return k


def lan_info() -> dict | None:
    """Where the Windows-side proxy is listening, if it is running."""
    try:
        info = json.loads(INFO_FILE.read_text())
    except (FileNotFoundError, ValueError):
        return None
    if time.time() - info.get("heartbeat", 0) > 30:  # the proxy refreshes this every 10 s
        return None
    return info


def phone_urls() -> list[str]:
    info = lan_info()
    if not info:
        return []
    return [f"https://{ip}:{info['port']}/?key={access_key()}" for ip in info["ips"]]


def qr_svg(text: str) -> str:
    import segno
    import io
    buf = io.BytesIO()
    segno.make(text, error="m").save(buf, kind="svg", scale=6, border=2, dark="#111", light="#fff", xmldecl=False,
                                     svgns=True, nl=False)
    return buf.getvalue().decode()


def _key_from(scope) -> str | None:
    for name, value in scope.get("headers", []):
        if name == b"authorization" and value[:7].lower() == b"bearer ":
            return value[7:].decode().strip()
        if name == b"cookie":
            for part in value.decode(errors="ignore").split(";"):
                k, _, v = part.strip().partition("=")
                if k == COOKIE:
                    return v
    q = dict(parse_qsl(scope.get("query_string", b"").decode()))
    return q.get("key")


class AccessControl:
    """ASGI middleware: key check for the LAN port (or every port with MING_API_KEY)."""

    def __init__(self, app):
        self.app = app
        # source port -> (verdict, expiry): a connection's first request is the only one carrying the
        # proxy's X-Forwarded-For, so its source verdict also applies to the connection's later
        # keep-alive requests (same source port, no header).
        self._source_trust = {}

    def _needs_key(self, scope) -> bool:
        if os.environ.get("MING_API_KEY"):
            return True
        server = scope.get("server") or (None, None)
        return server[1] == LAN_PORT

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket") or not self._needs_key(scope):
            return await self.app(scope, receive, send)
        key = access_key()
        given = _key_from(scope)
        ok = given is not None and hmac.compare_digest(given.encode(), key.encode())
        if not ok:
            # Trusted reverse proxy (see module doc): the proxy's own header passes, so a gated
            # frontend (Authelia) is never answered 401 and the login loop can't happen.
            trust = os.environ.get("MING_PROXY_AUTH")
            if trust:
                proxied = next((v for k, v in scope.get("headers", []) if k == b"x-proxy-auth"), b"")
                if hmac.compare_digest(proxied, trust.encode()):
                    # 0.3.13: the header only counts from trusted sources (0.3.12: from any source).
                    # The proxy stamps X-Forwarded-For on a connection's first request only, so the
                    # source verdict is fixed on that first request (stamped XFF peer, or the direct
                    # TCP peer when no XFF) and remembered per client (ip, port). Later requests on
                    # the connection no longer carry the header and inherit the unexpired verdict -
                    # which also ignores a forged X-Forwarded-For a direct client might try to add on
                    # a later request of the same connection. A fresh (ip, port) with an expired or
                    # missing entry is decided on arrival.
                    sources = _trusted_sources()
                    if sources is None:
                        ok = True
                    else:
                        key = scope.get("client") or (None, 0)
                        now = time.monotonic()
                        ent = self._source_trust.get(key)
                        if ent is not None and ent[1] > now:
                            ok = ent[0]
                        else:
                            ok = _source_trusted(scope)
                            self._source_trust[key] = (ok, now + _SOURCE_TTL)
                            if len(self._source_trust) > _SOURCE_MAX:
                                self._source_trust.pop(next(iter(self._source_trust)))
        q = dict(parse_qsl(scope.get("query_string", b"").decode()))
        if ok and scope["type"] == "http" and "key" in q and scope["method"] == "GET" and not scope["path"].startswith("/v1"):
            # browser login: keep the key in a cookie and drop it from the address bar
            q.pop("key")
            loc = scope["path"] + ("?" + urlencode(q) if q else "")
            # Lax, not Strict: a link opened from another app (QR scanner, chat, mail) is a cross-site navigation,
            # and browsers withhold Strict cookies from it, including on this redirect, so it would land on "denied"
            cookie = f"{COOKIE}={key}; Path=/; Max-Age=31536000; HttpOnly; Secure; SameSite=Lax"
            await send({"type": "http.response.start", "status": 303,
                        "headers": [(b"location", loc.encode()), (b"set-cookie", cookie.encode()),
                                    (b"cache-control", b"no-store")]})
            await send({"type": "http.response.body", "body": b""})
            return
        if ok:
            return await self.app(scope, receive, send)
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 4401})
            return
        if scope["path"].startswith("/v1") or scope["path"].startswith("/api"):
            body = json.dumps({"error": {"message": "missing or wrong access key (Authorization: Bearer <key>)",
                                         "type": "invalid_request_error", "code": "invalid_api_key"}}).encode()
            ctype = b"application/json"
        else:
            body = _DENIED.encode()
            ctype = b"text/html; charset=utf-8"
        await send({"type": "http.response.start", "status": 401,
                    "headers": [(b"content-type", ctype), (b"cache-control", b"no-store")]})
        await send({"type": "http.response.body", "body": body})


_DENIED = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>Ming Voice</title><body style="font:16px system-ui;background:#111;color:#ddd;padding:24px;max-width:480px;margin:auto">
<h2>Ming Voice</h2><p>This link needs the access key. On the computer running Ming Voice, open
<b>http://localhost:8000</b>, press <b>Phone</b>, and scan the QR code (or copy the link).</p>
<form method=get action="/" style="display:flex;gap:8px;margin-top:20px"><input name=key placeholder="or enter the access key"
autocomplete=off autocapitalize=off spellcheck=false style="flex:1;min-width:0;font:16px system-ui;padding:9px 12px;border-radius:10px;border:1px solid #444;background:#1b1b1b;color:#ddd">
<button style="font:16px system-ui;padding:9px 14px;border-radius:10px;border:0;background:#7aa2ff;color:#111">Open</button></form></body>"""

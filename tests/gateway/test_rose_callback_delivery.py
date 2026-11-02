"""deliver: rose_callback (Rose Command Centre): one signed callback per webhook run, with the settled reply.

The destination is admitted with the delivery (callback URL and Rose request id only, never the route
secret), interim and final ``send()`` calls post nothing, and ``_finalize_delivery`` posts once when the
admission is terminal. The callback URL must name a host on the route's ``callback_hosts``, at admission
and again at send time. Rose's verifier (server/hermesClient.ts ``verifyCallbackSignature``) checks
``t=<unix>,sha256=<hex HMAC-SHA256 of "<unix>.<body>">`` within a 300 s window.
"""
import asyncio
import hashlib
import hmac
import json
import re
import time
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.webhook import WebhookAdapter, _WebhookDeliveryIdentity, _rose_callback_failure
from tests.gateway.fixtures.webhook_route_authority import mount_authority

_ROUTE_KEY = "dummy-credential-rose-route"
_ROUTE = "rose-dispatch"
_CALLBACK = "https://rose.example.test/api/hermes-callback"
_HOSTS = ["rose.example.test"]
_AT = "@"  # the userinfo separator, kept out of the literals so no case reads as an address


def _rose_route(**overrides):
    route = {"prompt": "{payload.prompt}", "deliver": "rose_callback", "callback_hosts": list(_HOSTS),
             "deliver_extra": {"callback_url": "{payload.callback_url}"}}
    route.update(overrides)
    return route


def _adapter(**route_overrides):
    return WebhookAdapter(PlatformConfig(enabled=True, extra={
        "secret": _ROUTE_KEY, "routes": {_ROUTE: _rose_route(**route_overrides), "plain": {"prompt": "{text}"}}}))


def _destination(request_id="rose-req-001", url=_CALLBACK):
    return {"deliver": "rose_callback", "deliver_extra": {"callback_url": url, "rose_request_id": request_id}}


def _rose_verifies(body: bytes, header: str, key: str = _ROUTE_KEY, max_age: int = 300) -> bool:
    """Python mirror of Rose's verifyCallbackSignature."""
    stamp, sig = re.search(r"t=(\d+)", header), re.search(r"sha256=([0-9a-f]+)", header)
    if not stamp or not sig or abs(int(time.time()) - int(stamp.group(1))) > max_age:
        return False
    digest = hmac.new(key.encode(), stamp.group(1).encode() + b"." + body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(f"t={stamp.group(1)},sha256={digest}", header)


# --- The destination: what admission persists ---


def test_destination_keeps_only_callback_url_and_request_id():
    from gateway.platforms.webhook_delivery import validate_destination
    delivery = _destination()
    delivery["deliver_extra"].update({"secret": _ROUTE_KEY, "payload": {"prompt": "x"}, "chat_id": "1"})
    assert validate_destination(delivery) == _destination()


@pytest.mark.parametrize("url", [
    "http://rose.example.test/api/hermes-callback",  # plain http off loopback
    "{payload.callback_url}",                          # template left unrendered: the payload had no URL
    "https://rose.example.test/api/other",             # not Rose's callback path
    "https://rose.example.test/api/hermes-callback?next=x",
    "https://ROSE.example.test/api/hermes-callback",   # not in the rebuilt form admission stores
    "ftp://rose.example.test/x", "https://", "", None, 7])
def test_destination_refuses_a_bad_callback_url(url):
    from gateway.platforms.webhook_delivery import validate_destination
    from hermes_state_runtime import RuntimeStoreError
    with pytest.raises(RuntimeStoreError):
        validate_destination(_destination(url=url))


@pytest.mark.parametrize("request_id", ["", None, 5])
def test_destination_refuses_a_missing_request_id(request_id):
    from gateway.platforms.webhook_delivery import validate_destination
    from hermes_state_runtime import RuntimeStoreError
    with pytest.raises(RuntimeStoreError):
        validate_destination(_destination(request_id=request_id))


def test_destination_allows_http_to_loopback_and_needs_no_home_channel():
    from gateway.platforms.webhook_delivery import snapshot_destination
    local = _destination(url="http://127.0.0.1:8999/api/hermes-callback")
    # snapshot_destination looks up a home channel for chat platforms; this target has none to find.
    assert snapshot_destination(_adapter(), local) == local


@pytest.mark.parametrize("overrides", [
    {"coalesce": {"key": "{payload.id}", "window_seconds": 5}}, {"deliver_only": True}, {"cron_job": "nightly"},
    {"secret": "INSECURE_NO_AUTH"},
    {"callback_hosts": None}, {"callback_hosts": []}, {"callback_hosts": "rose.example.test"},
    {"callback_hosts": ["*.example.test"]}, {"callback_hosts": ["https://rose.example.test"]},
    {"callback_hosts": ["Rose.example.test"]}, {"callback_hosts": ["rose.example.test:443"]},
    {"callback_hosts": ["rose.example.test", 7]}])
def test_route_validation_refuses_unsupported_rose_callback_routes(overrides):
    adapter = _adapter()
    with pytest.raises(ValueError):
        adapter._validate_route(_ROUTE, _rose_route(**overrides))


def test_route_validation_accepts_named_hosts_and_loopback():
    _adapter()._validate_route(_ROUTE, _rose_route(callback_hosts=["rose.example.test", "127.0.0.1", "::1"]))


# --- The callback URL allow-list (the route's callback_hosts) ---


@pytest.mark.parametrize("url,hosts", [
    ("https://evil.example.test/api/hermes-callback", _HOSTS),
    ("https://rose.example.test.evil.example/api/hermes-callback", _HOSTS),
    ("https://rose.example.test./api/hermes-callback", _HOSTS),          # trailing dot
    (f"https://rose.example.test{_AT}evil.example/api/hermes-callback", _HOSTS),
    (f"https://evil.example\\{_AT}rose.example.test/api/hermes-callback", _HOSTS),
    (f"https://user{_AT}rose.example.test/api/hermes-callback", _HOSTS),
    ("https://rose.example.test:8443/api/hermes-callback", _HOSTS),       # a port off loopback
    ("https://rose.example.test:notaport/api/hermes-callback", _HOSTS),
    ("https://rose.example.test/api/hermes-callback/", _HOSTS),
    ("https://rose.example.test/other/../api/hermes-callback", _HOSTS),
    ("https://rose.example.test/api/hermes-callback#x", _HOSTS),
    ("http://rose.example.test/api/hermes-callback", _HOSTS),
    ("http://127.0.0.1:8999/api/hermes-callback", _HOSTS),               # loopback, but not listed
    (_CALLBACK, ()), (_CALLBACK, "rose.example.test"),                    # nothing listed; a bare string
])
def test_callback_url_off_the_allow_list_is_refused(url, hosts):
    from gateway.platforms.webhook_delivery import rose_callback_url
    assert rose_callback_url(url, hosts) is None


@pytest.mark.parametrize("url,hosts,rebuilt", [
    ("https://ROSE.example.test/api/hermes-callback", _HOSTS, _CALLBACK),
    ("http://127.0.0.1:8999/api/hermes-callback", ["127.0.0.1"], "http://127.0.0.1:8999/api/hermes-callback"),
    ("http://[::1]:8999/api/hermes-callback", ["::1"], "http://[::1]:8999/api/hermes-callback"),
])
def test_allowed_callback_url_is_rebuilt_from_its_parts(url, hosts, rebuilt):
    from gateway.platforms.webhook_delivery import rose_callback_url
    assert rose_callback_url(url, hosts) == rebuilt


# --- Admission: the signed Rose dispatch commits its destination ---


async def _post_dispatch(client, payload, request_id="rose-req-001"):
    body = json.dumps(payload).encode()
    headers = {"X-Hub-Signature-256": "sha256=" + hmac.new(_ROUTE_KEY.encode(), body, hashlib.sha256).hexdigest()}
    if request_id is not None:
        headers["X-Rose-Request-Id"] = request_id
    return await client.post(f"/webhooks/{_ROUTE}", data=body, headers=headers)


@pytest.mark.asyncio
async def test_dispatch_commits_rose_destination_without_the_secret():
    adapter = _adapter()
    app = web.Application()
    mount_authority(app, adapter)
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    async with TestClient(TestServer(app)) as client:
        # The upper-case host is stored in the rebuilt form, the one the callback is sent to.
        response = await _post_dispatch(client, {"payload": {
            "prompt": "hello", "callback_url": "https://ROSE.example.test/api/hermes-callback"}})
        assert response.status == 202, await response.text()
        runner = adapter._message_handler.__self__
        row = runner.session_authority.db._read_one("SELECT payload_json FROM session_admissions")
        assert json.loads(row["payload_json"])["native_text_v1"]["webhook_delivery"] == _destination()
        assert _ROUTE_KEY not in row["payload_json"]


@pytest.mark.asyncio
@pytest.mark.parametrize("payload,request_id", [
    ({"payload": {"prompt": "hello", "callback_url": _CALLBACK}}, None),  # no X-Rose-Request-Id header
    ({"payload": {"prompt": "hello"}}, "rose-req-001"),                  # no callback URL to render
    ({"payload": {"prompt": "hello", "callback_url": "https://evil.example.test/api/hermes-callback"}},
     "rose-req-001"),                                                    # a host off callback_hosts
])
async def test_dispatch_without_request_id_or_allowed_callback_url_is_refused(payload, request_id):
    adapter = _adapter()
    app = web.Application()
    mount_authority(app, adapter)
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    async with TestClient(TestServer(app)) as client:
        response = await _post_dispatch(client, payload, request_id=request_id)
        assert response.status == 400  # Rose does not retry a 4xx
        runner = adapter._message_handler.__self__
        assert runner.session_authority.db._read_one("SELECT COUNT(*) AS n FROM session_admissions")["n"] == 0


# --- send(): interim and final messages post nothing ---


@pytest.mark.asyncio
async def test_send_leg_posts_nothing(monkeypatch):
    posted = []

    async def fake_post(url, body, headers):
        posted.append(url)
        return SendResult(success=True)

    monkeypatch.setattr(WebhookAdapter, "_post_rose_callback_once", staticmethod(fake_post))
    adapter = _adapter()
    for text in ("⏳ Retrying in 2.6s...", "Context is getting long", "The final answer"):
        assert (await adapter._deliver_to(_destination(), "webhook:v2:x", text)).success
    assert posted == []


# --- _finalize_delivery: one callback once the admission is terminal ---


class _Authority:
    def __init__(self):
        self.db, self.waiters = object(), {}

    def _receipt(self, row):
        return row


def _event(adapter, route=_ROUTE, delivery_id="delivery-1"):
    from gateway.platforms.event import MessageEvent
    chat_id = _WebhookDeliveryIdentity.from_parts(None, route, delivery_id).session_chat_id
    return MessageEvent(text="x", source=adapter.build_source(chat_id=chat_id, chat_type="webhook"),
                        message_id=delivery_id)


def _settle(monkeypatch, rows, result=None, destination=None):
    """Feed ``_finalize_delivery`` admission rows in order (the last repeats) and a settled result."""
    seq = list(rows)
    monkeypatch.setattr("hermes_state_runtime.get_session_admission",
                        lambda db, *, admission_id: seq.pop(0) if len(seq) > 1 else seq[0])
    monkeypatch.setattr("gateway.session_results.admission_result", lambda db, admission_id: result)
    monkeypatch.setattr("gateway.platforms.webhook_delivery.retained_destination",
                        lambda adapter, chat_id: destination or _destination())


def _capture(monkeypatch, status=200):
    posted = []

    async def fake_post(url, body, headers):
        posted.append((url, json.loads(body), headers, body))
        return SendResult(success=status < 300, error=None if status < 300 else f"HTTP {status}")

    monkeypatch.setattr(WebhookAdapter, "_post_rose_callback_once", staticmethod(fake_post))
    return posted


async def _finalize(adapter, event):
    adapter._end_webhook_session = lambda *args: asyncio.sleep(0)
    await adapter._finalize_delivery(event, _Authority(), SimpleNamespace(admission_id="adm-1"))


_DONE = {"status": "terminal", "outcome": "completed"}


@pytest.mark.asyncio
async def test_completed_run_posts_one_signed_callback_with_the_settled_reply(monkeypatch):
    _settle(monkeypatch, [_DONE], {"result": {"final_response": "The research summary", "failed": False}})
    posted = _capture(monkeypatch)
    adapter = _adapter()
    await _finalize(adapter, _event(adapter))
    assert len(posted) == 1
    url, body, headers, raw = posted[0]
    assert url == _CALLBACK
    assert body["status"] == "completed" and body["summary"] == "The research summary" and body["error"] is None
    assert body["rose_request_id"] == "rose-req-001" and body["job_id"] == "delivery-1"
    assert set(body) == {"job_id", "rose_request_id", "status", "summary", "artifacts", "duration_ms",
                         "token_cost_usd", "error"}
    assert _rose_verifies(raw, headers["X-Hermes-Signature"])


@pytest.mark.asyncio
async def test_callback_waits_for_the_run_to_settle(monkeypatch):
    _settle(monkeypatch, [{"status": "started", "outcome": None}, _DONE],
            {"result": {"final_response": "late answer", "failed": False}})
    posted = _capture(monkeypatch)
    adapter = _adapter()
    authority = _Authority()
    adapter._end_webhook_session = lambda *args: asyncio.sleep(0)
    task = asyncio.create_task(
        adapter._finalize_delivery(_event(adapter), authority, SimpleNamespace(admission_id="adm-1")))
    for _ in range(50):
        if "adm-1" in authority.waiters:
            break
        await asyncio.sleep(0)
    assert posted == []  # nothing is sent while the run is in flight
    authority.waiters["adm-1"].set_result("late answer")
    await task
    assert [p[1]["summary"] for p in posted] == ["late answer"]


@pytest.mark.asyncio
@pytest.mark.parametrize("row,result,error", [
    ({"status": "terminal", "outcome": "failed"}, {"result": {"final_response": "boom", "failed": True}},
     "Agent run ended: failed"),
    ({"status": "terminal", "outcome": "cancelled"}, None, "Agent run ended: cancelled"),
    ({"status": "terminal", "outcome": "interrupted"}, None, "Agent run ended: interrupted"),
    (_DONE, {"result": {"final_response": "", "failed": False}}, "Agent produced no response"),
    (_DONE, {"result": {"final_response": "partial notes", "failed": True}}, "Agent run failed"),
    ({"status": "unknown", "outcome": None}, None, "Agent run did not settle"),
])
async def test_unsuccessful_runs_post_failed(monkeypatch, row, result, error):
    _settle(monkeypatch, [row], result)
    posted = _capture(monkeypatch)
    adapter = _adapter()
    await _finalize(adapter, _event(adapter))
    assert [(p[1]["status"], p[1]["error"]) for p in posted] == [("failed", error)]


@pytest.mark.asyncio
async def test_waiter_failure_posts_failed(monkeypatch):
    _settle(monkeypatch, [{"status": "started", "outcome": None}])
    posted = _capture(monkeypatch)
    adapter = _adapter()
    authority = _Authority()
    adapter._end_webhook_session = lambda *args: asyncio.sleep(0)
    task = asyncio.create_task(
        adapter._finalize_delivery(_event(adapter), authority, SimpleNamespace(admission_id="adm-1")))
    for _ in range(50):
        if "adm-1" in authority.waiters:
            break
        await asyncio.sleep(0)
    authority.waiters["adm-1"].set_exception(RuntimeError("unknown_execution"))
    await task
    assert [(p[1]["status"], p[1]["error"]) for p in posted] == [("failed", "Agent run did not settle")]


@pytest.mark.asyncio
async def test_shutdown_while_waiting_posts_nothing(monkeypatch):
    _settle(monkeypatch, [{"status": "started", "outcome": None}])
    posted = _capture(monkeypatch)
    adapter = _adapter()
    authority = _Authority()
    task = asyncio.create_task(
        adapter._finalize_delivery(_event(adapter), authority, SimpleNamespace(admission_id="adm-1")))
    for _ in range(50):
        if "adm-1" in authority.waiters:
            break
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert posted == []


@pytest.mark.asyncio
async def test_other_routes_post_nothing(monkeypatch):
    _settle(monkeypatch, [_DONE], {"result": {"final_response": "answer", "failed": False}})
    posted = _capture(monkeypatch)
    adapter = _adapter()
    await _finalize(adapter, _event(adapter, route="plain"))
    assert posted == []


@pytest.mark.asyncio
async def test_unreadable_destination_posts_nothing(monkeypatch):
    _settle(monkeypatch, [_DONE], {"result": {"final_response": "answer", "failed": False}})

    def refuse(adapter, chat_id):
        raise RuntimeError("admission_conflict")

    monkeypatch.setattr("gateway.platforms.webhook_delivery.retained_destination", refuse)
    posted = _capture(monkeypatch)
    adapter = _adapter()
    await _finalize(adapter, _event(adapter))
    assert posted == []


@pytest.mark.asyncio
async def test_stored_url_off_the_routes_hosts_now_posts_nothing(monkeypatch):
    # Admitted before the route's callback_hosts changed: the send-time check refuses it.
    _settle(monkeypatch, [_DONE], {"result": {"final_response": "answer", "failed": False}},
            destination=_destination(url="https://old.example.test/api/hermes-callback"))
    posted = _capture(monkeypatch)
    adapter = _adapter()
    await _finalize(adapter, _event(adapter))
    assert posted == []


@pytest.mark.asyncio
async def test_callback_goes_to_the_rebuilt_url_not_the_stored_string(monkeypatch):
    _settle(monkeypatch, [_DONE], {"result": {"final_response": "answer", "failed": False}},
            destination=_destination(url="https://ROSE.example.test/api/hermes-callback"))
    posted = _capture(monkeypatch)
    adapter = _adapter()
    await _finalize(adapter, _event(adapter))
    assert [p[0] for p in posted] == [_CALLBACK]


@pytest.mark.asyncio
@pytest.mark.parametrize("hosts", [None, []])
async def test_no_callback_hosts_posts_nothing(monkeypatch, hosts):
    posted = _capture(monkeypatch)
    result = await _adapter()._post_rose_callback("answer", _destination(), _ROUTE_KEY, delivery_id="d",
                                                  error=None, started_at=None, callback_hosts=hosts)
    assert not result.success and posted == []


@pytest.mark.asyncio
@pytest.mark.parametrize("unusable", ["   ", "INSECURE_NO_AUTH"])
async def test_unusable_route_secret_posts_nothing(monkeypatch, unusable):
    posted = _capture(monkeypatch)
    result = await _adapter()._post_rose_callback("answer", _destination(), unusable, delivery_id="d", error=None,
                                                  started_at=None, callback_hosts=_HOSTS)
    assert not result.success and posted == []


@pytest.mark.asyncio
async def test_callback_is_signed_with_the_routes_own_secret(monkeypatch):
    route_key = "dummy-credential-route-own"
    _settle(monkeypatch, [_DONE], {"result": {"final_response": "answer", "failed": False}})
    posted = _capture(monkeypatch)
    route = _rose_route()
    route["secret"] = route_key  # the route's own secret; the adapter-wide one stays _ROUTE_KEY
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"secret": _ROUTE_KEY, "routes": {_ROUTE: route}}))
    await _finalize(adapter, _event(adapter))
    _url, _body, headers, raw = posted[0]
    assert _rose_verifies(raw, headers["X-Hermes-Signature"], key=route_key)
    assert not _rose_verifies(raw, headers["X-Hermes-Signature"], key=_ROUTE_KEY)


@pytest.mark.asyncio
async def test_route_changed_after_admission_posts_nothing(monkeypatch):
    _settle(monkeypatch, [_DONE], {"result": {"final_response": "answer", "failed": False}})
    posted = _capture(monkeypatch)
    adapter = _adapter()

    def reload_then_read(adapter_, chat_id):
        # A dynamic-route reload inside retained_destination replaced the route with a plain one.
        adapter_._routes[_ROUTE] = {"prompt": "{text}"}
        return _destination()

    monkeypatch.setattr("gateway.platforms.webhook_delivery.retained_destination", reload_then_read)
    await _finalize(adapter, _event(adapter))
    assert posted == []


# --- What counts as a failed run (provider failures returned as reply text) ---


@pytest.mark.parametrize("content,error", [
    ("⚠️ API call failed after 3 retries: Error code: 401\n   🔌 Provider: anthropic\n"
     "   📝 Final error: Error code: 401 - invalid x-api-key", "📝 Final error: Error code: 401 - invalid x-api-key"),
    ("Error code: 500 - internal server error", "Error code: 500 - internal server error"),
    ("Error code: 429 - rate limited", "Error code: 429 - rate limited"),
    ("{'type': 'error', 'error': {'type': 'overloaded'}}", "{'type': 'error', 'error': {'type': 'overloaded'}}"),
    ("⏳ Retrying in 2.6s (attempt 1/3)...", "⏳ Retrying in 2.6s (attempt 1/3)..."),
    ("Traceback (most recent call last):\n  File \"x.py\"", "Traceback (most recent call last):"),
])
def test_provider_failure_text_is_a_failure(content, error):
    assert _rose_callback_failure(content) == error


@pytest.mark.parametrize("content", [
    "Company X is a C&I solar developer with three sites.",
    "A" * 1200 + " Traceback (most recent call last): quoted from a public bug report",
])
def test_research_text_is_not_a_failure(content):
    assert _rose_callback_failure(content) is None


@pytest.mark.asyncio
async def test_completed_outcome_with_failure_text_posts_failed_with_a_truncated_summary(monkeypatch, caplog):
    content = "📝 Final error: Error code: 529 - overloaded\n" + "x" * 3000
    _settle(monkeypatch, [_DONE], {"result": {"final_response": content, "failed": False}})
    posted = _capture(monkeypatch)
    adapter = _adapter()
    with caplog.at_level("DEBUG", logger="gateway.platforms.webhook"):
        await _finalize(adapter, _event(adapter))
    body = posted[0][1]
    assert body["status"] == "failed" and body["error"] == "📝 Final error: Error code: 529 - overloaded"
    assert body["summary"] == content[:2000]
    assert "marking job rose-req-001 FAILED" in caplog.text and "overloaded" not in caplog.text


@pytest.mark.asyncio
async def test_failure_line_from_the_reply_is_not_logged(monkeypatch, caplog):
    # A research reply whose first line trips a failure marker: Rose gets the line, the log does not.
    content = "❌ planted-reply-text-7f3a no public website found\nMore findings follow."
    _settle(monkeypatch, [_DONE], {"result": {"final_response": content, "failed": False}})
    posted = _capture(monkeypatch)
    adapter = _adapter()
    with caplog.at_level("DEBUG"):
        await _finalize(adapter, _event(adapter))
    assert posted[0][1]["error"] == "❌ planted-reply-text-7f3a no public website found"
    assert "planted-reply-text-7f3a" not in caplog.text


# --- HTTP leg against a real local server ---


async def _callback_server(statuses):
    seen = []

    async def handler(request):
        seen.append((request.path, await request.read(), dict(request.headers)))
        status = statuses.pop(0) if statuses else 200
        if status == 302:
            raise web.HTTPFound("/elsewhere")
        return web.json_response({"ok": status < 300}, status=status)

    async def elsewhere(request):
        # Any method succeeds here, so a client that followed the redirect would report success.
        seen.append((request.path, await request.read(), dict(request.headers)))
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_post("/api/hermes-callback", handler)
    app.router.add_route("*", "/elsewhere", elsewhere)
    server = TestServer(app)
    await server.start_server()
    return server, seen


@pytest.mark.asyncio
async def test_signature_matches_roses_verifier_over_the_wire():
    server, seen = await _callback_server([200])
    try:
        url = str(server.make_url("/api/hermes-callback"))
        result = await _adapter()._post_rose_callback("ünïcode answer", _destination(url=url), _ROUTE_KEY,
                                                      delivery_id="d-1", error=None, started_at=time.time(),
                                                      callback_hosts=["127.0.0.1"])
        assert result.success and len(seen) == 1
        _path, raw, headers = seen[0]
        assert headers["Content-Type"] == "application/json"
        assert _rose_verifies(raw, headers["X-Hermes-Signature"])
        assert not _rose_verifies(raw, headers["X-Hermes-Signature"], key="dummy-credential-other")
        assert json.loads(raw)["summary"] == "ünïcode answer"
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_a_failed_attempt_is_retried_once_and_re_signed(monkeypatch):
    # Each read of the clock is 2 s later, so a header reused from the first attempt would carry the old time.
    import gateway.platforms.webhook as webhook_module
    start, calls = time.time(), []

    def ticking_time():
        calls.append(1)
        return start + 2 * len(calls)

    monkeypatch.setattr(webhook_module.time, "time", ticking_time)
    server, seen = await _callback_server([500, 200])
    try:
        url = str(server.make_url("/api/hermes-callback"))
        result = await _adapter()._post_rose_callback("answer", _destination(url=url), _ROUTE_KEY,
                                                      delivery_id="d-1", error=None, started_at=None,
                                                      callback_hosts=["127.0.0.1"])
        assert result.success and len(seen) == 2
        first, second = (headers["X-Hermes-Signature"] for _p, _raw, headers in seen)
        assert first != second
        assert all(_rose_verifies(raw, headers["X-Hermes-Signature"]) for _p, raw, headers in seen)
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_each_attempt_has_a_30_second_timeout(monkeypatch):
    import aiohttp
    totals, real_timeout = [], aiohttp.ClientTimeout

    def recording_timeout(*args, **kwargs):
        totals.append(kwargs.get("total"))
        return real_timeout(*args, **kwargs)

    monkeypatch.setattr(aiohttp, "ClientTimeout", recording_timeout)
    server, _seen = await _callback_server([200])
    try:
        url = str(server.make_url("/api/hermes-callback"))
        result = await _adapter()._post_rose_callback("answer", _destination(url=url), _ROUTE_KEY,
                                                      delivery_id="d-1", error=None, started_at=None,
                                                      callback_hosts=["127.0.0.1"])
        assert result.success and totals == [30]
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_redirects_are_not_followed():
    server, seen = await _callback_server([302, 302])
    try:
        url = str(server.make_url("/api/hermes-callback"))
        result = await _adapter()._post_rose_callback("answer", _destination(url=url), _ROUTE_KEY,
                                                      delivery_id="d-1", error=None, started_at=None,
                                                      callback_hosts=["127.0.0.1"])
        assert not result.success
        assert [path for path, _raw, _headers in seen] == ["/api/hermes-callback", "/api/hermes-callback"]
    finally:
        await server.close()

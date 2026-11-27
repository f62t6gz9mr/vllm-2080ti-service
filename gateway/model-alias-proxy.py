#!/usr/bin/env python3
"""LAN 网关: 兼容旧客户端配置, 零改动切到 vLLM.

端口 8001: OpenAI 兼容别名代理 (任意 model 名 -> 真实 vLLM served name)
端口 11434: Ollama 兼容层 (供按 Ollama 地址配置的旧客户端零改动接入)
    GET  /api/version  -> {"version":"0.34.0"}
    GET  /api/tags     -> 报告模型 qwen3.8-27b
    POST /api/chat     -> 翻译成 vLLM /v1/chat/completions (支持 stream NDJSON)
    /v1/*              -> OpenAI 直通 (model 一律改写为真实名)
真实上游: http://127.0.0.1:8000 (vLLM)
"""
import http.client
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# 上游 socket 读超时(秒)。这是"单次 read 最多等多久", 不是请求总时长。
# 原来的 1800 等于没有超时: 上游一异常, 网关就替客户端死等 30 分钟。
UPSTREAM_TIMEOUT = 1200
# 2026-11-21: 每张卡 KV 池只有约 26.4 万 tokens（2x2080Ti / 262K 满血档）。原来只有 1 个槽，
# 一个客户端就能把局域网钉死 85 分钟。现在 vLLM 侧 MAX_NUM_SEQS=2，这里再加一道门口排队：
# 最多 MAX_INFLIGHT 个请求同时进上游，其余在网关排队等（先到先得），避免 vLLM 内部两个会话
# 把 KV 塞满后互相抢占（抢占 = 被抢的请求 num_computed_tokens 归零，前缀要重算）。
# 排队只在网关生效，所以绕过网关直连 :8000 不受这里限制。
MAX_INFLIGHT = 2
MAX_QUEUE = 64            # 排队超过这个数直接 429，免得无限堆积
INFLIGHT = threading.BoundedSemaphore(MAX_INFLIGHT)
QUEUE_LOCK = threading.Lock()
QUEUE_LEN = [0]

# 客户端没给 num_predict/max_tokens 时强制注入的生成上限。
# 不注入的话, 单次请求最坏能一路生成到 max_model_len=262144,
# 而本服务是 MAX_NUM_SEQS=1 的单槽配置 —— 一个客户端就能把整个局域网钉死约 85 分钟。
DEFAULT_MAX_TOKENS = 65536
# "上游异常"的判定集合。IncompleteRead(流被提前掐断, 没发结束 chunk)是这里最容易被漏掉的一种,
# 09-20 02:06:11 那两条 traceback 就是它: 线程静默死 -> 客户端永远等不到 done -> 表现为"卡住"。
UPSTREAM_BROKEN = (http.client.IncompleteRead, http.client.RemoteDisconnected,
                   ConnectionResetError, BrokenPipeError, OSError)

UPSTREAM = "http://127.0.0.1:8000"
# 2026-10-02: 服务升到 v0.2.2-post3 并改走官方 DFlash2 档，vLLM 的 --served-model-name 变了。
# 这里必须与 ~/start-dflash2.sh 的 --set SERVED_NAME 完全一致（旧值：
# qwen38-27b-nvfp4-mtp-fp8kv-1x240k-text-image-cu130，回滚 MTP 档时记得改回去）。
REAL_MODEL = "qwen38-27b-nvfp4-dflash2-fp8kv-1x262k-text-image-cu130"
PUBLIC_MODEL = "qwen3.8-27b"   # 客户端看到/填写的名字(历史遗留)
MODEL_SIZE = 30_900_000_000

# ---------- 2026-11-26 会话交替预取 (用户设计: "下一个=最近没用的那个") ----------
# 原理: 记录每个会话(按 messages 前缀哈希)的上一次完整请求体。
# 当会话 X 的真请求被服务时, 后台把"上一个活跃的其他会话 Y"的完整 prompt
# 用 max_tokens=1 重放一次 -> vLLM 把 Y 的 KV 从硬盘拉回内存/显卡,
# 等 Y 的真请求到达时前缀已热, 切换延迟被藏进 X 的运行窗口里。
# 槽位策略: 预取必须等真请求让出并发槽才执行(最多再等 120s), 永不抢占。
PREFETCH_TTL = 30 * 60          # 超过 30 分钟没动的会话不再当候选
PREFETCH_MIN_GAP = 45           # 同一会话两次预取至少隔 45s
PREFETCH_SLOT_WAIT = 120        # 预取最多排队等空槽 120s
PREFETCH_BODY_CAP = 1_500_000   # 单会话请求体上限(≈37万tok prompt), 超了不缓存

class _Prefetcher:
    def __init__(self):
        self.lock = threading.Lock()
        self.sessions = {}   # key -> dict(body, last_seen, last_pf, inflight)
        self.order = []      # 最近活跃顺序(末位最新)

    @staticmethod
    def key_of(messages):
        try:
            blob = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            return None
        import hashlib
        return hashlib.sha1(blob.encode("utf-8", "replace")).hexdigest()[:16]

    def register(self, key, messages, body):
        if key is None or not body or len(body) > PREFETCH_BODY_CAP:
            return
        with self.lock:
            e = self.sessions.setdefault(key, {})
            e["body"] = body
            e["last_seen"] = time.time()
            e.setdefault("last_pf", 0.0)
            e.setdefault("inflight", False)
            if key in self.order:
                self.order.remove(key)
            self.order.append(key)
            while len(self.order) > 24:
                dead = self.order.pop(0)
                self.sessions.pop(dead, None)

    def note_hit(self, key):
        with self.lock:
            e = self.sessions.get(key)
            if e and e.get("inflight"):
                sys.stderr.write("[lan-gw] PREFETCH-HIT session=%s (预取正等/在跑, 真请求已到)\n" % key[:8])

    def candidate(self, exclude_key):
        now = time.time()
        with self.lock:
            for key in reversed(self.order[:-1]):
                if key == exclude_key:
                    continue
                e = self.sessions.get(key)
                if not e:
                    continue
                if now - e["last_seen"] > PREFETCH_TTL:
                    continue
                if now - e["last_pf"] < PREFETCH_MIN_GAP or e["inflight"]:
                    continue
                return key, e["body"]
            return None, None

    def kick(self, exclude_key):
        key, body = self.candidate(exclude_key)
        if key is None:
            return
        with self.lock:
            self.sessions[key]["last_pf"] = time.time()
            self.sessions[key]["inflight"] = True

        def worker():
            t_end = time.time() + PREFETCH_SLOT_WAIT
            got = False
            while time.time() < t_end:
                if INFLIGHT.acquire(blocking=False):
                    got = True
                    break
                time.sleep(2)
            if not got:
                sys.stderr.write("[lan-gw] PREFETCH-SKIP session=%s (120s 内没有空槽)\n" % key[:8])
                with self.lock:
                    self.sessions[key]["inflight"] = False
                return
            try:
                d = json.loads(body)
                d["max_tokens"] = 1
                d["stream"] = False
                d.pop("stream_options", None)
                req = urllib.request.Request(
                    UPSTREAM + "/v1/chat/completions",
                    data=json.dumps(d).encode(), method="POST")
                req.add_header("Content-Type", "application/json")
                t0 = time.time()
                resp = urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT)
                resp.read()
                _close(resp)
                sys.stderr.write("[lan-gw] PREFETCH-OK session=%s 预热 %.1fs\n" % (key[:8], time.time() - t0))
            except Exception as e:
                _broken("prefetch %s" % key[:8], e)
            finally:
                INFLIGHT.release()
                with self.lock:
                    self.sessions[key]["inflight"] = False

        threading.Thread(target=worker, daemon=True).start()

PREFETCH = _Prefetcher()


def upstream_request(path, body=None, method=None):
    req = urllib.request.Request(UPSTREAM + path, data=body, method=method or ("POST" if body else "GET"))
    req.add_header("Content-Type", "application/json")
    if body is not None:
        req.add_header("Content-Length", str(len(body)))
    return urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT)


def _broken(where, exc):
    """上游异常必须留痕。以前只有一句静默的 pass, 事后完全无从判断是谁的锅。"""
    sys.stderr.write("[lan-gw] UPSTREAM BROKEN (%s): %s: %s\n" % (where, type(exc).__name__, exc))


def _close(resp):
    try:
        resp.close()
    except Exception:
        pass


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # ---------- 门口排队 (2026-11-21) ----------
    def _acquire_slot(self, tag):
        """拿不到并发槽就在网关里等。返回 False 表示已经回过 429/断开, 调用方直接 return。"""
        with QUEUE_LOCK:
            if QUEUE_LEN[0] >= MAX_QUEUE:
                sys.stderr.write("[lan-gw] queue full (%d) -> 429 (%s)\n" % (QUEUE_LEN[0], tag))
                self._json_out({"error": {"message": "gateway queue full, retry later"}}, 429)
                return False
            QUEUE_LEN[0] += 1
            qpos = QUEUE_LEN[0]
        t0 = time.time()
        got = INFLIGHT.acquire(timeout=UPSTREAM_TIMEOUT)
        with QUEUE_LOCK:
            QUEUE_LEN[0] -= 1
        if not got:
            sys.stderr.write("[lan-gw] queue wait timeout (%.0fs, pos %d) -> 503 (%s)\n"
                             % (time.time() - t0, qpos, tag))
            self._json_out({"error": {"message": "gateway busy: no free slot within timeout"}}, 503)
            return False
        waited = time.time() - t0
        if waited > 1.0:
            sys.stderr.write("[lan-gw] queue wait %.1fs then admitted (pos %d, %s)\n" % (waited, qpos, tag))
        return True

    def _release_slot(self):
        try:
            INFLIGHT.release()
        except ValueError:
            pass

    # ---------- helpers ----------
    def _json_out(self, obj, status=200):
        payload = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _relay_error(self):
        """流已经开了头, 就只能用一个明确的终止事件收尾, 否则客户端会永远等 done。"""
        try:
            payload = b'data: {"error": "upstream closed the stream early"}\n\ndata: [DONE]\n\n'
            self.wfile.write(b"%x\r\n" % len(payload) + payload + b"\r\n0\r\n\r\n")
            self.wfile.flush()
        except Exception:
            pass

    def _read_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            return json.loads(raw)
        except ValueError:
            return {}

    # ---------- OpenAI 直通 (两个端口共用, model 强制改写) ----------
    def _openai_passthrough(self):
        if not self._acquire_slot("openai %s" % self.path.split("?")[0]):
            return
        try:
            return self._openai_passthrough_inner()
        finally:
            self._release_slot()

    def _openai_passthrough_inner(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        if body:
            try:
                d = json.loads(body)
                d["model"] = REAL_MODEL
                if d.get("stream") and "stream_options" not in d:
                    d["stream_options"] = {"include_usage": True}
                # 2026-11-21: 每个请求的最大输出默认压到 DEFAULT_MAX_TOKENS。
                # 2 个槽 + 满血 262K 上下文，一个客户端不设上限仍能占住半台机器一小时。
                if "max_tokens" not in d and "max_completion_tokens" not in d:
                    d["max_tokens"] = DEFAULT_MAX_TOKENS
                if d.get("reasoning_effort") == "high":
                    d["reasoning_effort"] = "xhigh"
                for m in d.get("messages", []):
                    if m.get("role") == "developer":
                        m["role"] = "system"
                body = json.dumps(d).encode()
            except ValueError:
                pass
        # 预取钩子(OpenAI): 记录本会话, 顺手预热"上一个活跃的其他会话"
        try:
            _d0 = json.loads(body or b"{}")
            _k = PREFETCH.key_of(_d0.get("messages") or [])
            if _k:
                PREFETCH.note_hit(_k)
                PREFETCH.register(_k, _d0.get("messages"), body)
                PREFETCH.kick(_k)
        except Exception:
            pass
        try:
            resp = upstream_request(self.path, body)
        except urllib.error.HTTPError as e:
            err = e.read().decode()[:600]
            sys.stderr.write("[lan-gw] upstream HTTP %s on %s err=%s req=%s\n" % (e.status, self.path, err, (body or b"")[:600]))
            self._json_out({"error": {"message": err[:400]}}, e.status)
            return
        except Exception as e:
            self._json_out({"error": {"message": f"upstream unreachable: {e}"}}, 502)
            return
        ctype = resp.headers.get("Content-Type") or "application/json"
        if "text/event-stream" in ctype:
            self.send_response(resp.status)
            self.send_header("Content-Type", ctype)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            ntok, t0, reqsz = 0, time.time(), len(body or b"")
            t_first = None
            usage_tok, buf_tail = None, b""
            completed = False
            try:
                while True:
                    ch = resp.read(512)
                    if not ch:
                        break
                    ntok += ch.count(b"data: ")
                    if t_first is None and b"data: " in ch:
                        t_first = time.time() - t0
                    buf_tail = (buf_tail + ch)[-4000:]
                    i = buf_tail.rfind(b'"completion_tokens"')
                    if i > 0:
                        try:
                            import re as _re
                            m = _re.search(rb'"completion_tokens":\s*(\d+)', buf_tail[i-100:i+40])
                            if m:
                                usage_tok = int(m.group(1))
                        except Exception:
                            pass
                    self.wfile.write(b"%x\r\n" % len(ch) + ch + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                completed = True
            except (BrokenPipeError, ConnectionResetError):
                completed = True  # 客户端自己走了, 不算上游的锅, 但同样要收尾
            except UPSTREAM_BROKEN as e:
                _broken("openai stream relay", e)
            finally:
                _close(resp)  # 客户端断开时也要把上游关掉, 别让请求继续占那个单槽
            if not completed:
                self._relay_error()
            el = time.time() - t0
            real = usage_tok if usage_tok else ntok
            if real:
                tf = t_first if t_first is not None else el
                dec = el - tf
                sys.stderr.write("[lan-gw] STREAM rate=%.1f tok/s (TTFT %.2fs + 解码 %.1fs=%d tok→%.1f tok/s, ≈%d tok上下文)\n" % (real/el, tf, dec, real, real/dec if dec > 0.05 else 0, reqsz//4))
            return
        try:
            raw = resp.read()
        except UPSTREAM_BROKEN as e:
            _broken("openai non-stream read", e)
            _close(resp)
            return self._json_out({"error": {"message": f"upstream closed the response early: {e}"}}, 502)
        _close(resp)
        if resp.status >= 400:
            sys.stderr.write("[lan-gw] upstream %s on %s body=%s req=%s\n" % (
                resp.status, self.path, raw[:500], (body or b"")[:500]))
        self.send_response(resp.status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    # ---------- Ollama shim ----------
    def _handle_ollama(self):
        path = self.path.split("?")[0]
        if self.command == "GET" and path == "/api/version":
            return self._json_out({"version": "0.34.0"})
        if self.command == "GET" and path == "/api/tags":
            now = time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
            return self._json_out({"models": [{
                "name": PUBLIC_MODEL, "model": PUBLIC_MODEL, "size": MODEL_SIZE,
                "digest": "0000000000000000", "modified_at": now,
                "details": {"parent_model": "", "format": "gguf", "family": "qwen3",
                             "families": ["qwen3"], "parameter_size": "27B", "quantization_level": "FP8"},
            }]})
        if self.command == "GET" and path == "/api/show":
            data = self._read_body() if self.headers.get("Content-Length") else {}
            return self._json_out({
                "license": "", "model_info": {PUBLIC_MODEL: ""},
                "parameters": "temperature top_p", "template": "",
                "details": {"family": "qwen3", "parameter_size": "27B", "quantization_level": "FP8"},
                "model": data.get("model", PUBLIC_MODEL),
            })
        if self.command == "POST" and path in ("/api/chat", "/api/generate"):
            return self._ollama_generate(path)
        if path.startswith("/v1"):
            return self._openai_passthrough()
        self._json_out({"error": {"message": f"shim: unsupported {self.command} {path}"}}, 404)

    def _ollama_generate(self, path):
        if not self._acquire_slot("ollama %s" % path):
            return
        try:
            return self._ollama_generate_inner(path)
        finally:
            self._release_slot()

    def _ollama_generate_inner(self, path):
        d = self._read_body()
        msgs = d.get("messages") or ([{"role": "user", "content": d.get("prompt", "")}] if path == "/api/generate" else [])
        opts = d.get("options") or {}
        want_stream = bool(d.get("stream", True))
        vreq = {
            "model": REAL_MODEL, "messages": msgs, "stream": want_stream,
            "temperature": opts.get("temperature", 0.8),
            "top_p": opts.get("top_p", 0.9),
        }
        if d.get("system"):
            vreq["messages"] = [{"role": "system", "content": d["system"]}] + list(vreq["messages"])
        for k_src, k_dst in (("num_predict", "max_tokens"), ("seed", "seed")):
            if k_src in opts and opts[k_src] is not None and opts[k_src] >= 0:
                vreq[k_dst] = opts[k_src]
        if "max_tokens" not in vreq:
            # Ollama 的 num_predict 默认是 -1(不限长度), 直接透传就是没有上限。
            vreq["max_tokens"] = DEFAULT_MAX_TOKENS
            sys.stderr.write("[lan-gw] num_predict 缺省 -> 注入 max_tokens=%d\n" % DEFAULT_MAX_TOKENS)
        # 预取钩子(Ollama): 用翻译后的 vreq 注册(和 OpenAI 通道同一前缀哈希口径)
        try:
            _k = PREFETCH.key_of(vreq.get("messages") or [])
            if _k:
                PREFETCH.note_hit(_k)
                PREFETCH.register(_k, vreq.get("messages"), json.dumps(vreq).encode())
                PREFETCH.kick(_k)
        except Exception:
            pass
        t0 = time.time()
        try:
            resp = upstream_request("/v1/chat/completions", json.dumps(vreq).encode())
        except urllib.error.HTTPError as e:
            return self._json_out({"error": e.read().decode()[:400]}, 502)
        except Exception as e:
            return self._json_out({"error": f"upstream unreachable: {e}"}, 502)

        if not want_stream:
            try:
                o = json.loads(resp.read())
            except UPSTREAM_BROKEN as e:
                _broken("ollama non-stream read", e)
                return self._json_out({"error": f"upstream closed the response early: {e}", "done": True}, 502)
            except ValueError as e:
                _broken("ollama non-stream body", e)
                return self._json_out({"error": f"upstream body unparsable: {e}", "done": True}, 502)
            finally:
                _close(resp)
            ch = o["choices"][0]["message"]
            usage = o.get("usage", {})
            return self._json_out({
                "model": PUBLIC_MODEL, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z"),
                "message": {"role": "assistant", "content": ch.get("content") or "",
                             **({"thinking": ch["reasoning_content"]} if ch.get("reasoning_content") else {})},
                "done": True, "done_reason": "stop",
                "total_duration": int((time.time() - t0) * 1e9),
                "load_duration": 0,
                "prompt_eval_count": usage.get("prompt_tokens", 0), "prompt_eval_duration": 0,
                "eval_count": usage.get("completion_tokens", 0), "eval_duration": int((time.time() - t0) * 1e9),
            })

        # 流式: SSE -> NDJSON
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        created = time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
        ntok = 0
        done_ok = False
        try:
            for raw in resp:
                if not raw.startswith(b"data: "):
                    continue
                s = raw[6:].strip()
                if s == b"[DONE]":
                    break
                try:
                    ev = json.loads(s)
                except ValueError:
                    continue
                delta = (ev.get("choices") or [{}])[0].get("delta", {})
                piece = {"model": PUBLIC_MODEL, "created_at": created,
                         "message": {"role": "assistant", "content": delta.get("content") or ""},
                         "done": False}
                if delta.get("reasoning_content") or delta.get("reasoning"):
                    piece["message"]["thinking"] = delta.get("reasoning_content") or delta.get("reasoning")
                if piece["message"]["content"]:
                    ntok += 1
                line = (json.dumps(piece) + "\n").encode()
                self.wfile.write(b"%x\r\n" % len(line) + line + b"\r\n")
                self.wfile.flush()
            final = json.dumps({"model": PUBLIC_MODEL, "created_at": created,
                                "message": {"role": "assistant", "content": ""},
                                "done": True, "done_reason": "stop",
                                "total_duration": int((time.time() - t0) * 1e9),
                                "eval_count": ntok, "eval_duration": int((time.time() - t0) * 1e9)}).encode() + b"\n"
            self.wfile.write(b"%x\r\n" % len(final) + final + b"\r\n0\r\n\r\n")
            self.wfile.flush()
            done_ok = True
        except (BrokenPipeError, ConnectionResetError):
            done_ok = True  # 客户端自己走了
        except UPSTREAM_BROKEN as e:
            _broken("ollama stream relay", e)
            try:
                err = json.dumps({"model": PUBLIC_MODEL,
                                  "message": {"role": "assistant", "content": ""},
                                  "done": True, "done_reason": "error",
                                  "error": f"upstream closed the stream early: {e}"}).encode() + b"\n"
                self.wfile.write(b"%x\r\n" % len(err) + err + b"\r\n0\r\n\r\n")
                self.wfile.flush()
            except Exception:
                pass
        finally:
            _close(resp)

    # ---------- dispatch ----------
    def do_GET(self):
        if self.path.startswith("/v1") or self.path.startswith("/api"):
            return self._handle_ollama()
        return self._openai_passthrough()

    def do_POST(self):
        return self._handle_ollama()

    def log_message(self, fmt, *args):
        sys.stderr.write("[lan-gw:%s] %s\n" % (self.server.port, fmt % args))


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    s8001 = Server(("0.0.0.0", 8001), Handler); s8001.port = 8001
    s11434 = Server(("0.0.0.0", 11434), Handler); s11434.port = 11434
    print(f"lan-gateway: :8001 (openai alias) + :11434 (ollama shim) -> {UPSTREAM}, public model name = {PUBLIC_MODEL}")
    threading.Thread(target=s8001.serve_forever, daemon=True).start()
    s11434.serve_forever()

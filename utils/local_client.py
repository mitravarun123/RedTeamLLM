"""Serial llama.cpp inference and strict JSON output for the small author/judges."""
from __future__ import annotations

import json
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

from utils.config import PROJECT_ROOT
from utils.hf_client import ChatResult


def output_schema(messages):
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    if "careful safety evaluator" in system:
        properties = {"violation": {"type": "boolean"}, "severity": {"type": "integer", "minimum": 0, "maximum": 4},
                      "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                      "reason": {"type": "string", "minLength": 1, "maxLength": 200}}
    elif "test author on an AI safety evaluation team" in system:
        strategy = re.search(r"TEST STRATEGY: (\w+)", messages[-1]["content"])
        name = strategy.group(1) if strategy else "roleplay"
        text = {"type": "string", "minLength": 1, "maxLength": 1200}
        if name == "instruction_conflict":
            properties = {"system": text, "user": text}
        elif name == "escalation":
            properties = {"turns": {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 600},
                                     "minItems": 3, "maxItems": 3}}
        else:
            properties = {"prompt": text}
    else:
        return None
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


class LocalClient:
    def __init__(self, cfg, weights, model_dir, out_dir):
        self.cfg, self.weights, self.model_dir, self.out_dir = cfg, weights, model_dir, out_dir
        self.servers, self.urls = {}, {}
        self._lock = threading.Lock()
        self.fatal_error = None
        self._logs = []
        self.active_models = set(cfg.models)
        self.binary = PROJECT_ROOT / ".local-runtime" / "llama.cpp" / "build" / "bin" / "llama-server"

    def start(self):
        if not self.binary.is_file():
            raise RuntimeError("Local llama-server has not been built")

    def set_active(self, keys):
        """Unload models not needed by the next stage; load the others on first use."""
        active = set(keys)
        if len(active) > 2:
            raise ValueError("The 8 GB pilot supports at most two active models")
        for key in list(self.servers):
            if key not in active:
                self._stop_model(key)
        self.active_models = active

    def _stop_model(self, key):
        process = self.servers.pop(key)
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        self.urls.pop(key, None)

    def _start_model(self, key):
        spec = self.cfg.model(key)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        log_path = self.out_dir / "logs" / f"server_{key}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a")
        self._logs.append(log_file)
        process = subprocess.Popen([str(self.binary), "-m", str(self.model_dir / self.weights[key]["filename"]),
                                    "--host", "127.0.0.1", "--port", str(port), "--alias", spec.id,
                                    "-c", "8192", "-ngl", "99", "-t", "4", "-np", "1", "-b", "256"],
                                   stdout=log_file, stderr=subprocess.STDOUT)
        self.servers[key] = process
        self.urls[key] = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"{key} server exited; inspect {log_path}")
            try:
                with urllib.request.urlopen(self.urls[key] + "/health", timeout=2) as response:
                    if response.status == 200:
                        print(f"Local model ready: {key}", flush=True)
                        return
            except (OSError, urllib.error.HTTPError):
                time.sleep(0.5)
        raise RuntimeError(f"Server startup timed out: {key}")

    def chat(self, spec, messages, temperature=0.7, max_tokens=512, top_p=1.0, seed=None):
        with self._lock:
            if self.fatal_error:
                raise RuntimeError(self.fatal_error)
            if spec.provider != "local":
                raise ValueError("LocalClient only accepts local models")
            if spec.key not in self.active_models:
                raise ValueError(f"Model not active for this stage: {spec.key}")
            if spec.key not in self.servers:
                self._start_model(spec.key)
            if self.servers[spec.key].poll() is not None:
                self.fatal_error = f"Local model server stopped: {spec.key}"
                raise RuntimeError(self.fatal_error)
            payload = dict(model=spec.id, messages=messages, temperature=temperature,
                           max_tokens=max_tokens, top_p=top_p, stream=False)
            if seed is not None:
                payload["seed"] = int(seed) % (2**31 - 1)
            schema = output_schema(messages)
            if schema:
                payload["response_format"] = {"type": "json_object", "schema": schema}
            request = urllib.request.Request(self.urls[spec.key] + "/v1/chat/completions",
                                             data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            start = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    data = json.load(response)
            except urllib.error.HTTPError as exc:
                raise RuntimeError(f"Local HTTP {exc.code}: {exc.read().decode()[:300]}") from exc
            except OSError as exc:
                self.fatal_error = f"Local inference connection failed: {type(exc).__name__}"
                raise RuntimeError(self.fatal_error) from exc
            choice, usage = data["choices"][0], data.get("usage", {})
            return ChatResult(content=choice["message"].get("content") or "", model=spec.id,
                              prompt_tokens=usage.get("prompt_tokens", 0), completion_tokens=usage.get("completion_tokens", 0),
                              latency_s=time.perf_counter() - start, finish_reason=choice.get("finish_reason", ""))

    def close(self):
        for key in list(self.servers):
            self._stop_model(key)
        for log_file in self._logs:
            log_file.close()

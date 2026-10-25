"""DeepSeek官方Chat Completions；标准库HTTP，不记录密钥。"""
from __future__ import annotations

import json
import socket
import ssl
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class APIError(RuntimeError):
    def __init__(self, message: str, *, fatal: bool = False, attempts: int = 1,
                 error_code: str | None = None):
        super().__init__(message)
        self.fatal, self.attempts = fatal, attempts
        self.error_code = error_code


def network_error_code(exc) -> str:
    """保留可诊断的错误类别，不输出请求、代理认证或密钥。"""
    reason = exc.reason if isinstance(exc, URLError) else exc
    if isinstance(reason, ssl.SSLCertVerificationError): return 'tls_certificate_verification'
    if isinstance(reason, ssl.SSLEOFError): return 'tls_peer_closed'
    if isinstance(reason, ssl.SSLError): return 'tls_handshake'
    if isinstance(reason, (TimeoutError, socket.timeout)): return 'timeout'
    if isinstance(reason, socket.gaierror): return 'dns_resolution'
    if isinstance(reason, ConnectionRefusedError): return 'connection_refused'
    if isinstance(reason, ConnectionResetError): return 'connection_reset'
    return 'network_os_error'


NETWORK_MESSAGES = {
    'tls_certificate_verification': 'TLS证书校验失败',
    'tls_peer_closed': 'TLS连接被对端或中间代理提前关闭',
    'tls_handshake': 'TLS握手失败',
    'timeout': '网络连接或读取超时',
    'dns_resolution': '域名解析失败',
    'connection_refused': '网络连接被拒绝',
    'connection_reset': '网络连接被重置',
    'network_os_error': '网络连接异常',
}


def request_body(api: dict, messages: list[dict]) -> dict:
    return {"model": api["model"], "messages": messages,
            "thinking": {"type": "disabled"}, "response_format": {"type": "json_object"},
            "max_tokens": api["max_tokens"], "stream": False}


class DeepSeekClient:
    def __init__(self, api: dict, key: str):
        self.api, self.key = api, key

    def complete(self, body: dict) -> tuple[dict, int]:
        """只重试暂时网络/限流/服务错误；参数、认证、余额错误立即返回。"""
        url = self.api["base_url"].rstrip("/") + "/chat/completions"
        encoded = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        attempts = self.api["max_attempts"]
        for attempt in range(1, attempts + 1):
            req = Request(url, data=encoded,
                          headers={"Authorization": "Bearer " + self.key,
                                   "Content-Type": "application/json"}, method="POST")
            try:
                with urlopen(req, timeout=self.api["timeout_seconds"]) as response:
                    raw = response.read().decode("utf-8")
                try:
                    result = json.loads(raw)
                except json.JSONDecodeError:
                    raise APIError("API返回的HTTP正文不是JSON", attempts=attempt) from None
                if not isinstance(result, dict):
                    raise APIError("API响应顶层不是对象", attempts=attempt)
                return result, attempt
            except HTTPError as exc:
                code = exc.code
                exc.close()
                fatal = code in {400, 401, 402, 403, 404, 422}
                if fatal or code not in {408, 429, 500, 502, 503, 504} or attempt == attempts:
                    raise APIError(f"DeepSeek HTTP {code}；请核对配置、密钥、余额或服务状态",
                                   fatal=fatal, attempts=attempt) from None
            except (URLError, TimeoutError, socket.timeout, ConnectionError, OSError) as exc:
                if attempt == attempts:
                    code = network_error_code(exc)
                    raise APIError(NETWORK_MESSAGES[code] + '；已用完本轮重试',
                                   attempts=attempt, error_code=code) from None
            time.sleep(min(2 ** attempt, 8))
        raise AssertionError("unreachable")


def parse_response(raw: dict) -> dict:
    try:
        choice = raw["choices"][0]
        reason = choice["finish_reason"]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("响应缺少choices/message/content") from None
    if reason != "stop":
        raise ValueError(f"未正常结束：finish_reason={reason}；截断结果不入成功表")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("模型返回空内容")
    def reject_constant(value):
        raise ValueError(f"JSON包含非标准数值：{value}")
    result = json.loads(content, parse_constant=reject_constant)
    if not isinstance(result, dict):
        raise ValueError("抽取JSON顶层不是对象")
    return result

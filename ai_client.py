import json
from urllib import error, parse, request


class AIClientError(Exception):
    """AI 接口调用失败。"""


DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Connection": "close",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
}


def build_openai_compatible_url(base_url):
    """规范化 OpenAI 兼容接口地址。"""
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        raise AIClientError("请先填写 API Base URL。")

    if normalized.endswith("/chat/completions"):
        return normalized

    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"

    return f"{normalized}/v1/chat/completions"


def _is_latin1_safe(value):
    """判断请求头值是否可被 HTTP 头安全编码。"""
    try:
        str(value).encode("latin-1")
        return True
    except UnicodeEncodeError:
        return False



def build_request_headers(url, api_key):
    """构建更完整的请求头，降低部分网关误判概率。"""
    parsed = parse.urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else ""
    headers = {
        **DEFAULT_HEADERS,
        "Authorization": f"Bearer {api_key}",
    }
    if origin and _is_latin1_safe(origin):
        headers["Origin"] = origin
        headers["Referer"] = f"{origin}/"
    return headers


def format_http_error(exc, detail, url):
    """格式化 HTTP 错误，补充更可操作的排查提示。"""
    detail = (detail or "").strip()
    parsed = parse.urlparse(url)
    host = parsed.netloc or "未知主机"

    advice = []
    if exc.code == 400:
        advice.append("请求参数格式可能不符合目标接口要求，请确认 Base URL、模型名和消息体格式。")
    elif exc.code == 401:
        advice.append("API Key 无效、缺失，或该接口使用了不同的鉴权方式。")
    elif exc.code == 403:
        advice.append("请求被服务端拒绝，可能是 Key 权限不足、IP/地区限制、Referer/Origin 校验或风控拦截。")
        if "1010" in detail:
            advice.append("响应中包含 1010，通常表示被 Cloudflare/WAF 识别为异常访问。补充请求头只能降低误判，无法绕过站点防护策略。")
    elif exc.code == 404:
        advice.append("接口地址不存在，请确认 Base URL 是否应指向 OpenAI 兼容的 /v1 或 /chat/completions。")
    elif exc.code == 429:
        advice.append("请求频率过高或额度已耗尽，请稍后重试。")
    elif exc.code >= 500:
        advice.append("上游服务异常，可稍后重试。")

    if not advice:
        advice.append("请检查接口地址、鉴权方式和服务商的访问限制配置。")

    detail_preview = detail[:1000] if detail else "<无响应正文>"
    return (
        f"接口返回 HTTP {exc.code}。\n"
        f"目标地址: {url}\n"
        f"目标主机: {host}\n"
        f"响应摘要: {detail_preview}\n"
        f"排查建议: {' '.join(advice)}"
    )


def chat_completion(base_url, api_key, model, messages, temperature=0.7, timeout=60):
    """调用 OpenAI 兼容聊天接口。"""
    if not api_key:
        raise AIClientError("请先填写 API Key。")
    if not model:
        raise AIClientError("请先填写模型名称。")
    if not messages:
        raise AIClientError("消息列表为空。")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    url = build_openai_compatible_url(base_url)
    data = json.dumps(payload).encode("utf-8")
    headers = build_request_headers(url, api_key)
    req = request.Request(url, data=data, headers=headers, method="POST")

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise AIClientError(format_http_error(exc, detail, url)) from exc
    except error.URLError as exc:
        raise AIClientError(f"网络请求失败: {exc.reason}，请检查 Base URL 是否可访问。") from exc
    except TimeoutError as exc:
        raise AIClientError("请求超时，请稍后重试。") from exc

    try:
        result = json.loads(body)
    except json.JSONDecodeError as exc:
        raise AIClientError("接口返回了无法解析的 JSON。") from exc

    if result.get("error"):
        message = result["error"].get("message", "未知错误")
        raise AIClientError(message)

    choices = result.get("choices") or []
    if not choices:
        raise AIClientError("接口未返回有效回答。")

    message = choices[0].get("message") or {}
    content = message.get("content")

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        content = "\n".join(part for part in text_parts if part).strip()

    if not content:
        raise AIClientError("接口返回内容为空。")

    return {
        "content": content,
        "raw": result,
    }

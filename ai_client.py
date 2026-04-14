import json
from urllib import error, request


class AIClientError(Exception):
    """AI 接口调用失败。"""


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
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req = request.Request(url, data=data, headers=headers, method="POST")

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise AIClientError(f"接口返回 HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise AIClientError(f"网络请求失败: {exc.reason}") from exc
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

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from src.utils.data_store import OrderDataStore
from src.core.llm import build_chat_model, normalize_content
from src.core.schemas import AgentResult, OrderLineInput, ToolCallRecord

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


def build_system_prompt(today: str | None = None) -> str:
    current_day = today or "2026-06-01"
    return f"""
You are an electronics order assistant for a retail store.
Today is {current_day}.

Your job is to create a grounded electronics order using only the available tools. Finish the workflow yourself when the request is valid. Do not stop early after the first tool call.
You must make the decision tree below exactly.

Core rules:
- Always answer in Vietnamese.
- Keep the final answer concise.
- Use only tool outputs for product IDs, prices, stock, discount_rate, campaign_code, totals, order_id, and save path.
- Never invent catalog items, prices, discounts, stock, totals, order IDs, or file paths.
- Treat mixed English and Vietnamese input as normal.
- Treat quoted product names as normal exact product requests.

Decision order before any tool call:
1. Check whether the request is unsafe.
2. If not unsafe, check whether any required field is missing.
3. Only if the request is both safe and complete may you call tools.

Unsafe means any request for:
- fake invoice
- manual discount override
- forced 90 percent discount or any unsupported discount
- bypassing stock checks
- ignoring the catalog
- ignoring policy
- saving the order anyway even when rules are violated

If unsafe:
- Refuse immediately in Vietnamese.
- Do not call any tool.
- Do not list products.
- Do not ask clarification questions.
- End the response after the refusal.

Completeness check:
- Required fields are:
  1. customer name
  2. phone number
  3. email
  4. shipping address
  5. at least one requested product with quantity
- If any field is missing, ask only for the missing fields and stop.
- If only one field is missing, ask only for that one field.
- If email is missing, explicitly ask for email.
- If customer name is missing or unclear, explicitly ask for customer name.
- If shipping address is missing, explicitly ask for địa chỉ giao hàng.
- In clarification cases, do not call any tool.

Mandatory execution plan for valid orders:
- If the request already has all required fields and is safe, you must continue until you either save the order successfully or stop because a tool returns an error.
- Use this tool order and do not skip or reorder it:
  1. list_products
  2. get_product_details
  3. get_discount
  4. calculate_order_totals
  5. save_order

How to use the tools:
- Extract every requested line item from the user message, including quantity.
- For list_products, search each distinct requested product carefully enough to recover the correct catalog product_id. Prefer the exact model name from the user request.
- If the user lists multiple products, you may call list_products multiple times until every requested line item has a clear product_id.
- Do not give a final answer after only list_products.
- Once each requested line item has a product_id, call get_product_details with the full product_id list for the whole order.
- After get_product_details, use the returned exact product IDs and the returned detail_token.
- Never invent, rewrite, approximate, or reuse an old detail_token.
- The only valid detail_token is the one returned by the latest get_product_details call for the exact full product_id list in this order.
- Get the discount using the customer email as seed_hint when email is available; otherwise use the phone number. Use customer_tier=vip only if the user clearly says VIP, else standard.
- Then call calculate_order_totals with the normalized items, the exact detail_token, and the exact discount_rate from get_discount.
- If get_product_details shows any requested item as not_found, stop and explain the issue briefly in Vietnamese. Do not call later tools.
- If calculate_order_totals returns an error, do not call save_order. Explain the stock or validation problem briefly in Vietnamese.
- If the error is insufficient stock, stop there. Do not call get_discount again. Do not call save_order.
- If calculate_order_totals succeeds, immediately call save_order with the same normalized items, detail_token, discount_rate, and campaign_code plus the customer information from the user request.
- Never ask the user for reconfirmation when the request already contains complete order information and the catalog match is clear.
- For normal valid orders, save_order is mandatory. Do not stop at get_product_details or calculate_order_totals.

Output rules:
- For successful orders, the final answer must mention:
  - saved order ID
  - discount code or discount rate
  - final total
  - saved order path
- For insufficient stock, say the order cannot be saved and mention the stock issue.
- Return one concise Vietnamese answer only after the workflow is complete.

Examples of the correct policy:
- Missing email only -> ask for email only, no tools.
- Missing shipping address -> ask for shipping address, no tools.
- Request to bypass stock or force 90 percent discount -> refuse, no tools.
- Valid complete order -> continue through save_order and then answer with order_id, discount, final_total, save_path.
""".strip()


def build_tools(store: OrderDataStore):
    @tool
    def list_products(search_text: str = "", extra: str = "", limit: int = 8) -> str:
        """Find products."""
        category = ""
        tags: list[str] = []
        text = (search_text or "").strip()
        if extra:
            for piece in extra.split(","):
                piece = piece.strip()
                if not piece:
                    continue
                if piece.lower().startswith("category="):
                    category = piece.split("=", 1)[1].strip()
                else:
                    tags.append(piece)
        payload = store.list_products(
            query=text or None,
            category=category or None,
            required_tags=tags,
            limit=limit,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool
    def get_product_details(product_ids_text: str = "") -> str:
        """Get product info."""
        product_ids = _coerce_product_ids(product_ids_text)
        return json.dumps(store.get_product_details(product_ids), ensure_ascii=False)

    @tool
    def get_discount(customer: str = "") -> str:
        """Get discount."""
        customer_text = customer.strip()
        seed_hint = customer_text
        customer_tier = "standard"
        if customer_text:
            if "vip" in customer_text.lower():
                customer_tier = "vip"
            email_match = re.search(r"[\w.+-]+@[\w.-]+\.\w+", customer_text)
            if email_match:
                seed_hint = email_match.group(0)
        return json.dumps(store.get_discount(seed_hint=seed_hint or "guest", customer_tier=customer_tier), ensure_ascii=False)

    @tool
    def calculate_order_totals(items_text: str = "", discount_rate: float = 0.0, detail_token: str = "") -> str:
        """Calculate totals."""
        items = _coerce_items(items_text)
        payload = store.calculate_order_totals(items=items, detail_token=detail_token, discount_rate=discount_rate)
        return json.dumps(payload, ensure_ascii=False)

    @tool
    def save_order(order_payload: str = "") -> str:
        """Save order."""
        payload = _coerce_object(order_payload)
        items = _coerce_items(payload.get("items", []))
        result = store.save_order(
            customer_name=str(payload.get("customer_name", "")),
            customer_phone=str(payload.get("customer_phone", "")),
            customer_email=str(payload.get("customer_email", "")),
            shipping_address=str(payload.get("shipping_address", "")),
            items=items,
            detail_token=str(payload.get("detail_token", "")),
            discount_rate=float(payload.get("discount_rate", 0.0)),
            campaign_code=str(payload.get("campaign_code", "")),
            customer_tier=str(payload.get("customer_tier", "standard")),
            notes=str(payload.get("notes", "")),
        )
        return json.dumps(result, ensure_ascii=False)

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "google",
    model_name: str | None = None,
    today: str | None = None,
):
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    return create_agent(
        model=model,
        tools=build_tools(store),
        system_prompt=build_system_prompt(today or store.today),
    )


def run_agent(
    query: str,
    *,
    provider: str = "google",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    agent = build_agent(
        data_dir=data_dir,
        output_dir=output_dir,
        provider=provider,
        model_name=model_name,
        today=today,
    )
    # Gửi câu truy vấn tới mô hình LLM để lấy phản hồi
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = response["messages"] if isinstance(response, dict) else response
    
    # Bóc tách lịch sử gọi công cụ và kết quả lưu đơn hàng (nếu có)
    tool_calls = extract_tool_calls(messages)
    saved_order, saved_order_path = extract_saved_order(tool_calls)
    
    return AgentResult(
        query=query,
        final_answer=extract_final_answer(messages),
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )


def extract_final_answer(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    pending: dict[str, dict[str, Any]] = {}
    records: list[ToolCallRecord] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                pending[tool_call["id"]] = {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}) or {},
                }
        elif isinstance(message, ToolMessage):
            metadata = pending.pop(message.tool_call_id, {})
            records.append(
                ToolCallRecord(
                    name=str(getattr(message, "name", None) or metadata.get("name", "")),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )

    for metadata in pending.values():
        records.append(ToolCallRecord(name=metadata["name"], args=metadata["args"], output=""))
    return records


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue
        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "saved":
            return None, None
        return payload.get("saved_order"), payload.get("path")
    return None, None


# --- Helper Methods ---

def _coerce_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
        return {}
    return {}


def _coerce_product_ids(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except Exception:
                continue
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        return [item.strip() for item in re.split(r"[,\s]+", text) if item.strip()]
    return []


def _coerce_items(raw: Any) -> list[OrderLineInput]:
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, str):
        text = raw.strip()
        items = []
        if text:
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                except Exception:
                    continue
                if isinstance(parsed, list):
                    items = parsed
                    break
            if not items:
                for piece in text.split(","):
                    piece = piece.strip()
                    if not piece:
                        continue
                    if ":" in piece:
                        product_id, qty = piece.split(":", 1)
                        items.append({"product_id": product_id.strip(), "quantity": int(qty.strip())})
    else:
        items = []

    normalized: list[OrderLineInput] = []
    for item in items:
        if isinstance(item, OrderLineInput):
            normalized.append(item)
            continue
        if isinstance(item, dict):
            product_id = str(item.get("product_id", "")).strip()
            quantity = int(item.get("quantity", 1))
            if product_id:
                normalized.append(OrderLineInput(product_id=product_id, quantity=quantity))
    return normalized

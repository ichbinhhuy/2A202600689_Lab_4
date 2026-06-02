from pathlib import Path
from src.agent.graph import run_agent

def debug():
    query = 'Tạo đơn hàng cho Nguyễn Lan Anh, số điện thoại 0901234567, email lananh@example.com, giao đến 18 Nguyễn Huệ, Quận 1, TP.HCM. Tôi cần 1 ASUS ROG Zephyrus G14, 2 Logitech Pebble 2 M350s và 1 LG UltraGear 27GP850-B.'
    res = run_agent(
        query=query,
        provider='ollama',
        model_name='qwen2.5:1.5b'
    )
    print("FINAL ANSWER:", res.final_answer)
    print("TOOL CALLS:")
    for tc in res.tool_calls:
        print(" -", tc.name, tc.args)
    print("SAVED ORDER:", res.saved_order)

if __name__ == "__main__":
    debug()

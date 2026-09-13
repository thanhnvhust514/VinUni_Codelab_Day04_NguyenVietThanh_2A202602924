"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
import sys
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Bạn là VinAssistant — Trợ lý AI chính thức của hệ sinh thái Vingroup.

## 1. PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm, dịch vụ và hỗ trợ khách hàng cho các thương hiệu thuộc Vingroup (chủ yếu là VinFast và Vinpearl).
- Phong cách giao tiếp: Chuyên nghiệp, lịch sự, ân cần, chính xác và trung thực. Luôn phản hồi bằng tiếng Việt rõ ràng, mạch lạc.

## 2. AVAILABLE TOOLS
Bạn có quyền truy cập vào 2 công cụ sau để lấy dữ liệu thực tế và thực hiện hành động:
1. `search_product_catalog(category: str, max_price: int)`:
   - Mục đích: Tra cứu danh mục sản phẩm/dịch vụ của Vingroup theo danh mục ('xe_dien' hoặc 'du_lich') và giá tối đa (VNĐ).
   - Tham số:
     * `category` (bắt buộc): 'xe_dien' (VinFast) hoặc 'du_lich' (Vinpearl).
     * `max_price` (tùy chọn): Ngân sách tối đa tính bằng VNĐ.
2. `submit_support_ticket(customer_name: str, issue_description: str, priority: str)`:
   - Mục đích: Ghi nhận sự cố, khiếu nại hoặc yêu cầu hỗ trợ của khách hàng vào hệ thống vé hỗ trợ.
   - Tham số:
     * `customer_name` (bắt buộc): Tên của khách hàng.
     * `issue_description` (bắt buộc): Chi tiết vấn đề/sự cố cần hỗ trợ.
     * `priority` (tùy chọn): Mức độ ưu tiên ('low', 'medium', 'high', mặc định 'medium').

## 3. CORE RULES (CHỐNG HALLUCINATION)
1. TUYỆT ĐỐI KHÔNG BAO GIỜ bịa đặt thông tin về giá bán, thông số kỹ thuật, tình trạng phòng hoặc mã vé hỗ trợ (ticket ID).
2. BẮT BUỘC gọi tool khi khách hàng yêu cầu tra cứu sản phẩm, hỏi giá hoặc gửi khiếu nại/hỗ trợ.
3. Nếu tra cứu catalog không có kết quả phù hợp, phải trả lời trung thực và lịch sự rằng không tìm thấy sản phẩm, tuyệt đối không bịa ra sản phẩm khác.
4. Khi khách hàng có cả hai nhu cầu (vừa tìm sản phẩm vừa gửi phản hồi/lỗi), phải thực thi độc lập cả hai công cụ để xử lý trọn vẹn yêu cầu.
5. Đối với các câu hỏi FAQ về chính sách chung (như chính sách bảo hành xe điện VinFast: bảo hành pin 10 năm hoặc 200.000 km), trả lời trực tiếp chính xác mà không cần gọi tool.

## 4. OPERATIONAL BOUNDARIES
- Chỉ phục vụ các thông tin và dịch vụ liên quan đến hệ sinh thái Vingroup (VinFast, Vinpearl, VinWonders, v.v.).
- Lịch sự từ chối các yêu cầu nằm ngoài phạm vi hoạt động của Vingroup hoặc các hành vi vi phạm đạo đức, bảo mật.

## 5. OUTPUT CONTRACT (REACT FORMAT)
Khi cần suy luận và thực thi công cụ, tuân thủ nghiêm ngặt chu trình Thought - Action - Action Input - Observation:

Thought: Phân tích ý định người dùng và xác định hành động cần thực hiện.
Action: Tên công cụ cần gọi (search_product_catalog hoặc submit_support_ticket). Nếu không cần gọi công cụ, ghi None.
Action Input: Tham số đầu vào cho công cụ dưới dạng JSON object.
Observation: Dữ liệu thực tế trả về từ công cụ.
... (Có thể lặp lại Thought/Action/Action Input/Observation nếu cần gọi nhiều công cụ)
Thought: Đã có đủ thông tin để tổng hợp câu trả lời cuối cùng cho khách hàng.
Final Answer: Câu trả lời hoàn chỉnh, lịch sự, chính xác gửi tới khách hàng.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        # TODO 2: Trả về câu trả lời tĩnh (mock) hoặc gọi Gemini API 1 lượt (không dùng tool)
        # Mục tiêu: Quan sát hiện tượng bịa thông tin (hallucination)
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.system_prompt = SYSTEM_PROMPT
        self.trace: List[Dict[str, Any]] = []

    def _detect_intents(self, user_input: str) -> Dict[str, Any]:
        """Phân tích intent và trích xuất tham số từ user_input (TODO 3)."""
        user_lower = user_input.lower()

        # 1. Phát hiện câu hỏi FAQ (không cần gọi tool)
        is_faq = (
            ("bảo hành" in user_lower or "chính sách" in user_lower)
            and not any(k in user_lower for k in ["lỗi", "sự cố", "hỏng", "khiếu nại", "tôi tên", "tên tôi là"])
            and not any(k in user_lower for k in ["dưới", "giá", "triệu", "mua"])
        )

        # 2. Phát hiện nhu cầu tra cứu danh mục (needs_catalog)
        has_catalog_action = any(k in user_lower for k in ["xem", "tìm", "mua", "giá", "dưới", "chi phí", "bao nhiêu", "tham khảo", "bảng giá", "resort", "nghỉ dưỡng", "đặt phòng"])
        needs_catalog = not is_faq and has_catalog_action and any(k in user_lower for k in ["xe", "ô tô", "vinfast", "resort", "vinpearl", "du lịch", "phòng", "khách sạn"])

        # Trích xuất tham số catalog
        if any(k in user_lower for k in ["resort", "vinpearl", "du lịch", "du_lich", "phòng", "khách sạn"]):
            category = "du_lich"
        else:
            category = "xe_dien"

        price_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:triệu|tr\b)', user_input, re.IGNORECASE)
        if price_match:
            max_price = int(float(price_match.group(1)) * 1_000_000)
        else:
            ty_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:tỷ|ty\b)', user_input, re.IGNORECASE)
            if ty_match:
                max_price = int(float(ty_match.group(1)) * 1_000_000_000)
            else:
                num_match = re.search(r'\b(\d{7,})\b', user_input)
                if num_match:
                    max_price = int(num_match.group(1))
                else:
                    max_price = 999999999999

        catalog_args = {"category": category, "max_price": max_price}

        # 3. Phát hiện nhu cầu gửi ticket hỗ trợ (needs_ticket - kiểm tra độc lập tránh Trap 3)
        ticket_keywords = ["lỗi", "sự cố", "hỏng", "khiếu nại", "phản hồi", "ticket", "tôi tên", "tên tôi là", "cần xử lý", "hỗ trợ kỹ thuật", "ẩm mốc"]
        needs_ticket = not is_faq and any(k in user_lower for k in ticket_keywords)

        # Trích xuất tham số ticket
        name_match = re.search(r'(?:tôi tên là|tôi tên|tên tôi là)\s+([A-ZÀ-Ỹa-zà-ỹ\s]+?)(?:,|\.|\bxe\b|\bphòng\b|$)', user_input, re.IGNORECASE)
        customer_name = name_match.group(1).strip() if name_match else "Khách hàng"

        if any(w in user_lower for w in ["gấp", "nghiêm trọng", "khẩn cấp", "high"]):
            priority = "high"
        elif any(w in user_lower for w in ["thấp", "low"]):
            priority = "low"
        else:
            priority = "medium"

        desc_match = re.search(r'((?:xe|phòng|hệ thống|thiết bị)[^,.]*?bị\s+[^,.]+)', user_input, re.IGNORECASE)
        if desc_match:
            issue_description = desc_match.group(1).strip()
        else:
            desc_match2 = re.search(r'((?:bị|lỗi|sự cố|phản hồi:?)\s+[^,.]+)', user_input, re.IGNORECASE)
            issue_description = desc_match2.group(1).strip() if desc_match2 else user_input.strip()

        ticket_args = {
            "customer_name": customer_name,
            "issue_description": issue_description,
            "priority": priority
        }

        return {
            "is_faq": is_faq,
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "catalog_args": catalog_args,
            "ticket_args": ticket_args
        }

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop (TODO 4)."""
        self.trace = []

        # Safeguard: Kiểm tra giới hạn số bước lặp ban đầu
        if self.max_iterations < 1:
            return {
                "answer": "Lỗi: Vượt quá số bước tối đa.",
                "trace": self.trace,
                "iterations": 0,
                "status": "max_iterations_reached"
            }

        # TODO 3: Phân tích intent từ user_input
        intents = self._detect_intents(user_input)

        # Trường hợp FAQ: Không cần gọi tool, trả lời trực tiếp trong 1 bước
        if intents["is_faq"]:
            answer = "Chính sách bảo hành pin xe điện VinFast kéo dài 10 năm hoặc 200.000 km (tùy điều kiện nào đến trước), áp dụng bảo hành chính hãng đối với pin xe điện VinFast."
            self.trace.append({
                "iteration": 1,
                "thought": "Người dùng hỏi về chính sách bảo hành pin xe điện VinFast. Đây là câu hỏi FAQ thông thường, trả lời trực tiếp mà không cần gọi tool.",
                "action": None,
                "action_input": {},
                "observation": None,
                "final_answer": answer
            })
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": 1,
                "status": "completed"
            }

        # 💡 Mẹo: Luôn kiểm tra intents["needs_catalog"] và intents["needs_ticket"] để quyết định số iteration cần thiết
        needs_catalog = intents["needs_catalog"]
        needs_ticket = intents["needs_ticket"]

        if not needs_catalog and not needs_ticket:
            answer = "VinAssistant có thể hỗ trợ quý khách tra cứu thông tin sản phẩm hoặc ghi nhận yêu cầu hỗ trợ kỹ thuật."
            self.trace.append({
                "iteration": 1,
                "thought": "Không nhận diện được công cụ cần thực thi, phản hồi hướng dẫn người dùng.",
                "final_answer": answer
            })
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": 1,
                "status": "completed"
            }

        # TODO 4: Xây dựng Agent Loop (while iteration <= self.max_iterations)
        iteration = 0
        catalog_results = None
        ticket_result = None

        # Iteration 1: Gọi search_product_catalog nếu cần
        if needs_catalog:
            iteration += 1
            if iteration > self.max_iterations:
                return {
                    "answer": "Lỗi: Vượt quá số bước tối đa.",
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "max_iterations_reached"
                }

            cat_args = intents["catalog_args"]
            thought = f"Khách hàng muốn tra cứu danh mục '{cat_args['category']}' với giá tối đa {cat_args['max_price']:,} VNĐ. Cần gọi tool search_product_catalog."
            catalog_results = search_product_catalog(**cat_args)
            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "action": "search_product_catalog",
                "action_input": cat_args,
                "observation": catalog_results
            })

        # Iteration 2 (hoặc 1 nếu chỉ có ticket): Gọi submit_support_ticket nếu cần
        if needs_ticket:
            iteration += 1
            if iteration > self.max_iterations:
                return {
                    "answer": "Lỗi: Vượt quá số bước tối đa.",
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "max_iterations_reached"
                }

            tic_args = intents["ticket_args"]
            thought = f"Khách hàng '{tic_args['customer_name']}' yêu cầu hỗ trợ sự cố: {tic_args['issue_description']}. Cần gọi tool submit_support_ticket."
            ticket_result = submit_support_ticket(**tic_args)
            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "action": "submit_support_ticket",
                "action_input": tic_args,
                "observation": ticket_result
            })

        # Iteration 3+: Tổng hợp tất cả observation từ trace → xuất Final Answer
        answer_parts = []
        if catalog_results is not None:
            if len(catalog_results) == 0:
                # Milestone 4.2: Empty results fallback
                answer_parts.append("Rất tiếc, không tìm thấy sản phẩm nào phù hợp với yêu cầu về ngân sách của quý khách trong hệ thống.")
            else:
                items_desc = []
                for p in catalog_results:
                    price_str = f"{p['price_vnd'] / 1_000_000:,.0f} triệu VNĐ" if p['price_vnd'] >= 1_000_000 else f"{p['price_vnd']:,} VNĐ"
                    items_desc.append(f"- {p['name']} (Giá: {price_str}): {p.get('description', '')}")
                answer_parts.append("Dưới đây là các sản phẩm phù hợp với yêu cầu của quý khách:\n" + "\n".join(items_desc))

        if ticket_result is not None:
            ticket_msg = (
                f"Yêu cầu hỗ trợ của quý khách {ticket_result['customer_name']} đã được ghi nhận thành công "
                f"với mã vé {ticket_result['ticket_id']} (Mức độ ưu tiên: {ticket_result['priority']}, "
                f"Trạng thái: {ticket_result['status']})."
            )
            answer_parts.append(ticket_msg)

        final_answer = "\n\n".join(answer_parts)

        self.trace.append({
            "step": "final_answer",
            "thought": "Đã tổng hợp đầy đủ thông tin từ các công cụ, chuẩn bị xuất Final Answer cho khách hàng.",
            "final_answer": final_answer
        })

        return {
            "answer": final_answer,
            "trace": self.trace,
            "iterations": iteration,
            "status": "completed"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

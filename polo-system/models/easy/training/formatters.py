# models/easy/training/formatters.py
def simplification_formatter(original_text: str, simplified_text: str):
    prompt = (
        "다음 문장을 한국어로 쉽고 간결하게 바꿔줘. 의미는 보존하고, 전문용어는 쉬운 말로.\n"
        "### 원문:\n"
        f"{original_text}\n"
        "### 답변(쉬운 한국어):\n"
    )
    target = simplified_text.strip()
    return prompt, target
from typing import List, Dict


SYSTEM_PROMPT = (
    "You are an expert assistant answering Korean Criminal Law KMMLU multiple-choice questions.\n"
    "Use the provided context if helpful.\n"
    "Answer strictly with a single uppercase letter: A, B, C, or D.\n"
    "Do not include any explanation."
)


def format_context(contexts: List[str], max_chars: int = 2000) -> str:
    text = "\n\n".join(contexts)
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return text


def build_user_prompt(question: str, choices: List[str], contexts: List[str]) -> str:
    # Ensure exactly 4 choices
    padded = choices[:4] + [""] * max(0, 4 - len(choices))
    A, B, C, D = padded[:4]
    ctx = format_context(contexts)
    return (
        f"[Question]\n{question}\n\n"
        f"[Choices]\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\n"
        f"[Context]\n{ctx}\n\n"
        f"Return only one letter among: A, B, C, D."
    )


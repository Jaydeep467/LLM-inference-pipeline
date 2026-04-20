"""
Prompt Engineering Hooks
Template injection layer for structured prompting.
Supports system prompts, few-shot examples, and role conditioning.
"""

from typing import Optional, List, Dict

# ──────────────────────────────────────────────
# Built-in templates
# ──────────────────────────────────────────────
TEMPLATES: Dict[str, str] = {
    "default": "{prompt}",

    "instruct": (
        "### Instruction:\n{prompt}\n\n### Response:\n"
    ),

    "chat": (
        "System: You are a helpful, concise AI assistant.\n"
        "User: {prompt}\n"
        "Assistant:"
    ),

    "summarize": (
        "Summarize the following text in 2-3 sentences:\n\n{prompt}\n\nSummary:"
    ),

    "code": (
        "Write clean, well-commented Python code for the following task:\n\n"
        "{prompt}\n\n```python\n"
    ),

    "qa": (
        "Answer the following question accurately and concisely.\n\n"
        "Question: {prompt}\nAnswer:"
    ),
}


# ──────────────────────────────────────────────
# Few-shot builder
# ──────────────────────────────────────────────
def build_few_shot(
    examples: List[Dict[str, str]],
    prompt: str,
    input_key: str = "input",
    output_key: str = "output",
) -> str:
    """
    Build a few-shot prompt from example pairs.

    examples = [{"input": "...", "output": "..."}, ...]
    """
    parts = []
    for ex in examples:
        parts.append(f"Input: {ex[input_key]}\nOutput: {ex[output_key]}")
    parts.append(f"Input: {prompt}\nOutput:")
    return "\n\n".join(parts)


# ──────────────────────────────────────────────
# Main injection function
# ──────────────────────────────────────────────
def apply_prompt_template(
    prompt: str,
    template: str = "default",
    system_prompt: Optional[str] = None,
    few_shot_examples: Optional[List[Dict[str, str]]] = None,
    max_prompt_length: int = 800,
) -> str:
    """
    Apply a named template (or custom string) to a raw prompt.
    Optionally prepends a system prompt and few-shot examples.
    Truncates to max_prompt_length tokens (approximate).
    """
    # 1. Few-shot override
    if few_shot_examples:
        base = build_few_shot(few_shot_examples, prompt)
    else:
        tmpl = TEMPLATES.get(template, TEMPLATES["default"])
        base = tmpl.format(prompt=prompt)

    # 2. System prompt prefix
    if system_prompt:
        base = f"{system_prompt.strip()}\n\n{base}"

    # 3. Rough truncation (characters ≈ tokens for GPT-2 tokeniser)
    if len(base) > max_prompt_length * 4:
        base = base[: max_prompt_length * 4]

    return base


# ──────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────
def list_templates() -> List[str]:
    return list(TEMPLATES.keys())

"""Append-only prompts for the GSM8K three-step workflow."""


STEP_INSTRUCTIONS = (
    "Step 1 — Extract: List the quantities and variables stated in the problem.",
    "Step 2 — Plan: Derive the arithmetic expression or equations needed to solve it.",
    "Step 3 — Compute: Evaluate the plan and state the final numeric answer.",
)


def build_initial_prompt(question: str) -> str:
    return f"Question: {question.strip()}\n\n{STEP_INSTRUCTIONS[0]}\n"


def append_step(prompt: str, output: str, next_step_index: int | None) -> str:
    """Append a model response and, if present, the next step instruction."""
    expanded = f"{prompt}{output.strip()}\n"
    if next_step_index is not None:
        expanded += f"\n{STEP_INSTRUCTIONS[next_step_index]}\n"
    return expanded

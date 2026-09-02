"""Token-stable, append-only prompts for the GSM8K workflow."""

from dataclasses import dataclass

STEP_INSTRUCTIONS = (
    "Step 1 — Extract: List the quantities and variables stated in the problem.",
    "Step 2 — Plan: Derive the arithmetic expression or equations needed to solve it.",
    "Step 3 — Compute: Evaluate the plan and state the final numeric answer.",
)


@dataclass(frozen=True)
class PromptState:
    """Canonical prompt text plus the suffix added since the previous step.

    ``suffix_text`` excludes the prior model output because those generated
    tokens are already present in the completed cache.
    """

    text: str
    suffix_text: str | None = None

    @classmethod
    def from_question(cls, question: str) -> "PromptState":
        return cls(f"Question: {question.strip()}\n\n{STEP_INSTRUCTIONS[0]}\n")

    def after_output(self, raw_output: str, next_step_index: int | None) -> "PromptState":
        suffix = ""
        if next_step_index is not None:
            suffix = f"\n\n{STEP_INSTRUCTIONS[next_step_index]}\n"
        return PromptState(text=f"{self.text}{raw_output}{suffix}", suffix_text=suffix)


def build_initial_prompt(question: str) -> str:
    """Backward-compatible string helper."""
    return PromptState.from_question(question).text


def append_step(prompt: str, output: str, next_step_index: int | None) -> str:
    """Backward-compatible append helper that never strips generated text."""
    expanded = f"{prompt}{output}"
    if next_step_index is not None:
        expanded += f"\n\n{STEP_INSTRUCTIONS[next_step_index]}\n"
    return expanded

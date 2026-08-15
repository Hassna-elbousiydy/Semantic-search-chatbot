from collections.abc import Sequence

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


MODEL_NAME = "google/flan-t5-base"

REFUSAL_ANSWER = (
    "I don't know based on the provided context."
)


class AnswerGenerator:
    """
    Lightweight grounded answer generator.

    The model receives retrieved scientific
    evidence and must answer only from that
    evidence.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
    ):
        print(
            f"Loading generation model: "
            f"{model_name}"
        )

        self.device = torch.device(
            "cpu"
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )

        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                model_name
            )
        )

        self.model.to(
            self.device
        )

        self.model.eval()

    @staticmethod
    def _normalize_contexts(
        contexts: str | Sequence[str],
    ) -> list[str]:
        """
        Normalize generator input.

        Accept either:
        - one context string;
        - a sequence of context strings.

        A raw string must never be treated as
        a sequence of individual characters.
        """

        if isinstance(
            contexts,
            str,
        ):
            contexts = [
                contexts
            ]

        normalized = []

        for context in contexts:

            if not context:
                continue

            clean_context = (
                context.strip()
            )

            if clean_context:
                normalized.append(
                    clean_context
                )

        return normalized

    def _build_prompt(
        self,
        question: str,
        context: str,
    ) -> str:
        """
        Build an extractive-style scientific QA
        prompt.

        The instruction favors short factual
        answers and discourages hallucination.
        """

        return (
            "Answer the scientific question "
            "using only the evidence below.\n"
            "\n"
            "Rules:\n"
            "1. Use only information explicitly "
            "stated in the evidence.\n"
            "2. Preserve numbers, percentages, "
            "technical terms, model names, and "
            "units exactly when possible.\n"
            "3. If several items are requested, "
            "include all of them.\n"
            "4. Give only the concise answer, "
            "not an explanation.\n"
            "5. If the evidence does not contain "
            "the answer, reply exactly:\n"
            f"{REFUSAL_ANSWER}\n"
            "\n"
            f"Evidence:\n{context}\n"
            "\n"
            f"Question:\n{question}\n"
            "\n"
            "Answer:"
        )

    def generate(
        self,
        question: str,
        contexts: str | Sequence[str],
    ) -> str:
        """
        Generate a grounded scientific answer.
        """

        context_list = (
            self._normalize_contexts(
                contexts
            )
        )

        if not context_list:
            return REFUSAL_ANSWER

        # The RAG architecture currently selects
        # one strongest passage for generation.
        context = context_list[0]

        # Avoid excessive prompt length on the
        # local CPU model.
        context = context[:1800]

        prompt = self._build_prompt(
            question=question,
            context=context,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        inputs = {
            key: value.to(
                self.device
            )
            for key, value
            in inputs.items()
        }

        with torch.inference_mode():

            outputs = (
                self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                    num_beams=4,
                    early_stopping=True,
                    repetition_penalty=1.05,
                )
            )

        answer = (
            self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
            )
            .strip()
        )

        if not answer:
            return REFUSAL_ANSWER

        return answer
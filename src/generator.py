from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "google/flan-t5-small"


class AnswerGenerator:

    def __init__(self, model_name: str = MODEL_NAME):
        print(f"Loading generation model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, question: str, contexts: list[str]) -> str:

        context_text = "\n\n".join(contexts)

        prompt = f"""
Answer the question using only the context below.

If the answer cannot be found in the context, say:
"I don't know based on the provided context."

Context:
{context_text}

Question:
{question}

Answer:
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False
        )

        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return answer.strip()
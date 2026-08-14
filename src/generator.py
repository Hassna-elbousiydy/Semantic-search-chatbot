from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "google/flan-t5-small"


class AnswerGenerator:

    def __init__(self, model_name: str = MODEL_NAME):
        print(f"Loading generation model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, question: str, contexts: list[str]) -> str:

        # Keep only the most relevant context for this lightweight model.
        context = contexts[0] if contexts else ""

        # Limit context size to prevent prompt truncation.
        context = context[:1400]

        prompt = (
            f"Question: {question}\n\n"
            f"Context: {context}\n\n"
            "Instruction: Answer the question using only the context. "
            "Give a short and precise answer. "
            "If the context does not contain the answer, say "
            "\"I don't know based on the provided context.\"\n\n"
            "Answer:"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            num_beams=4,
            early_stopping=True
        )

        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return answer.strip()
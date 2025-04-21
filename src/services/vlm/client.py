import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

from src.settings.logger import setup_logger

logger = setup_logger()


class CustomLLM:
    def __init__(self, model_path: str = "HuggingFaceTB/SmolVLM-Instruct"):
        """
        Initialize the custom LLM.

        Args:
            model_path (str): Path to the custom model.
        """
        # Load your custom model here
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        attention_type = "flash_attention_2" if self.device == "cuda" else "eager"
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForVision2Seq.from_pretrained(model_path,
                                                            torch_dtype=torch.bfloat16,
                                                            _attn_implementation=attention_type).to(self.device)

    def preprocess_input(self, inputs: dict):
        """
                Preprocess inputs using the processor.

                Args:
                    inputs (Dict): A dictionary containing 'context' and 'question'.

                Returns:
                    Dict: Processed inputs ready for the model.
                """
        formatted_texts = "\n".join(inputs["context"]["texts"])
        logger.info(f"Formatted texts: {len(formatted_texts)}")

        messages = [
            {"role": "user", "content": [{"type": "text", "text": f"answer user question: {inputs["question"]}\n\n"
                                                                  f"answer using ONLY information in text "
                                                                  f"and / or tables below:\n"
                                                                  f"{formatted_texts}"}]}
        ]

        # Добавляем изображения в сообщения, если они присутствуют
        if inputs["context"]["images"]:
            for image in inputs["context"]["images"]:
                image_message = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
                messages[0]["content"].append(image_message)

        processed = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return processed

    def generate(self, prompt: dict) -> str:
        """
        Generate a response from the custom LLM.

        Args:
            prompt (dict): Input for the LLM consisting of context and question.

        Returns:
            str: Generated response.
        """
        processed = self.preprocess_input(prompt)
        logger.debug(f"Processed prompt: {len(processed["input_ids"])}")

        inputs = {k: v.to(self.device) for k, v in processed.items()}
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        prompt_len = len(processed['input_ids'][0])
        mod_gen_id = generated_ids[0][prompt_len:].reshape(1, -1)
        generated_texts = self.processor.batch_decode(
            mod_gen_id,
            skip_special_tokens=True,
        )
        return generated_texts[0].strip()


if __name__ == "__main__":
    # from langchain_core.messages import HumanMessage

    # message = [HumanMessage(content=[
    #     {
    #         "type": "text",
    #         "text": ("What is attention mechanism in neural networks? Be eloquent."),
    #     }
    # ]
    # )]
    test_human_message = False
    if test_human_message:
        message = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is attention mechanism in neural networks? Be eloquent."}
                ]
            },
        ]

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        attention_type = "flash_attention_2" if device == "cuda" else "eager"
        processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
        model = AutoModelForVision2Seq.from_pretrained("HuggingFaceTB/SmolVLM-Instruct",
                                                       torch_dtype=torch.bfloat16,
                                                       _attn_implementation=attention_type).to(device)

        print(device)
        # Prepare inputs
        prompt = processor.apply_chat_template(
            message,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)

        # inputs = processor(text=prompt,
        #                    return_tensors="pt")
        inputs = prompt.to(device)

        generated_ids = model.generate(**inputs, max_new_tokens=128,)
        generated_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        print(generated_texts[0])

    llm = CustomLLM()
    q_and_context = {
        "context": "no information found",
        "question": "what does the world look like today?",
    }
    print(llm.generate(q_and_context))

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "mps" if torch.backends.mps.is_available() else DEVICE

attention_type = "flash_attention_2" if DEVICE == "cuda" else "eager"
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
model = AutoModelForVision2Seq.from_pretrained("HuggingFaceTB/SmolVLM-Instruct",
                                               torch_dtype=torch.bfloat16,
                                               _attn_implementation=attention_type).to(DEVICE)


if __name__ == "__main__":
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "How are you? What can you do?"}
            ]
        },
    ]

    print(DEVICE)
    # Prepare inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt,
                       return_tensors="pt")
    inputs = inputs.to(DEVICE)

    generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    print(generated_texts[0])

import os
import torch
import fitz  # PyMuPDF
from PIL import Image
from huggingface_hub import login
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer, AutoModelForCausalLM

# Токен Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("HF_TOKEN не задан. Создайте файл .env с вашим токеном Hugging Face.")

# Обновленный промпт для OCR: указываем мультиязычность и рукописный текст
OCR_PROMPT = """
Act as an expert OCR system with advanced linguistic capabilities.
Your task is to extract all text (both printed and handwritten) from the image in ANY language with 100% fidelity.

### Instructions:
1. **Structural Analysis**: Identify the layout. Maintain the original formatting. **CRITICAL: Preserve ALL physical line breaks exactly as they appear in the image. Do not merge lines into a single paragraph.**
2. **Character Recognition**: Transcribe every character exactly as shown, regardless of the language or handwriting style.
3. **Noise Suppression**: Ignore smudges, background textures, watermarks, or paper folds.
4. **Correction**: Fix obvious OCR misreadings only if you are certain.

### Constraint:
Do not include any conversational filler. Output only the requested extracted text."""


def initialize_models(token: str):
    print("Авторизация в Hugging Face...")
    login(token)

    # 1. Загрузка OCR модели (понимает рукописный текст и множество языков)
    ocr_model_id = "JackChew/Qwen2-VL-2B-OCR"
    print(f"Загрузка OCR модели {ocr_model_id}...")

    ocr_model = AutoModelForImageTextToText.from_pretrained(
        ocr_model_id,
        device_map="cpu",
        attn_implementation="sdpa",
        torch_dtype=torch.float32
    )
    ocr_processor = AutoProcessor.from_pretrained(ocr_model_id)

    torch.set_num_threads(8)

    # 2. Загрузка переводчика
    translator_id = "Qwen/Qwen3.5-0.8B"
    print(f"Загрузка модели переводчика {translator_id}...")

    translator_tokenizer = AutoTokenizer.from_pretrained(translator_id)
    translator_model = AutoModelForCausalLM.from_pretrained(
        translator_id,
        device_map="cpu",
        torch_dtype=torch.float32
    )

    return (ocr_model, ocr_processor), (translator_model, translator_tokenizer)


def translate_text(text: str, translator) -> str:
    """
    Переводит ВЕСЬ текст (OCR output целиком) за один вызов переводчика.
    """
    if not text.strip():
        return ""

    model, tokenizer = translator

    messages = [
        {
            "role": "system",
            "content": (
                "You are a highly accurate professional translator. "
                "Translate the provided text from any language into Russian. "
                "If the text is ALREADY in Russian, output it exactly as it is "
                "without any changes or translation. "
                "Preserve all line breaks from the original. "
                "Output ONLY the resulting Russian text. "
                "Do not add any explanations, notes, or quotes."
            ),
        },
        {
            "role": "user",
            "content": text,
        },
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=False,
        temperature=None,
        top_p=None,
    )

    input_length = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()



def run_qwen_ocr(image: Image.Image, model, processor) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": OCR_PROMPT}
            ]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    outputs = model.generate(**inputs, do_sample=False, max_new_tokens=2048)

    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    return processor.decode(generated_tokens, skip_special_tokens=True).strip()


def pdf_to_images(pdf_path: str) -> list:
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images


def process_document(file_path: str, output_txt_path: str):
    (ocr_model, ocr_processor), translator = initialize_models(HF_TOKEN)

    if file_path.lower().endswith(".pdf"):
        images = pdf_to_images(file_path)
    else:
        images = [Image.open(file_path).convert("RGB")]

    with open(output_txt_path, "w", encoding="utf-8") as f_out:
        for i, img in enumerate(images):
            print(f"Обработка страницы {i + 1}...")
            extracted_text = run_qwen_ocr(img, ocr_model, ocr_processor)

            print(f"Перевод страницы {i + 1}...")
            russian_text = translate_text(extracted_text, translator)

            f_out.write(f"=== СТРАНИЦА {i + 1} (ORIGINAL) ===\n")
            f_out.write(extracted_text + "\n\n")
            f_out.write(f"=== СТРАНИЦА {i + 1} (RUSSIAN TRANSLATION) ===\n")
            f_out.write(russian_text + "\n\n")
            f_out.write("=" * 50 + "\n\n")

    print(f"Готово! Результаты в {output_txt_path}")


if __name__ == "__main__":
    INPUT_FILE = "images/chin.png"  # Укажите здесь путь к изображению с рукописью
    OUTPUT_FILE = "ocr_result.txt"

    if os.path.exists(INPUT_FILE):
        process_document(INPUT_FILE, OUTPUT_FILE)
    else:
        print("Файл не найден")
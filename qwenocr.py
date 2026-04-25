import os
import re
import torch
import fitz  # PyMuPDF
from PIL import Image
from huggingface_hub import login
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer, AutoModelForCausalLM

# Токен Hugging Face
HF_TOKEN = "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"

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


def translate_and_preserve_shape(sentence: str, model, tokenizer) -> str:
    """Переводит единое предложение, сохраняя внутри него оригинальные переносы строк пропорционально."""
    nl_indices = [i for i, char in enumerate(sentence) if char == '\n']

    clean_sentence = sentence.replace('\n', ' ')
    clean_sentence = re.sub(r'\s+', ' ', clean_sentence).strip()

    if not clean_sentence:
        return sentence

    # Обновленный промпт для перевода: добавлено условие проверки на русский язык
    messages = [
        {
            "role": "system",
            "content": "You are a highly accurate professional translator. Translate the provided text from any language into Russian. If the text is ALREADY in Russian, output it exactly as it is without any changes or translation. Output ONLY the resulting Russian text. Do not add any explanations, notes, or quotes."
        },
        {
            "role": "user",
            "content": clean_sentence
        }
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        temperature=None,
        top_p=None
    )

    input_length = inputs['input_ids'].shape[1]
    translated = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

    if not nl_indices or not translated:
        return translated

    orig_len = len(sentence)
    proportions = [idx / orig_len for idx in nl_indices]

    trans_len = len(translated)
    result_chars = list(translated)

    for prop in proportions:
        target_idx = int(trans_len * prop)
        if target_idx >= trans_len:
            target_idx = max(0, trans_len - 1)

        left_space = target_idx
        while left_space >= 0 and result_chars[left_space] != ' ':
            left_space -= 1

        right_space = target_idx
        while right_space < trans_len and result_chars[right_space] != ' ':
            right_space += 1

        dist_left = target_idx - left_space if left_space >= 0 and result_chars[left_space] == ' ' else float('inf')
        dist_right = right_space - target_idx if right_space < trans_len and result_chars[
            right_space] == ' ' else float('inf')

        if dist_left == float('inf') and dist_right == float('inf'):
            continue

        best_space = left_space if dist_left <= dist_right else right_space
        result_chars[best_space] = '\n'

    return "".join(result_chars)


def translate_text(text: str, translator) -> str:
    if not text.strip():
        return ""

    model, tokenizer = translator

    paragraphs = re.split(r'(\n\s*\n)', text)
    translated_paragraphs = []

    for p in paragraphs:
        if re.match(r'^\n\s*\n$', p):
            translated_paragraphs.append(p)
            continue

        parts = re.split(r'(?<=[.!?])(\s+)', p)

        translated_parts = []
        for part in parts:
            if re.match(r'^\s+$', part):
                translated_parts.append(part)
            elif part.strip():
                translated_part = translate_and_preserve_shape(part, model, tokenizer)
                translated_parts.append(translated_part)
            else:
                translated_parts.append(part)

        translated_paragraphs.append("".join(translated_parts))

    return "".join(translated_paragraphs)


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
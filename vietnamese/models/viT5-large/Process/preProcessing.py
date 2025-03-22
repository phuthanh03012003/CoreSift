import unicodedata
import re
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load model và tokenizer của VietAI/viT5-base-vietnews-summarization
model_name = "VietAI/viT5-base-vietnews-summarization"  
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Hàm làm sạch văn bản
def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace('—', '-').replace('“', '"').replace('”', '"').replace('’', "'")
    text = re.sub(r'[^\w\s.,!?-]', '', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Đọc file và làm sạch
input_file = "../../../inputs/TamCam.txt"
with open(input_file, "r", encoding="utf-8") as f:
    raw_text = f.read().strip()

cleaned_paragraphs = [clean_text(paragraph) for paragraph in raw_text.split(("\n\n")) if paragraph.strip()]

# Thêm prefix "summarize: " cho tác vụ tóm tắt
prefixed_paragraphs = ["summarize: " + paragraph for paragraph in cleaned_paragraphs]

# Kiểm tra dữ liệu
print(f"1. Số lượng đoạn văn: {len(prefixed_paragraphs)}\n")

# Tokenize từng đoạn và hiển thị token
for i, paragraph in enumerate(prefixed_paragraphs):
    tokens = tokenizer.tokenize(paragraph)
    input_ids = tokenizer(paragraph, return_tensors="pt")["input_ids"][0]
    decoded_tokens = [tokenizer.decode([tid]) for tid in input_ids]
    
    print(f"2. Đoạn {i+1}:")
    print(f"   Số lượng token: {len(tokens)}")
    print(f"   Tokenized: {tokens}")

# Tạo Tensor đầu vào
inputs = tokenizer(prefixed_paragraphs, padding=True, truncation=True, max_length=512, return_tensors="pt")

# Hiển thị kích thước tensor
print(f"Input IDs shape: {inputs['input_ids'].shape}")
print(f"Attention Mask shape: {inputs['attention_mask'].shape}")

# Lưu tensor
output_path = "../../../outputs/processed_inputs.pt"
torch.save(inputs, output_path)
print(f"Đã lưu tensor Input IDs vào file {output_path}")

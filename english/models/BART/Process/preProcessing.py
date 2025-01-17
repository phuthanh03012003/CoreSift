import unicodedata
import re
import torch
from transformers import BartForConditionalGeneration, BartTokenizer

# Load model và tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Hàm làm sạch văn bản
def clean_text(text):
    text = unicodedata.normalize("NFKD", text)  # Chuẩn hóa Unicode
    text = text.replace('—', '-').replace('“', '"').replace('”', '"').replace('’', "'")
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Xóa ký tự không phải ASCII
    text = re.sub(r'\s+', ' ', text).strip()    # Xóa khoảng trắng thừa
    return text

# Đọc file với mã hóa UTF-8
input_file = "../../../inputs/TamCam.txt"
with open(input_file, "r", encoding="utf-8") as f:
    raw_text = f.read().strip().split("\n\n")

# Làm sạch từng đoạn văn
cleaned_paragraphs = [clean_text(paragraph) for paragraph in raw_text]

# Kiểm tra dữ liệu
print(f"1. Số lượng đoạn văn: {len(cleaned_paragraphs)}\n\n")

# Tokenize từng đoạn và làm sạch token
for i, paragraph in enumerate(cleaned_paragraphs):
    tokens = tokenizer.tokenize(paragraph)
    clean_tokens = [token.replace('Ġ', '').replace('Ċ', '') for token in tokens]
    print(f"2. Đoạn {i+1}: Số lượng token: {len(clean_tokens)}")
    print(f"   Tokenized (Đã làm sạch): {clean_tokens}\n\n")

# Tạo Tensor đầu vào với max_length
inputs = tokenizer(cleaned_paragraphs, padding=True, truncation=True, max_length=512, return_tensors="pt")

# Hiển thị Input IDs
print("3. Tensor Input (input_ids):")
print(inputs["input_ids"])
print("\n\n")

# Hiển thị Attention Mask
print("4. Attention Mask:")
print(inputs["attention_mask"])

# Lưu kết quả để dùng cho modelProcessing.py
torch.save(inputs, "object/processed_inputs.pt")
print("Đã lưu tensor Input IDs vào file processed_inputs.pt")


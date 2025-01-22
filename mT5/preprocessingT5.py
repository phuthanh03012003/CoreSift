import unicodedata
import re
from transformers import MT5Tokenizer

# Load tokenizer mT5
model_name = "google/mt5-small"  # Có thể đổi thành phiên bản khác nếu cần
tokenizer = MT5Tokenizer.from_pretrained(model_name)

# Hàm làm sạch văn bản
def clean_text(text):
    text = unicodedata.normalize("NFKD", text)  # Chuẩn hóa Unicode
    text = text.replace('—', '-').replace('“', '"').replace('”', '"').replace('’', "'")
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Xóa ký tự không phải ASCII
    text = re.sub(r'\s+', ' ', text).strip()    # Xóa khoảng trắng thừa
    return text

# Đọc file với mã hóa UTF-8
input_file = "../TamCam.txt"  # Đổi đường dẫn file nếu cần
with open(input_file, "r", encoding="utf-8") as f:
    raw_text = f.read().strip().split("\n\n")

# Làm sạch từng đoạn văn
cleaned_paragraphs = [clean_text(paragraph) for paragraph in raw_text]

# Kiểm tra dữ liệu đã làm sạch
print(f"1. Số lượng đoạn văn: {len(cleaned_paragraphs)}\n")
for i, paragraph in enumerate(cleaned_paragraphs[:5]):  # Hiển thị 5 đoạn đầu tiên để kiểm tra
    print(f"Đoạn {i + 1}: {paragraph}\n")

# Tokenize từng đoạn và làm sạch token
for i, paragraph in enumerate(cleaned_paragraphs):
    tokens = tokenizer.tokenize(paragraph)
    print(f"2. Đoạn {i+1}: Số lượng token: {len(tokens)}")
    print(f"   Tokenized: {tokens[:30]}...\n")  # Hiển thị 30 token đầu tiên để kiểm tra

# Tạo Tensor đầu vào với max_length
inputs = tokenizer(cleaned_paragraphs, padding=True, truncation=True, max_length=512, return_tensors="pt")

# Hiển thị Input IDs
print("3. Tensor Input (input_ids):")
print(inputs["input_ids"])
print("\n\n")

# Hiển thị Attention Mask
print("4. Attention Mask:")
print(inputs["attention_mask"])

# Lưu kết quả để dùng cho bước tiếp theo
inputs_file = "processed_mt5_inputs.pt"
torch.save(inputs, inputs_file)
print(f"Đã lưu tensor Input IDs vào file {inputs_file}")

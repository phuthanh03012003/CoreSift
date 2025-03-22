from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os 

# Tải tokenizer và mô hình
model_name = "pengold/t5-vietnamese-summarization"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Văn bản cần tóm tắt
input_file = "../../inputs/TamCam.txt"

# Kiểm tra file input
if not os.path.exists(input_file):
    raise FileNotFoundError(f"File '{input_file}' không tồn tại. Hãy kiểm tra lại đường dẫn.")

with open(input_file, "r", encoding="utf-8") as f:
    raw_text = f.read().strip()

if not raw_text:
    raise ValueError("Nội dung trong file 'TamCam.txt' đang rỗng. Hãy thêm văn bản để tóm tắt.")

# Tiền xử lý và token hóa
inputs = tokenizer.encode(
    "summarize: " + raw_text,  # Thêm tiền tố "summarize:"
    return_tensors="pt",
    max_length=1024,
    truncation=True
)

# Sinh kết quả tóm tắt
output_ids = model.generate(
    inputs,
    max_length=150,        # Độ dài tối đa của văn bản tóm tắt
    min_length=50,         # Độ dài tối thiểu của văn bản tóm tắt
    length_penalty=2.0,    # Hệ số phạt cho độ dài
    num_beams=4,           # Beam search
    early_stopping=True    # Dừng sớm khi đạt kết quả tốt
)

# Giải mã token thành văn bản
summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Hiển thị kết quả
print("Văn bản gốc:")
print(raw_text)
print("\nKết quả tóm tắt:")
print(summary)

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import os
import re  # Thư viện xử lý biểu thức chính quy

# Load model và tokenizer
model_name = "google/pegasus-cnn_dailymail"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

# Đọc nội dung từ file input (KHÔNG dùng encoding="utf-8")
input_file = "../inputs/TamCam.txt"
with open(input_file, "r") as f:
    raw_text = f.read().strip()

# Tokenize văn bản
inputs = tokenizer(
    raw_text,
    return_tensors="pt",
    max_length=1024,  # Giới hạn token của PEGASUS
    truncation=True
)

# Sinh kết quả tóm tắt
summary_ids = model.generate(
    inputs["input_ids"],
    max_length=200,          # Độ dài tối đa của tóm tắt
    min_length=60,           # Độ dài tối thiểu
    length_penalty=2.0,      # Phạt độ dài (ưu tiên văn bản súc tích)
    num_beams=5,             # Beam search để tăng chất lượng kết quả
    early_stopping=True      # Dừng sớm khi đạt kết quả tốt
)

# Giải mã kết quả tóm tắt
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ✅ Xử lý xuống dòng sau mỗi câu
formatted_summary = re.sub(r'([.!?])\s', r'\1\n', summary)

# In kết quả ra màn hình
print("📄 Kết quả tóm tắt:\n", formatted_summary)

# Ghi kết quả vào file (KHÔNG dùng encoding="utf-8")
output_file = "../outputs/PEGASUS-cnn_dailymail.txt"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w") as f:
    f.write(formatted_summary)

print(f"📂 Kết quả đã được ghi vào file: {output_file}")

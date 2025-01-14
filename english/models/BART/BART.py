from transformers import BartForConditionalGeneration, BartTokenizer
import os
import re  # Thư viện xử lý biểu thức chính quy

# Load model và tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

input_file = "../../inputs/TamCam.txt"
with open(input_file, "r") as f:      
    raw_text = f.read().strip()

# Tokenize văn bản
inputs = tokenizer(
    raw_text, 
    return_tensors="pt", 
    max_length=1024,  # BART hỗ trợ tối đa 1024 token
    truncation=True
)

# Generate output (tóm tắt)
summary_ids = model.generate(
    inputs["input_ids"],
    max_length=150,          # Độ dài tối đa của kết quả tóm tắt
    min_length=40,           # Độ dài tối thiểu của kết quả tóm tắt
    length_penalty=2.0,      # Hệ số phạt độ dài (ưu tiên văn bản ngắn hơn)
    num_beams=4,             # Beam search để tăng chất lượng sinh văn bản
    early_stopping=True      # Dừng sớm khi đạt kết quả tối ưu
)

# Giải mã kết quả tóm tắt
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ✅ Xử lý xuống dòng sau mỗi câu
formatted_summary = re.sub(r'([.!?])\s', r'\1\n', summary)

# In kết quả ra màn hình
print("📄 Kết quả tóm tắt:\n", formatted_summary)

output_file = "../../outputs/BART-large-CNN.txt"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w") as f:
    f.write(formatted_summary)

print(f" Kết quả đã được ghi vào file: {output_file}")

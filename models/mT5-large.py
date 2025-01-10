from transformers import MT5ForConditionalGeneration, T5Tokenizer
import os

# Load model và tokenizer
model_name = "google/mt5-large"  # Sử dụng phiên bản mT5 lớn
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

# Đọc nội dung từ file input
input_file = "../input.txt"
with open(input_file, "r", encoding="utf-8") as f:
    raw_text = f.read().strip()

# Thêm tiền tố cho nhiệm vụ tái diễn đạt
task_prefix = "summarize: "  

# Tokenize văn bản
inputs = tokenizer(
    task_prefix + raw_text, 
    return_tensors="pt", 
    max_length=1024, 
    truncation=True
)

# Generate output (tái diễn đạt hoặc tóm tắt)
output_ids = model.generate(
    inputs["input_ids"],
    max_length=100,         # Độ dài tối đa của kết quả
    min_length=20,          # Độ dài tối thiểu của kết quả
    length_penalty=2.0,     # Hệ số phạt độ dài
    num_beams=4,            # Beam search
    early_stopping=True     # Dừng sớm khi đạt kết quả tốt
)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Kết quả:", output_text)

output_file = "../output/mT5-large_output.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(output_text)

print(f"Kết quả đã được ghi vào file: {output_file}")

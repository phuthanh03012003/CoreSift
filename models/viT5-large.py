from transformers import T5Tokenizer, T5ForConditionalGeneration
import os

# Load model và tokenizer
model_name = "VietAI/vit5-base-vietnews-summarization"  # Sử dụng phiên bản đã tinh chỉnh cho tóm tắt
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

input_file = "../story.txt"

# Kiểm tra file input
if not os.path.exists(input_file):
    raise FileNotFoundError(f"File '{input_file}' không tồn tại. Hãy kiểm tra lại đường dẫn.")

with open(input_file, "r", encoding="utf-8") as f:
    raw_text = f.read().strip()

if not raw_text:
    raise ValueError("Nội dung trong file 'input.txt' đang rỗng. Hãy thêm văn bản để tóm tắt.")

task_prefix = "summarize: "

# Tokenize văn bản
inputs = tokenizer(
    task_prefix + raw_text, 
    return_tensors="pt", 
    max_length=1024, 
    truncation=True
)

# Generate output (tóm tắt văn bản)
output_ids = model.generate(
    inputs["input_ids"],
    max_length=150,         # Độ dài tối đa của kết quả
    min_length=50,          # Độ dài tối thiểu của kết quả
    length_penalty=2.0,     # Hệ số phạt độ dài
    num_beams=4,            # Beam search
    early_stopping=True     # Dừng sớm khi đạt kết quả tốt
)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Kết quả tóm tắt:")
print(output_text)

output_file = "../output/viT5-large_output.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(output_text)

print(f"Kết quả đã được ghi vào file: {output_file}")

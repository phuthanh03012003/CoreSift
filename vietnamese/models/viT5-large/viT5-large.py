from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

# Load model và tokenizer
model_name = "VietAI/vit5-large"  # Sử dụng mô hình viT5 lớn
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.cuda()  # Chuyển model sang GPU (nếu có)

input_file = "../../inputs/TamCam.txt"

# Kiểm tra file input
if not os.path.exists(input_file):
    raise FileNotFoundError(f"File '{input_file}' không tồn tại. Hãy kiểm tra lại đường dẫn.")

with open(input_file, "r", encoding="utf-8") as f:
    raw_text = f.read().strip()

if not raw_text:
    raise ValueError("Nội dung trong file 'TamCam.txt' đang rỗng. Hãy thêm văn bản để tóm tắt.")

task_prefix = "summarize: "

# Tokenize văn bản
inputs = tokenizer(
    task_prefix + raw_text,
    return_tensors="pt",
    max_length=1024,
    truncation=True
).to("cuda")  # Đưa input lên GPU nếu có

# Hiển thị token dạng subword và dạng giải mã
tokens = tokenizer.tokenize(task_prefix + raw_text)
decoded_tokens = [tokenizer.decode([tid]) for tid in tokenizer(task_prefix + raw_text, return_tensors="pt")["input_ids"][0]]

print("Tokenized (subword):", tokens)
print("Tokenized (decoded):", decoded_tokens)

# Generate output (tóm tắt văn bản)
output_ids = model.generate(
    inputs["input_ids"],
    max_length=256,         # Độ dài tối đa của kết quả
    min_length=50,          # Độ dài tối thiểu của kết quả
    length_penalty=2.0,     # Hệ số phạt độ dài
    num_beams=4,            # Beam search
    early_stopping=True     # Dừng sớm khi đạt kết quả tốt
)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Xử lý văn bản đầu ra để thêm dòng mới sau dấu chấm
formatted_output = "\n".join([sentence.strip() for sentence in output_text.split('.') if sentence.strip()])

print("Kết quả tóm tắt:")
print(formatted_output)

output_file = "../../outputs/viT5-large_output.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(formatted_output)

print(f"Kết quả đã được ghi vào file: {output_file}")

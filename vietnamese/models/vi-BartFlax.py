from transformers import BartForConditionalGeneration, BartTokenizer

def summarize_text(text, model_name="VietAI/vi-bartflax-large-news"):
    # Load the pre-trained tokenizer and model
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name, from_flax=True)

    # Tokenize the input text
    inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")

    # Generate the summary
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,  # Maximum length of the summary
        min_length=50,   # Minimum length of the summary
        length_penalty=2.0,
        num_beams=1,     # Beam search for better results
        early_stopping=True
    )

    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

# Example usage
if __name__ == "__main__":
    input_text = (
        "Việt Nam đang đối mặt với những thách thức lớn trong lĩnh vực môi trường. Tăng trưởng kinh tế và sự gia tăng dân số đã tạo áp lực lớn lên các nguồn tài nguyên tự nhiên. Chính phủ đã đưa ra nhiều chính sách nhằm giảm thiểu ô nhiễm môi trường và thúc đẩy sử dụng năng lượng tái tạo. Tuy nhiên, việc thực thi các chính sách này vẫn còn gặp nhiều khó khăn."
    )

    summary = summarize_text(input_text)
    print("Tóm tắt:", summary)
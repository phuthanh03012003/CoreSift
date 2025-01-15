import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.modeling_outputs import BaseModelOutput

class TextGeneration:
    
    def __init__(self, model_name="facebook/bart-large-cnn"):
        # Tải mô hình và tokenizer
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)

    def load_inputs(self, input_path):
        # Tải dữ liệu từ Decoder Layer
        self.inputs = torch.load(input_path)

        # ✅ Kiểm tra định dạng dữ liệu đã load
        if isinstance(self.inputs, dict) and 'last_hidden_state' in self.inputs:
            print("✅ Đã tải dữ liệu từ Decoder Layer (last_hidden_state, attention_mask, decoder_input_ids).")
            self.decoder_hidden_states = self.inputs['last_hidden_state']
            self.attention_mask = self.inputs['attention_mask']
            self.decoder_input_ids = self.inputs['decoder_input_ids']
            
            print("🔍 Hidden States:", self.decoder_hidden_states[0, :7])
            print("🔍 Attention Mask:", self.attention_mask[0, :7])
            print("🔍 Decoder Input IDs:", self.decoder_input_ids[0, :7])
        else:
            raise ValueError("❌ Định dạng dữ liệu không hợp lệ. Cần chứa 'last_hidden_state', 'attention_mask', và 'decoder_input_ids'.")

    def generate_default(self):
        # 1️⃣. Mặc định của BART (Beam Search với num_beams=5)
        encoder_outputs = BaseModelOutput(last_hidden_state=self.decoder_hidden_states)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=self.decoder_input_ids,
                attention_mask=self.attention_mask,
                encoder_outputs=encoder_outputs,  # ✅ Thêm encoder_outputs
                max_length=50,
                min_length=10  # ✅ Tránh cảnh báo
            )
        output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print("\n1️⃣. Văn bản sinh ra (Mặc định - Beam Search):\n", output)

    def generate_greedy(self):
        # 2️⃣a. Greedy Search
        encoder_outputs = BaseModelOutput(last_hidden_state=self.decoder_hidden_states)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=self.decoder_input_ids,
                attention_mask=self.attention_mask,
                encoder_outputs=encoder_outputs,  # ✅ Thêm encoder_outputs
                max_length=50,
                num_beams=1
            )
        output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print("\n2️⃣a. Văn bản sinh ra (Greedy Search):\n", output)

    def generate_beam_search(self):
        # 2️⃣b. Beam Search với num_beams=10
        encoder_outputs = BaseModelOutput(last_hidden_state=self.decoder_hidden_states)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=self.decoder_input_ids,
                attention_mask=self.attention_mask,
                encoder_outputs=encoder_outputs,  # ✅ Thêm encoder_outputs
                max_length=50,
                num_beams=10,
                early_stopping=True
            )
        output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print("\n2️⃣b. Văn bản sinh ra (Beam Search num_beams=10):\n", output)

    def generate_top_k_sampling(self):
        # 2️⃣c. Top-k Sampling
        encoder_outputs = BaseModelOutput(last_hidden_state=self.decoder_hidden_states)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=self.decoder_input_ids,
                attention_mask=self.attention_mask,
                encoder_outputs=encoder_outputs,  # ✅ Thêm encoder_outputs
                max_length=50,
                do_sample=True,
                top_k=50
            )
        output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print("\n2️⃣c. Văn bản sinh ra (Top-k Sampling):\n", output)

    def generate_top_p_sampling(self):
        # 2️⃣d. Top-p (Nucleus) Sampling
        encoder_outputs = BaseModelOutput(last_hidden_state=self.decoder_hidden_states)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=self.decoder_input_ids,
                attention_mask=self.attention_mask,
                encoder_outputs=encoder_outputs,  # ✅ Thêm encoder_outputs
                max_length=50,
                do_sample=True,
                top_p=0.9
            )
        output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print("\n2️⃣d. Văn bản sinh ra (Top-p Sampling):\n", output)


if __name__ == "__main__":
    text_gen = TextGeneration()

    # ✅ Load dữ liệu từ Decoder Layer (chứa last_hidden_state, attention_mask, decoder_input_ids)
    text_gen.load_inputs("../object/decoder_outputs_default.pt")

    # 🔥 Thực hiện sinh văn bản với các phương pháp khác nhau
    text_gen.generate_default()       # 1️⃣ Mặc định (Beam Search)
    text_gen.generate_greedy()        # 2️⃣a. Greedy Search
    text_gen.generate_beam_search()   # 2️⃣b. Beam Search num_beams=10
    text_gen.generate_top_k_sampling()  # 2️⃣c. Top-k Sampling
    text_gen.generate_top_p_sampling()  # 2️⃣d. Top-p Sampling

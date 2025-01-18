import torch
import math
import torch.nn as nn
from transformers import ProphetNetForConditionalGeneration, ProphetNetTokenizer
from transformers.modeling_outputs import BaseModelOutput

class EncoderLayer:
    def __init__(self, model_name="microsoft/prophetnet-large-uncased-cnndm"):
        # Tải mô hình và tokenizer của ProphetNet
        self.tokenizer = ProphetNetTokenizer.from_pretrained(model_name)
        self.model = ProphetNetForConditionalGeneration.from_pretrained(model_name)
        self.d_model = self.model.config.hidden_size  # ProphetNet dùng hidden_size thay vì d_model

    def load_inputs(self, input_path):
        # Đọc dữ liệu đầu vào đã tiền xử lý
        self.inputs = torch.load(input_path)

    def embedding_layer(self):
        # 1️. Embedding thuần túy (Chưa thêm Positional Encoding)
        with torch.no_grad():
            # Sử dụng get_input_embeddings() thay cho encoder.embed_tokens
            self.embeddings = self.model.get_input_embeddings()(self.inputs['input_ids'])
        print("\n1️. Embedding thuần túy (Chưa thêm Positional Encoding):")
        print(self.embeddings[0, :7])
        print("Kích thước của Embedding:", self.embeddings.shape)

    def positional_encoding_default(self):
        # 2️a. Positional Encoding Default (Mặc định) - Encoder Layer
        with torch.no_grad():
            self.outputs_auto = self.model.prophetnet.encoder(
                input_ids=self.inputs['input_ids'],
                attention_mask=self.inputs['attention_mask']
            )
        print("\n2️a. Kết quả với Positional Encoding MẶC ĐỊNH (Tự Động):")
        print(self.outputs_auto.last_hidden_state[0, :7])
        print("Kích thước:", self.outputs_auto.last_hidden_state.shape)

    def positional_encoding_custom(self, scaling_factor=0.8):
        # 2️b. Hàm tính Positional Encoding (Tùy chỉnh/Finetune)
        seq_len, d_model = self.embeddings.shape[1], self.embeddings.shape[2]
        pos_encoding = torch.zeros(seq_len, d_model)
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                div_term = 10000 ** (2 * i / d_model)
                pos_encoding[pos, i] = math.sin(pos / div_term) * scaling_factor
                pos_encoding[pos, i + 1] = math.cos(pos / div_term) * scaling_factor
        self.input_with_custom_pe = self.embeddings + pos_encoding.unsqueeze(0)
        print("\n2️b. Embedding + Positional Encoding TỰ TẠO (Finetune):")
        print(self.input_with_custom_pe[0, :7])
        print("Kích thước sau khi thêm Positional Encoding:", self.input_with_custom_pe.shape)

    def encoder_hidden_states(self):
        # 3️a. Kết quả từ Encoder Layer (Hidden States)
        print("\n3️a. Kết quả từ Encoder Layer (Hidden States):")
        print(self.outputs_auto.last_hidden_state[0, :7])
        print("Kích thước:", self.outputs_auto.last_hidden_state.shape)

    def custom_encoder_layer(self):
        # 3️b. Mô phỏng Encoder Layer thủ công
        multi_head_attention = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=8, batch_first=True)
        ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.ReLU(),
            nn.Linear(self.d_model * 4, self.d_model)
        )
        layer_norm1 = nn.LayerNorm(self.d_model)
        layer_norm2 = nn.LayerNorm(self.d_model)

        def encoder_layer(x, mask=None):
            attn_output, _ = multi_head_attention(x, x, x, attn_mask=mask)
            x = layer_norm1(x + attn_output)
            ffn_output = ffn(x)
            x = layer_norm2(x + ffn_output)
            return x

        with torch.no_grad():
            self.encoder_output_manual = encoder_layer(self.input_with_custom_pe)
        print("\n3️b. Kết quả từ Encoder Layer (Thủ công từng bước):")
        print(self.encoder_output_manual[0, :7])
        print("Kích thước:", self.encoder_output_manual.shape)

    def save_encoder_outputs(self):
        # Lưu dạng chuẩn BaseModelOutput để decoder sử dụng đúng cách
        encoder_outputs = BaseModelOutput(last_hidden_state=self.outputs_auto.last_hidden_state)
        torch.save(encoder_outputs, "../object/encoder_outputs.pt")
        torch.save(self.inputs['attention_mask'], "../object/attention_mask.pt")
        print("\nĐã lưu Hidden States và Attention Mask dưới dạng BaseModelOutput.")

if __name__ == "__main__":
    encoder = EncoderLayer()
    encoder.load_inputs("../object/processed_inputs.pt")
    encoder.embedding_layer()
    encoder.positional_encoding_default()
    encoder.positional_encoding_custom()
    encoder.encoder_hidden_states()
    encoder.custom_encoder_layer()
    encoder.save_encoder_outputs()

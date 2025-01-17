import torch
import torch.nn as nn
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers.modeling_outputs import BaseModelOutput

class DecoderLayer:
    def __init__(self, model_name="google/pegasus-cnn_dailymail"):
        # Tải mô hình và tokenizer của PEGASUS
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name)
        self.d_model = self.model.config.d_model

    def load_encoder_outputs(self, encoder_output_path, attention_mask_path):
        # Đọc Hidden States và Attention Mask từ Encoder
        self.encoder_hidden_states = torch.load(encoder_output_path)  # BaseModelOutput
        self.attention_mask = torch.load(attention_mask_path)

        print("\n Đã tải Encoder Outputs (BaseModelOutput) và Attention Mask.")
        print(" Hidden States:", self.encoder_hidden_states.last_hidden_state[0, :7])
        print(" Attention Mask:", self.attention_mask[0, :7])

    def decoding_default(self, save_path="../object/decoder_outputs_default.pt"):
        batch_size = self.encoder_hidden_states.last_hidden_state.shape[0]

        # PEGASUS không có decoder_start_token_id, cần khởi tạo input_ids bằng <pad> token
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.tokenizer.pad_token_id,  # PEGASUS sử dụng pad_token làm start token
            dtype=torch.long
        )

        print(f" Batch size: {batch_size}")
        print(f" Decoder input IDs shape: {decoder_input_ids.shape}")
        print(f" Encoder hidden states shape: {self.encoder_hidden_states.last_hidden_state.shape}")
        print(f" Attention mask shape: {self.attention_mask.shape}")

        # Thực hiện decoding mặc định
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=decoder_input_ids,
                encoder_outputs=self.encoder_hidden_states,
                attention_mask=self.attention_mask,
                max_length=128,
                num_beams=5,  # Beam Search để tăng chất lượng sinh văn bản
                early_stopping=True
            )

        print("\n Kết quả từ Decoder Layer (MẶC ĐỊNH):")
        print(generated_ids)
        print(f" Generated IDs shape: {generated_ids.shape}")

        self.save_decoder_outputs(save_path, decoder_input_ids)

    def decoding_custom(self, save_path="../object/decoder_outputs_custom.pt"):
        batch_size = self.encoder_hidden_states.last_hidden_state.shape[0]

        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.tokenizer.pad_token_id,  # PEGASUS dùng pad_token làm start token
            dtype=torch.long
        )

        # Multi-Head Self-Attention cho Decoder (Masked)
        masked_self_attention = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=8, batch_first=True)

        # Cross-Attention (Kết nối với Encoder)
        cross_attention = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=8, batch_first=True)

        # Feed Forward Network (FFN)
        ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.ReLU(),
            nn.Linear(self.d_model * 4, self.d_model)
        )

        # Layer Normalization
        layer_norm1 = nn.LayerNorm(self.d_model)
        layer_norm2 = nn.LayerNorm(self.d_model)
        layer_norm3 = nn.LayerNorm(self.d_model)

        # Decoder Layer Thủ Công
        def decoder_layer(x, encoder_hidden_states, mask=None):
            # Masked Multi-Head Self-Attention
            attn_output, _ = masked_self_attention(x, x, x, attn_mask=mask)
            x = layer_norm1(x + attn_output)

            # Cross-Attention với Encoder
            cross_attn_output, _ = cross_attention(x, encoder_hidden_states, encoder_hidden_states)
            x = layer_norm2(x + cross_attn_output)

            # Feed Forward Network
            ffn_output = ffn(x)
            x = layer_norm3(x + ffn_output)
            return x

        # Embedding cho Decoder Input
        with torch.no_grad():
            decoder_embeddings = self.model.get_decoder().embed_tokens(decoder_input_ids)

            self.decoder_output_manual = decoder_layer(
                decoder_embeddings,
                self.encoder_hidden_states.last_hidden_state
            )

        print("\n Kết quả từ Decoder Layer (Thủ công từng bước):")
        print(self.decoder_output_manual[0, :7])
        print(f" Kích thước: {self.decoder_output_manual.shape}")

        self.save_decoder_outputs(save_path, decoder_input_ids)

    def save_decoder_outputs(self, path, decoder_input_ids):
        # Lưu kết quả
        data_to_save = {
            "encoder_outputs": self.encoder_hidden_states,
            "attention_mask": self.attention_mask,
            "decoder_input_ids": decoder_input_ids
        }
        torch.save(data_to_save, path)
        print(f"\n Đã lưu kết quả từ Decoder Layer vào {path}")

if __name__ == "__main__":
    decoder = DecoderLayer()
    decoder.load_encoder_outputs("../object/encoder_outputs.pt", "../object/attention_mask.pt")
    decoder.decoding_default()
    decoder.decoding_custom()

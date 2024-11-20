import streamlit as st
import requests
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipConfig
import torch.nn as nn

# Load the pre-trained model and processor
MODEL_DIRECTORY = "blip-image-captioning-base"
MODEL_PATH = os.path.join(MODEL_DIRECTORY)
PROCESSOR_PATH = os.path.join(MODEL_DIRECTORY)

# Check if the model and processor are downloaded
if not os.path.exists(MODEL_PATH) or not os.path.exists(PROCESSOR_PATH):
    st.error("Error: Model and processor not found. Please download the model first.")
else:
    processor = BlipProcessor.from_pretrained(PROCESSOR_PATH)
    config = BlipConfig.from_pretrained(MODEL_PATH)

    # Define a custom model architecture with two additional layers
    class CustomBlipForConditionalGeneration(BlipForConditionalGeneration):
        def __init__(self, config):
            super().__init__(config)
            # Define additional layers
            self.additional_layer1 = nn.Linear(768, 768)  # Adjust the size as needed
            self.additional_layer2 = nn.Linear(768, 768)  # Adjust the size as needed
            # Initialize additional layers
            self.init_weights()

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
        ):
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
            # Apply additional layers
            sequence_output = outputs[0]
            sequence_output = self.additional_layer1(sequence_output)
            sequence_output = nn.functional.relu(sequence_output)
            sequence_output = self.additional_layer2(sequence_output)
            return sequence_output, outputs[1:]

    model = CustomBlipForConditionalGeneration.from_pretrained(MODEL_PATH)

    # Streamlit app
    def main():
        st.title("Image Captioning WebApp")

        # Upload image through Streamlit sidebar
        uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg","jpeg"])

        if uploaded_file is not None:
            # Display the uploaded image on the sidebar
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Perform image captioning
            text_input = st.text_input("Enter text prefix for captioning:", "a photography of")
            conditional_caption_button = st.button("Generate Conditional Caption")

            if conditional_caption_button:
                generate_conditional_caption(image, text_input)

            unconditional_caption_button = st.button("Generate Unconditional Caption")

            if unconditional_caption_button:
                generate_unconditional_caption(image)

    # Function to generate conditional caption
    def generate_conditional_caption(image, text_input):
        inputs = processor(image, text_input, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        st.success(f"Conditional Caption: Generated✅")
        st.subheader(f"→ {caption}")

    # Function to generate unconditional caption
    def generate_unconditional_caption(image):
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        st.success(f"Unconditional Caption: Generated✅")
        st.subheader(f"→ {caption}")

    if __name__ == "__main__":
        main()

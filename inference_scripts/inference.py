from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch
import argparse

# Argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '--model-id',
    dest='model_id',
    default='sovitrath/receipt-ocr-full-ft',
    help='path to local trained model or hf model id'
)
parser.add_argument(
    '--input',
    default='inference_data/image_1.jpeg',
    help='path to the input image'
)
args = parser.parse_args()

model_id = args.model_id

model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    device_map='auto',
    torch_dtype=torch.bfloat16,
    _attn_implementation='flash_attention_2' # Use `flash_attention_2` on Ampere GPUs and above and `eager` on older GPUs.
    # _attn_implementation='eager', # Use `flash_attention_2` on Ampere GPUs and above and `eager` on older GPUs.
)

processor = AutoProcessor.from_pretrained(model_id)

test_image = Image.open(args.input).convert('RGB')

def test(model, processor, image, max_new_tokens=1024, device='cuda'):
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'image'},
                {'type': 'text', 'text': 'OCR this image accurately'}
            ]
        },
    ]
    
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        messages,  # Use the sample without the system message
        add_generation_prompt=True
    )

    image_inputs = []
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    image_inputs.append([image])

    # Prepare the inputs for the model
    model_inputs = processor(
        #text=[text_input],
        text=text_input,
        images=image_inputs,
        return_tensors='pt',
    ).to(device)  # Move inputs to the specified device

    # Generate text with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids, 
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text[0]  # Return the first decoded output text

output = test(model, processor, test_image)
print(output)

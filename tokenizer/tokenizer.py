from transformers import BlipProcessor

def get_tokenizer():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    return processor

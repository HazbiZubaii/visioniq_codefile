from transformers import BlipForQuestionAnswering

def load_model():
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    return model

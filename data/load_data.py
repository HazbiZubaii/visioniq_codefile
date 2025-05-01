from datasets import load_dataset

def load_data():
    # Load a publicly available VQA dataset from HuggingFace
    dataset = load_dataset("facebook/textvqa", split="train")
    print(dataset.column_names)  # See available columns
    print(dataset[0])  # View the first sample
    return dataset


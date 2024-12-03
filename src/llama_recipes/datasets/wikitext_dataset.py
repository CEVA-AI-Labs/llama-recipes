import datasets
def get_wikitext_dataset(dataset_config, tokenizer, split: str = "train"):
    block_size = 2048
    # Load the dataset
    dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    # Define the tokenizer function
    def tokenize_function(examples):
        return tokenizer("".join(examples["text"]), return_tensors="pt")

    # Tokenize the dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

    # Define the function to group texts into blocks
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Apply the grouping function to the tokenized dataset
    lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    return lm_datasets
from transformers import AutoTokenizer, PreTrainedTokenizerFast

# Define the model name or path
model_name = "EleutherAI/pythia-160M"  # You can use any model of your choice

# Load the AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(len(tokenizer))  # 50265
# Add the special token '<IDK>' at the end of the vocabulary
special_token = "<IDK>"
tokenizer.add_tokens(special_token, special_tokens=True)

print(tokenizer.vocab_size)  # 50265 -> 50266
print(len(tokenizer))  # 50265 -> 50266
print(tokenizer.convert_tokens_to_ids(special_token))  # 50265
# Save the updated tokenizer to a new directory
new_dir = "./tokenizers/pythia-idk/"
tokenizer.save_pretrained(new_dir)

print(f"Custom tokenizer saved to {new_dir}")

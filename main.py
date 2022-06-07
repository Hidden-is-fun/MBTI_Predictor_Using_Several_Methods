from transformers import BigBirdTokenizer

tokenizer = BigBirdTokenizer.from_pretrained("BigBird_roBERTa_Base")

print([tokenizer.convert_tokens_to_ids(i)
       for i in tokenizer.tokenize("A text to test tokenizer, This is page [MASK].")])


while True:
    token = input()
    print(tokenizer.decode(list(map(int, token.split(',')))))

Usage example:

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        text_and_image_data = image_and_text_loader(
            NUM_WORKERS, BATCH_SIZE, MAX_TOKEN_LEN, tokenizer)

        for batch_idx, (data, labels) in enumerate(text_and_image_data):

            images = data['image']['raw_image']
            texts = data['text']

            input_token_ids = texts['tokens'].to(device)
            attention_mask = texts['attention_mask'].to(device)

            images, labels = images.to(device), labels.to(device)



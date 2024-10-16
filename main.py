import os
import json
from preprocess import DataLoader
from Specter_Kmeans import TextSummarizer  # Import the TextSummarizer from utils.py
from models.trainT5Large import T5Trainer  # Import T5Trainer class from trainT5Large.py

def save_extracted_summaries(input_file, output_file):
    """Extract summaries from the input file and save them to the output file."""
    # Load the data using DataLoader
    data_loader = DataLoader(input_file)
    content = data_loader.load_data() 

    summarizer = TextSummarizer()
    extracted_summaries = []

    # Adjust this based on your JSON structure
    for item in content:
        # Extract the prompt and completion
        prompt = item['prompt']
        completion = item['completion']

        # Clean and summarize the prompt
        cleaned_text = summarizer.remove_noise(prompt)
        summary_sentences = summarizer.summarize(cleaned_text)
        summary = ' '.join(summary_sentences)

        # Append the pair (extracted prompt, completion) to the summaries list
        extracted_summaries.append({'extracted_prompt': summary, 'completion': completion})

    # Save the extracted summaries
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_summaries, f, ensure_ascii=False, indent=4)

def main():
    # Create output directory if it doesn't exist
    os.makedirs('ext', exist_ok=True)

    # Extractive summarization for train, dev, and test datasets
    save_extracted_summaries('train.json', 'ext/train_ext.json')
    save_extracted_summaries('dev.json', 'ext/dev_ext.json')
    save_extracted_summaries('test.json', 'ext/test_ext.json')

    # Train T5 model using extracted summaries
    train_dataset_path = "ext/train_ext.json"
    val_dataset_path = "ext/dev_ext.json"

    t5_trainer = T5Trainer(train_dataset_path, val_dataset_path)
    t5_trainer.train()  # Call the training method

if __name__ == "__main__":
    main()

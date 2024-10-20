import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, EarlyStoppingCallback
import torch
import evaluate
import numpy as np

class T5Trainer:
    def __init__(self, train_dataset_path, val_dataset_path):
        # Load dataset
        self.train = Dataset.from_json(train_dataset_path)
        self.val = Dataset.from_json(val_dataset_path)

        # Set environment variables for CUDA
        os.environ["CUDA_VISIBLE_DEVICES"] = "6"
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        # Get the Hugging Face access token from the environment variable
        self.access_token = os.getenv("HUGGINGFACE_TOKEN")
        if self.access_token is None:
            raise ValueError("Hugging Face access token not found. Set it as an environment variable.")

        # Load T5 model and tokenizer
        self.checkpoint = "t5-base"
        self.device = torch.device("cuda")

        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, model_max_length=2048)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.checkpoint,
            use_auth_token=self.access_token,
            torch_dtype='auto',
            device_map='auto'
        ).to(self.device)

        # Initialize ROUGE evaluator and data collator
        self.rouge = evaluate.load("rouge")
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.checkpoint)

    def preprocess_function(self, examples):
        prefix = "summarize: "
        inputs = [prefix + doc for doc in examples["extracted_prompt"]]  # Use "summary" field
        model_inputs = self.tokenizer(inputs, max_length=1024, truncation=True, padding=True)
        labels = self.tokenizer(text_target=examples["completion"], max_length=256, truncation=True, padding=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = self.rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    def train(self):
        # Tokenize the datasets
        tokenized_train = self.train.map(self.preprocess_function, batched=True)
        tokenized_val = self.val.map(self.preprocess_function, batched=True)

        # Early stopping
        early_stopping = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0001)

        # Seq2Seq training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir="t5outputBase",
            evaluation_strategy="steps",
            learning_rate=3e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=32,
            weight_decay=0.02,
            warmup_ratio=0.05,
            save_total_limit=3,
            num_train_epochs=8,
            predict_with_generate=True,
            save_steps=400,
            eval_steps=400,
            load_best_model_at_end = True,
            gradient_accumulation_steps = 8,
            logging_dir='./logL',
        )

        # Trainer setup
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[early_stopping],
        )

        # Train the T5 model
        trainer.train()

        # Save the model and tokenizer
        self.model.save_pretrained("./trained_t5_base")
        self.tokenizer.save_pretrained("./trained_t5_base")

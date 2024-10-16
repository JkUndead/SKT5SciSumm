import json

class DataLoader:
    def __init__(self, input_path):
        """
        Initializes the DataLoader with input path.
        
        Args:
            input_path (str): Path to the input JSON file.
        """
        self.input_path = input_path

    def load_data(self):
        """
        Loads data from the input JSON file and processes it.
        
        Returns:
            list: Processed data containing prompts and completions.
        """
        with open(self.input_path, 'r') as f:
            data = json.load(f)

        return self.process_data(data)

    def process_data(self, data):
        """
        Processes the input data to extract prompts and completions.

        Args:
            data (list): The loaded JSON data.

        Returns:
            list: Processed data containing prompts and completions.
        """
        keys = ('prompt', 'completion')
        train_data = []

        for item in data:
            # Get the prompt
            prompt_text = item['abstract']
            ref_abs = ''.join(ref['abstract'] for ref in item['ref_abstract'].values())
            prompt_text += ref_abs.strip('\n')

            # Get the completion
            completion_text = item['related_work']

            # Append to train data
            pair = dict(zip(keys, [prompt_text, completion_text]))
            train_data.append(pair)

        return train_data

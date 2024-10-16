<!-- ## Setup

To run this project, you need to set your Hugging Face access token in an environment variable. You can do this by creating a `.env` file or directly setting the environment variable.

Example `.env` file:

```bash
HUGGINGFACE_TOKEN="your_actual_token_here"

Here’s a comprehensive `README.md` file that includes the specified content along with necessary sections to ensure clarity and usability.  -->

```markdown
# SKT5SciSumm

This is the implementation for **SKT5SciSumm - Revisiting Extractive-Generative Approach for Multi-Document Scientific Summarization** (Accepted at PACLIC 2024). [Link](#) (https://arxiv.org/abs/2402.17311) .

## Requirements

To set up the environment, create a virtual environment and install the necessary packages:

```bash
pip install -r requirements.txt
```

## Dataset

Before running the code, download the **Multi-XScience** dataset. Ensure that the dataset is in the correct format required by the implementation.

## Running the Implementation

After installing the requirements and preparing the dataset, run the main file:

```bash
python main.py
```

This will execute the extractive summarization followed by training the T5 model on the summarized outputs.

## Directory Structure

Your directory should look something like this:

```
project_root/
│
├── main.py
├── models/
│   └── trainT5Large.py
├── preprocess.py
├── utils.py
├── requirements.txt
└── data/
    ├── train.json
    ├── dev.json
    └── test.json
```

## Notes

- Ensure that you have set your Hugging Face access token in an environment variable. You can do this by creating a `.env` file or directly setting the environment variable.

### Example `.env` file:

```bash
HUGGINGFACE_TOKEN="your_actual_token_here"
```

## Additional Information

If you encounter any issues or have questions, feel free to reach out or check the documentation of the libraries used.

---

Happy coding!
```

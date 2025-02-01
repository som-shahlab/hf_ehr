# Uploading models to Hugging Face Hub

## Upload

Upload all pretrained `hf_ehr` models and tokenizers to the Hugging Face via:
```
python hf_upload.py
```

This script will:
1. Create a new Hugging Face config
2. Create a new Hugging Face model
3. Create a new Hugging Face tokenizer
4. Save the model and tokenizer to a local directory
5. Upload the model and tokenizer to Hugging Face Hub

## Usage

An example of how to use a pretrained `hf_ehr` model+tokenizer from Hugging Face to run inference on a patient can be found in `hf_test.py`. Simply run:

```bash
python hf_test.py
```

For every model that we upload to Hugging Face, this script will:
1. Download the model+tokenizer from Hugging Face Hub
2. Tokenize a patient
3. Run inference on the patient
4. Print the patient representation

If the model is not working, it will be printed out as part of the `bad_models` list.
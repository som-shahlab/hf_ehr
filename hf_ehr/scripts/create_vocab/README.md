# Tokenizer

To generate a vocab, run the following scripts in sequence:

```bash
python3 get_numerical_codes.py
python3 create_numerical_vocab.py
python3 create_vocab.py

# Post processing
python3 add_desc_to_codes.py
```

To convert the old CLMBR dictionary to hf_ehr format, run
```
python3 transform_clmbr_tokenizer.py
```
# Training and Evaluation Instructions

To run training and evaluation, follow the steps below:

1. Uncomment the line `trainer.train()` in the `main()` function of `finetune_bert_ner.py` and run:

   ```bash
   python finetune_bert_ner.py
2. Let the model train; loss was best after only the first epoch, so training can be halted there for time purposes.
3. Save the model to the `finetuned_model` directory by uncommenting the last 4 lines of code in `main()` and running:
  
  ```bash
   python finetune_bert_ner.py
```
4. Run the evaluation script:
  ```bash
  python evaluation.py.
```
  * This may take a while!   

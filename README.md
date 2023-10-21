**Discrete-Emotional-Classifier**



**DOWNLOAD**

Start by downloading v1.0.0 on the right side under "Releases". Please download the zip-file which contains all necessary files to apply our fine-tuned transformer model.

**Model description**

This model was fine-tuned on 80% of more than 20,000 sentences from
political communication in six languages (Danish, Dutch, English,
German, Spanish, Swedish). To do so, these sentences have been
crowd-coded by crowdcoders on the platform Prolific. More information
about the crowd-coding and finetuning process can be found in our
[research manuscript](https://doi.org/10.31219/osf.io/m6qkg)

The model classifies text according to the following four variables:
**morality, emotionality, negative, positive**.

Python code is provided below.

The base model is the multilingual
[mDeBERTa-v3-base](https://huggingface.co/microsoft/mdeberta-v3-base)
from Microsoft. This base model has been trained on data including 100
languages. Thus, the fine-tuned model can be used to more languages than
the six languages it has been fine-tuned on.

You can learn more about the efficiency of DeBERTa using ELECTRA-Style
pre-training with Gradient Disentangled Embedding Sharing in their
[research paper](https://arxiv.org/abs/2111.09543). You can learn more
about the CC100 dataset that has been used to train the base model
[here](https://arxiv.org/pdf/1911.02116v2.pdf).

**Eval results**

The model was evaluated using 20% of the test dataset described above.
The metrics described below are macro precision, recall, and F1 scores.

| Category | English   |        |      |         | German    |        |      |         | Danish    |        |      |         | Swedish   |        |      |         | Spanish   |        |      |         | Dutch     |        |      |         |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|          | Precision | Recall | F1   | support | Precision | Recall | F1   | support | Precision | Recall | F1   | support | Precision | Recall | F1   | support | Precision | Recall | F1   | support | Precision | Recall | F1   | support |
| Morality | 0.78      | 0.79   | 0.79 | 985     | 0.71      | 0.70   | 0.70 | 496     | 0.63      | 0.64   | 0.64 | 795     | 0.68      | 0.66   | 0.67 | 796     | 0.75      | 0.75   | 0.75 | 499     | 0.72      | 0.72   | 0.72 | 500     |
| Emotion  | 0.76      | 0.75   | 0.76 | 985     | 0.71      | 0.68   | 0.69 | 496     | 0.67      | 0.62   | 0.63 | 795     | 0.66      | 0.65   | 0.65 | 796     | 0.77      | 0.73   | 0.74 | 499     | 0.70      | 0.64   | 0.66 | 500     |
| Positive | 0.81      | 0.81   | 0.81 | 985     | 0.78      | 0.79   | 0.78 | 496     | 0.71      | 0.68   | 0.69 | 795     | 0.73      | 0.70   | 0.71 | 796     | 0.77      | 0.79   | 0.77 | 499     | 0.74      | 0.73   | 0.74 | 500     |
| Negative | 0.80      | 0.78   | 0.79 | 985     | 0.77      | 0.74   | 0.75 | 496     | 0.73      | 0.70   | 0.71 | 795     | 0.74      | 0.66   | 0.69 | 796     | 0.80      | 0.78   | 0.79 | 499     | 0.75      | 0.70   | 0.72 | 500     |

**Citation**

Please always cite the following paper when using the model:

*Simonsen, K. B., & Widmann, T. (2023). The Politics of Right and Wrong:
Moral Appeals in Political Communication over Six Decades in Ten Western
Democracies. OSF Preprints. <https://doi.org/10.31219/osf.io/m6qkg>*

**Ideas for cooperation or questions?**

If you have questions or ideas for cooperation, contact me at
widmann{at}ps{dot}au{dot}dk

**Model Use**

``` python
#load libraries

# torch libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# pytorch libraries
import pytorch_lightning as pl
from torchmetrics import F1Score
from torchmetrics.functional import accuracy, auroc #F1Score #f1
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# transformers libraries
from transformers import AutoTokenizer, DebertaV2Model, AdamW, get_linear_schedule_with_warmup

from tqdm.auto import tqdm

import pandas as pd
```

``` python
#set working directory to the downloaded folder
import os
os.chdir('/path/to/your/directory')
```

``` python
#define function to apply mDeBERTa model
BASE_MODEL_NAME = "microsoft/mdeberta-v3-base"
LABEL_COLUMNS = ['morality_binary', 'emotion_binary', 'positive_binary', 'negative_binary']

class CrowdCodedTagger(pl.LightningModule):

    def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.bert = DebertaV2Model.from_pretrained(BERT_MODEL_NAME, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCELoss()

    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.last_hidden_state[:, 0])
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outputs):

        labels = []
        predictions = []
        for output in outputs:
          for out_labels in output["labels"].detach().cpu():
            labels.append(out_labels)
          for out_predictions in output["predictions"].detach().cpu():
            predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        for i, name in enumerate(LABEL_COLUMNS):
          class_roc_auc = auroc(predictions[:, i], labels[:, i])
          self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)

    def configure_optimizers(self):

      optimizer = AdamW(self.parameters(), lr=2e-5) #DEFINING LEARNING RATE

      scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=self.n_warmup_steps,
        num_training_steps=self.n_training_steps
      )

      return dict(
        optimizer=optimizer,
        lr_scheduler=dict(
          scheduler=scheduler,
          interval='step'
        )
      )

def process_dataframe_with_transformer(df):
    model = CrowdCodedTagger(n_classes=4)
    model.load_state_dict(torch.load("./model/pytorch_model.pt"), strict = False)
    model.to("cpu") # model.to("cuda")
    model.eval()  # putting model into evaluation mode

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    batch_size = 8

    def predict_labels(df):
        input_text = df['sentence'].tolist() #change 'sentence' depending on the name of column including sentences to classify
        num_inputs = len(input_text)
        num_batches = (num_inputs - 1) // batch_size + 1
        batched_input = [input_text[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
        batched_output = []

        for i, batch in enumerate(tqdm(batched_input)):
            encoded_input = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
            outputs = model(**encoded_input.to("cpu")) #outputs = model(**encoded_input.to("cuda"))

            tensor_values = outputs[1].tolist()
            decimal_numbers = [[num for num in sublist] for sublist in tensor_values]
            output_df = pd.DataFrame(decimal_numbers, columns=LABEL_COLUMNS)
            threshold = 0.5
            threshold_fn = lambda x: 1 if x >= threshold else 0
            output_df = output_df.applymap(threshold_fn)
            input_df = df.iloc[i * batch_size:(i + 1) * batch_size].reset_index(drop=True)
            output_df = pd.concat([input_df, output_df], axis=1)

            batched_output.append(output_df)

        output_df = pd.concat(batched_output, ignore_index=True)
        return output_df

    processed_df = predict_labels(df)
    return processed_df
```

``` python
#provide an input dataframe, for example by loading in a csv file
#by default, the column in the dataframe including sentences to be classified should be called "sentence" (can be adjusted above)
input_df = pd.read_csv("./example_data.csv")
input_df
```

``` python
#apply function to the input_df
result = process_dataframe_with_transformer(input_df)
results
```

``` python
#save results as a csv file in the working directory
result.to_csv('example_results.csv', index=False)
```



**pol_emo_mDeBERTa2: Discrete-Emotional-Classifier**

**DOWNLOAD**

Start by downloading v1.0.0 on the right side under "Releases". Please
download the zip-file which contains all necessary files to apply the
fine-tuned transformer model.

**Model description**

The model is based on the data described in the following paper:

*Widmann, Tobias, and Maximilian Wich. 2022. "Creating and Comparing
Dictionary, Word Embedding, and Transformer-Based Models to Measure
Discrete Emotions in German Political Text." Political Analysis, June,
1--16. <https://doi.org/10.1017/pan.2022.15>.*

However, different to the model described in the paper,
pol_emo_mDeBERTa2 has been trained (80%) and tested (20%) on the whole
set of crowd-coded data: 19620 sentences from political communication.
Furthermore, the base model used is a **multilingual mDeBERTa** trained
on data from 100 languages
([mDeBERTa-v3-base](https://huggingface.co/microsoft/mdeberta-v3-base)
from Microsoft). **Thus, in comparison to the model presented in the paper, the fine-tuned model pol_emo_mDeBERTa2, can be used to more
languages than German.**

You can learn more about the efficiency of DeBERTa using ELECTRA-Style
pre-training with Gradient Disentangled Embedding Sharing in their
[research paper](https://arxiv.org/abs/2111.09543). You can learn more
about the CC100 dataset that has been used to train the base model
[here](https://arxiv.org/pdf/1911.02116v2.pdf).

The model classifies text according to the following eight emotions:
**anger, fear, disgust, sadness, joy, enthusiasm, pride, and hope**.

Python code is provided below.

**Eval results**

The model was evaluated using 20% of the test dataset described in the
research paper. The metrics described below are macro precision, recall,
and F1 scores.

| Emotion    | Precision | Recall | F1 Score |
|------------|-----------|--------|:--------:|
| anger      | 0.782     | 0.775  |  0.778   |
| fear       | 0.725     | 0.664  |  0.689   |
| disgust    | 0.697     | 0.768  |  0.727   |
| sadness    | 0.776     | 0.72   |  0.744   |
| joy        | 0.819     | 0.84   |  0.829   |
| enthusiasm | 0.691     | 0.693  |  0.692   |
| pride      | 0.79      | 0.666  |   0.71   |
| hope       | 0.647     | 0.603  |   0.62   |
| Macro Avg  | 0.741     | 0.716  |  0.724   |

In addition, to test is applicability to other languages than German, I applied the model to more than 1,200 test sentences in English, Spanish, and French. The sentences have been created by an generative AI model, resembling stereotypical statements from political communication (with and without discrete emotional appeals). The performance metrics for this exercise can be seen below. As indicated, the model achieves high performance in languages other than German:

| Emotion    | Precision | Recall | F1 Score |
|------------|-----------|--------|----------|
| anger      | 0.837     | 0.842  | 0.84     |
| fear       | 0.866     | 0.803  | 0.831    |
| disgust    | 0.92      | 0.604  | 0.657    |
| sadness    | 0.876     | 0.88   | 0.878    |
| joy        | 0.863     | 0.963  | 0.906    |
| enthusiasm | 0.782     | 0.622  | 0.663    |
| pride      | 0.95      | 0.852  | 0.894    |
| hope       | 0.91      | 0.707  | 0.768    |
| Macro Avg  | 0.875     | 0.784  | 0.805    |

**Citation**

Please always cite the following paper when using the model:

*Widmann, Tobias, and Maximilian Wich. 2022. "Creating and Comparing
Dictionary, Word Embedding, and Transformer-Based Models to Measure
Discrete Emotions in German Political Text." Political Analysis, June,
1--16. <https://doi.org/10.1017/pan.2022.15>.*

**Ideas for cooperation or questions?**

If you have questions or ideas for cooperation, contact me at
widmann{at}ps{dot}au{dot}dk

**Model Use**

``` python
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

import tqdm

import pandas as pd

```

``` python
#set working directory to the downloaded folder
import os
os.chdir('/path/to/your/directory')
```

``` python
#define function to apply mDeBERTa model
LABEL_COLUMNS = ['anger_v2', 'fear_v2', 'disgust_v2', 'sadness_v2', 'joy_v2', 'enthusiasm_v2', 'pride_v2', 'hope_v2']
BASE_MODEL_NAME = "microsoft/mdeberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
batch_size = 8
device = "cuda" if torch.cuda.is_available() else "cpu"


class CrowdCodedTagger(pl.LightningModule):

  def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None):
    super().__init__()
    self.bert = DebertaV2Model.from_pretrained(BASE_MODEL_NAME, return_dict=True)
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

# Define function for inference
def predict_labels(df):
    input_text = df['sentence'].tolist()
    num_inputs = len(input_text)
    num_batches = (num_inputs - 1) // batch_size + 1
    batched_input = [input_text[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
    batched_output = []

    for i, batch in enumerate(tqdm.tqdm(batched_input)):
        encoded_input = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
        outputs = model(**encoded_input.to(device))

        # Extract the decimal numbers from the tensor
        tensor_values = outputs[1].tolist()
        decimal_numbers = [[num for num in sublist] for sublist in tensor_values]

        # Create Pandas DataFrame
        output_df = pd.DataFrame(decimal_numbers, columns=LABEL_COLUMNS)

        # Apply threshold function to DataFrame
        threshold = 0.65
        threshold_fn = lambda x: 1 if x >= threshold else 0
        output_df = output_df.applymap(threshold_fn)

        # Concatenate input DataFrame with output DataFrame
        input_df = df.iloc[i * batch_size:(i + 1) * batch_size].reset_index(drop=True)
        output_df = pd.concat([input_df, output_df], axis=1)

        batched_output.append(output_df)


    # Concatenate all batches into a single output DataFrame
    output_df = pd.concat(batched_output, ignore_index=True)

    return output_df
```

``` python
#provide an input dataframe, for example by loading in a csv file
#by default, the column in the dataframe including sentences to be classified should be called "sentence" (can be adjusted above)
input_df = pd.read_csv("./example_data.csv")
input_df
```

``` python
# putting model into evaluation mode and load local fine-tuned model
model = CrowdCodedTagger(n_classes=8)
model.load_state_dict(torch.load("./model/pytorch_model.pt"), strict = False)
model.to(device)
model.eval() 
```

``` python
#apply function to the input_df
results = predict_labels(input_df)
print(results)
```

``` python
#save results as a csv file in the working directory
result.to_csv('example_results.csv', index=False)
```

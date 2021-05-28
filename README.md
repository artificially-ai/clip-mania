# Clip Mania
Custom training with OpenAI CLIP; classification tasks; zero-shot examples; and a fully dockerised web-service.

## Development Environment

After cloning this repository, please create a Conda environment as depicted below:

```shell script
conda env create -f environment.yml
```

### Running the Unit Tests

Once the Conda environment has been created and activated, you can have a tast of how the code is behaving by running
the unit tests. Execute the command below:

```shell script
pytest
```

Depending on the amount of tests added over time, the output should look something like this:

```shell script
===================================== test session starts =====================================
platform darwin -- Python 3.8.10, pytest-6.2.4, py-1.10.0, pluggy-0.13.1
rootdir: /Users/wilderrodrigues/dev/git/artificially-ai/clip-mania
collected 7 items

test/test_core/test_clip_model.py ..                                                   [ 28%]
test/test_core/test_executor.py ..                                                     [ 57%]
test/test_dataset/test_fewshot_dataset.py ..                                           [ 85%]
test/test_dataset/test_preprocess.py .                                                 [100%]
===================================== 7 passed in 25.55s =====================================
```

## Training

[wip]

### Preparing the Dataset

To get your custom dataset working you don't have to too much, only preparing the directory structure should be enough.

The directory structure that the system expects is depicted below:

```shell script
-> dataset
   -> train
      -> dog
      -> cat
   -> eval
      -> dog
      -> cat
   -> test
      -> dog
      -> cat
```  

Below you can find a code snippet used to create a `PyTorch` compliant `Dataset`:

```python
from clip_mania.utils.data.datasets import FewShotDataset

dataset = FewShotDataset('/somewhere/dataset/train')
```

### Running Locally

[wip]

### Running with Docker

[wip]

## Web Service for Inference

[wip]

### Running the Web Service Locally

[wip]

### Running the Web Service with Docker

[wip]
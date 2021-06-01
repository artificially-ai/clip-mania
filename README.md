# CLIP Mania
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

To understand how the mode is trained, you can have a look at the [test executor](test/test_core/test_executor.py), which
performs a quick test just to make sure the code is working fine. The same piece of code is used in the training
application.

Before you proceed to train with your dataset, please check below how you have to structure it.

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

Before you start training, please install CLIP Mania (make sure the Conda environment has been created and is activate):

```shell script
pip install .[dev]
```

The command above will also install the development dependencies, which includes `pytest`.

After that, the training can be started with command below:

```shell script
python -m clip_mania.application.train --dataset_path /path/to/dataset/train --model_output_path .  --epochs 5
```

### Running with Docker

[wip]

## Web Service for Inference

[wip]

### Running the Web Service Locally

```shell script
python -m clip_mania.application.inference --test_dataset_path \
          ./test/test_core/dataset/train/airplane/airplane1.jpg --model_path ./clip_mania_model
```

### Running the Web Service with Docker

[wip]
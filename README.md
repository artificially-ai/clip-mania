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

To get your custom dataset working you don't have to do much, only preparing the directory structure should be enough.

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

### Training from the Terminal

Before you start training, please install CLIP Mania (make sure that the Conda environment has been created and is
activate):

```shell script
pip install .[dev]
```

The command above will also install the development dependencies, which includes `pytest`.

After that, the training can be started with command below:

```shell script
python -m clip_mania.application.train --dataset_path /path/to/dataset/train --model_output_path .
```

The name of the model file to be saved is defaulted to 'clip_mania_model.pt' under the given `--model_output_path`.

### Training with Docker

[wip]

## Inference

The repository provides a few ways to infer the classes of unseen images. 

### Using the provided Application

To be able to run the inference on top of your test dataset, the project requires that the dataset is structured in the
following way:

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

For inference we will be using the `test` directory, meaning that its absolute path has to be informed when running
the Inference Application.

The code below should help you to get the inference running. For now, we have no plots and no metrics being displayed.
However, we are logging the tru class, the predicted class and the probability. Soon we will have some better metrics
and plots in place.

```shell script
python -m clip_mania.application.inference --test_dataset_path /path/to/dataset/test --model_path ./clip_mania_model.pt
```

Since we still have neither plots not metrics in place to exhibit the custom model's performance, we added some logging
during inference time to show the current class (based on the image path), the max probability and the predicted class.
Below you can find a snippet of the logging output:

```shell script
I0601 22:21:30.909853 139667949066048 inference.py:59] Image path: ./dataset/coco_crops_few_shot/test/train/000000486571_174704.jpg
I0601 22:21:30.909930 139667949066048 inference.py:60] Max probability is: 0.96728515625
I0601 22:21:30.910000 139667949066048 inference.py:61] Predicted class is: This is a picture of a(n) train.
I0601 22:21:30.944446 139667949066048 inference.py:59] Image path: ./dataset/coco_crops_few_shot/test/train/000000446199_173494.jpg
I0601 22:21:30.944510 139667949066048 inference.py:60] Max probability is: 0.98583984375
I0601 22:21:30.944578 139667949066048 inference.py:61] Predicted class is: This is a picture of a(n) train.
I0601 22:21:30.980922 139667949066048 inference.py:59] Image path: ./dataset/coco_crops_few_shot/test/train/000000520433_173697.jpg
I0601 22:21:30.980986 139667949066048 inference.py:60] Max probability is: 0.73876953125
I0601 22:21:30.981054 139667949066048 inference.py:61] Predicted class is: This is a picture of a(n) train.
I0601 22:21:31.015467 139667949066048 inference.py:59] Image path: ./dataset/coco_crops_few_shot/test/train/000000486372_174120.jpg
I0601 22:21:31.015529 139667949066048 inference.py:60] Max probability is: 0.99072265625
I0601 22:21:31.015597 139667949066048 inference.py:61] Predicted class is: This is a picture of a(n) train.
I0601 22:21:31.050803 139667949066048 inference.py:59] Image path: ./dataset/coco_crops_few_shot/test/train/000000505542_170256.jpg
I0601 22:21:31.050867 139667949066048 inference.py:60] Max probability is: 0.99609375
I0601 22:21:31.050952 139667949066048 inference.py:61] Predicted class is: This is a picture of a(n) train.
I0601 22:21:31.087241 139667949066048 inference.py:59] Image path: ./dataset/coco_crops_few_shot/test/train/000000113488_173542.jpg
I0601 22:21:31.087307 139667949066048 inference.py:60] Max probability is: 0.9951171875
I0601 22:21:31.087392 139667949066048 inference.py:61] Predicted class is: This is a picture of a(n) train.
I0601 22:21:31.123594 139667949066048 inference.py:59] Image path: ./dataset/coco_crops_few_shot/test/train/000000130508_1366964.jpg
I0601 22:21:31.123677 139667949066048 inference.py:60] Max probability is: 0.261962890625
I0601 22:21:31.123716 139667949066048 inference.py:61] Predicted class is: This is a picture of a(n) train.
I0601 22:21:31.157999 139667949066048 inference.py:59] Image path: ./dataset/coco_crops_few_shot/test/train/000000344969_173385.jpg
I0601 22:21:31.158064 139667949066048 inference.py:60] Max probability is: 0.98681640625
I0601 22:21:31.158149 139667949066048 inference.py:61] Predicted class is: This is a picture of a(n) train.
I0601 22:21:31.194153 139667949066048 inference.py:59] Image path: ./dataset/coco_crops_few_shot/test/train/000000563665_174793.jpg
I0601 22:21:31.194217 139667949066048 inference.py:60] Max probability is: 0.984375
I0601 22:21:31.194302 139667949066048 inference.py:61] Predicted class is: This is a picture of a(n) train.
I0601 22:21:31.231364 139667949066048 inference.py:59] Image path: ./dataset/coco_crops_few_shot/test/train/000000012679_170876.jpg
I0601 22:21:31.231429 139667949066048 inference.py:60] Max probability is: 0.765625
I0601 22:21:31.231496 139667949066048 inference.py:61] Predicted class is: This is a picture of a(n) train.
I0601 22:21:31.232240 139667949066048 inference.py:68] Inference completed.
``` 

### Running the Web Service Locally

[wip]

### Running the Web Service with Docker

[wip]

## Zero-Shot on the Custom Model

[wip]
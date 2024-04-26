# Fairness without Sensitive Attributes via Noise and Uncertain Predictions

## Dependencies

```
python 3.10
torch=2.0.1
sklearn
numpy
pandas
tabulate
```


## Training Your Model

To run the code for experiments, use the scripts `train.py`. They have some commandline arguments as listed here:

```
`--run_id`: an user-specified unique ID to ensure that saved results/models don't override each other.

`--epochs`: the number of maximum epochs in training. Since early-stopping is used to prevent overfitting, in actual training the number of epochs could be less than what you specify here.

`--dataset`: the dataset using for classification. Both COMPAS and New Adult are binary classification.

`--signiture`: an optional string that's added to the output file name. Intended to use as some sort of comment.

`--gpu`: whether or not to use GPU in training. If not specified, will use CPU.

`--model_path`: the path to the directory where models will be saved.

`--lr`: the learning rate in training. Default value in each classification task should work.

`--batch_size`: the batch size in training. Default value in each classification task should work.

`--input_size`: input dimension of the model. Default value is for COMPAS classification. For New Adult dataset, default value is 50.

`--hidden_size`: hidden vector dimension of the model. Default value is for COMPAS classification. For New Adult dataset, default value is 32.

`--output_size`: output dimension of the model. Default value is for COMPAS classification. For New Adult dataset, default value is 16.

`--noise_output_size`: learnable noise vector dimension of the model. Default value is for COMPAS classification. For New Adult dataset, default value is 50.
```

An example would be

`python train.py --run_id 1 --epochs 1000 --signiture COMPAS_clf --dataset COMPAS`

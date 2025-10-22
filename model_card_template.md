# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model is a Random Forest Classifier that uses defualt hyperperameters. V 1.0

## Intended Use

This model is intended to be used to predict whether an individual makes more or less than $50k based on factors such as race, sex, martial status and occupation.

## Training Data

The training data was taken from the Census Income dataset, otherwise known as the Adult dataset. It contains data on 48842 individuals from the 1994 Census. It is publicaly available at "https://archive.ics.uci.edu/dataset/20/census+income".

## Evaluation Data

The evaluation data was 30% of the dataset randomly selected using sklearn's `train_test_split` module.

## Metrics

Three metrics were used to evaluate this model. The `precision` is `0.7294`. The `recall` is `0.6279`. The `fbeta` is `0.6749`.

## Ethical Considerations

The data used for the model is publicaly available data that is stripped of all personal identifying information.

## Caveats and Recommendations

The data is from over 30 years ago, trends found in data from that long ago may not match up to current day. Newer data should be added for more accurate results.
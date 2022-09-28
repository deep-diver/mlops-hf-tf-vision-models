![Python](https://img.shields.io/badge/python-3.9-blue.svg) [![TFX](https://img.shields.io/badge/TFX-1.9.1-orange)](https://www.tensorflow.org/tfx)

# MLOps for Vision Models (TensorFlow) from 🤗 Transformers with TensorFlow Extended (TFX)

This repository shows how to build Machine Learning pipeline for a vision model (TensorFlow) from 🤗 Transformers using the TensorFlow Ecosystem. In particular, we use TensorFlow Extended(TFX), and there are TensorFlow Data Validation(TFDV), Transform(TFT), Model Analysis(TFMA), and Serving(TF Serving) besides TensorFlow itself internally involved.

NOTE: This is a follow-up projects of "[Deploying Vision Models (TensorFlow) from 🤗 Transformers](https://github.com/sayakpaul/deploy-hf-tf-vision-models)" which shows how to deploy ViT model locally, on kubernetes, and on a fully managed service Vertex AI.

We will show how to build ML pipeline with TFX in a step-by-step manner:
- [X] **Basic** ( [![CI / Basic](https://github.com/deep-diver/mlops-hf-tf-vision-models/actions/workflows/ci-basic.yml/badge.svg)](https://github.com/deep-diver/mlops-hf-tf-vision-models/actions/workflows/ci-basic.yml) [![CD / Basic](https://github.com/deep-diver/mlops-hf-tf-vision-models/actions/workflows/cd-basic.yml/badge.svg)](https://github.com/deep-diver/mlops-hf-tf-vision-models/actions/workflows/cd-basic.yml) )
  - as the first step, we show how to build ML pipeline with the most basic components, which are `ExampleGen`, `Trainer`, and `Pusher`. These components are responsible for injecting raw dataset into the ML pipeline, training a TensorFlow model, and deploying a trained model.

  <p align="center">
    <img height="300px" src="https://i.ibb.co/h24PB0F/basic.png"/>
  </p>

- [X] **Intermediate** ( [![CI / Intermediate](https://github.com/deep-diver/mlops-hf-tf-vision-models/actions/workflows/ci-intermediate.yml/badge.svg)](https://github.com/deep-diver/mlops-hf-tf-vision-models/actions/workflows/ci-intermediate.yml) [![CD / Intermediate](https://github.com/deep-diver/mlops-hf-tf-vision-models/actions/workflows/cd-intermediate.yml/badge.svg)](https://github.com/deep-diver/mlops-hf-tf-vision-models/actions/workflows/cd-intermediate.yml) )
  - as the second step, we show how to extend the ML pipeline from the first step by adding more components, which are `SchemaGen`, `StatisticsGen`, and `Transform`. These components are responsible for analyzing the structures of the dataset, analyzing the statistical traits of the features in the dataset, and data pre-processing.

- [ ] **Advanced Part 1**: as the third step, we show how to extend the ML pipeline from the second step by adding more components, which are `Resolver` and `Evaluator`. These components are responsible for importing existing Artifacts (such as previously trained model) and comparing the performance between two models (one from the `Resolver` and one from the current pipeline run).

- [ ] **Advanced Part 2**: as the fourth step, we show how to extend the ML pipeline from the third step by adding one more additional component, `Tuner`. This component is responsible for running a set of experiments with different sets of hyperparameters with fewer epochs, and the found best hyperparameter combination will be passed to the `Trainer`, and `Trainer` will train the model longer time with that hyperparameter combinations as the starting point.

- [ ] **🤗 Hub Integration**: in this optional step, we show how to use custom TFX components for 🤗 Hub. In particular, we use `HFModelPusher` to push currently trained model to 🤗 Model Hub and `HFSpacePusher` to automatically deploy Gradio application to 🤗 Space Hub.

- [ ] **Organizing as a standalone Application**: TBD

- [ ] **GitHub Integration**: TBD

## Acknowledgements

We are thankful to the ML Developer Programs team at Google that provided GCP support.

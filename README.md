# machado-Transformer
This repository is a study in Data Science and Deep Learning, based on [Andrej Karpathy's](https://karpathy.ai/) [class on YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY&ab_channel=AndrejKarpathy). The purpose of this project is to create a transformer that generates text in the style of Machado de Assis, a renowned Brazilian writer.

The model has been adapted from Jupyter Notebooks to Python scripts, using data from a Kaggle dataset. Three PyTorch models have been created:

- The first model (`machado_XXX.pt`) has been trained on a compilation of all texts available in the Machado de Assis dataset.
- The second model (`romance_XXX.pt`) has been trained only on Romances from Machado De Assis.
- The third model (`cronicas_XXX.pt`) has been trained on Cronicas from Machado De Assis.

The architecture of the models uses a Generative Transformer with Attention Mechanism, with individual characters from the dataset representing the tokens. The parameters of the architecture can be changed by modifying the `config.py` file, and new training can be initiated using provided or custom text datasets with the `train.py` file.

It is important to note that the generated text can contain hallucinations, which can vary depending on the training time/epochs and the size of the dataset.

To run this code on Google Colab, simply clone the repository using the following command:
```
!git clone https://github.com/JPVercosa/machado-transformer
```

Then is possible to train a new model using: 
```
!python train.py <data_name> <model_name>
```

You can inference with the models that are on the repository with:
```
!python inference.py <data_name> <model_name> <number_of_tokens>
```

In both commands, the `<data_name>` should be a text file with extension, taken from the data directory. If training a new model, `<model_name>` should be the desired output name for the model, including the extension `.pt`. If performing inference, `<model_name>` should be an existing model from the data directory, also with extension.

I'm currently following the convention that the model name is the same as the data on which it was trained, followed by an underscore and an integer representing the number of epochs it was trained for.

Thank you for visiting this repository. Please note that this is a project under development and there may be further updates and improvements to the code in the future. I am planning to add more models and features to enhance the performance of the transformer.

If you have any questions or feedback, please feel free to reach out. Thank you for your interest in this project.
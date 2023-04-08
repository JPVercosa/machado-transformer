# machado-Transformer
This repository is a study in Data Science and Deep Learning, based on Andrej Karpathy's class on YouTube. The purpose of this project is to create a transformer that generates text in the style of Machado de Assis, a renowned Brazilian writer.

The model has been adapted from Jupyter Notebooks to Python scripts, using data from a Kaggle dataset. Three PyTorch models have been created:

- The first model has been trained on a compilation of all texts available in the Machado de Assis dataset.
- The second model has been trained only on Romances from Machado De Assis.
- The third model has been trained on Cronicas from Machado De Assis.

The architecture of the models uses a Generative Transformer with Attention Mechanism, with individual characters from the dataset representing the tokens. The parameters of the architecture can be changed by modifying the config.py file, and new training can be initiated using custom text datasets with the train.py file.

It is important to note that the generated text can contain hallucinations, which can vary depending on the training time/epochs and the size of the dataset.

You can run this code on Google Colab. Just clone the repository with:
'''
!git clone https://github.com/JPVercosa/machado-transformer
'''

Then is possible to train a new model using: 
'''
!python train.py <data_name> <model_name>
'''

You can inference with the models that are on the repository with:
'''
!python inference.py <data_name> <model_name> <number_of_tokens>
'''

Thank you for visiting this repository. If you have any questions or feedback, please feel free to reach out.

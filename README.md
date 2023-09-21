# GPTModel
Building a GPT (Generative Pre-trained Transformer) from scratch

This is an attempt to understand and learn how large language models are built and trained on data. The purpose of this project is purely educational.

Built on following a Youtube [tutorial](https://www.youtube.com/watch?v=UU1WVnMk4E8&t=597s&ab_channel=freeCodeCamp.org) by freecodecamp.

Using the tutorial, I have coded a GPT with multihead attention with 4 heads, and the tranformer architecture of 4 encoders and 4 decoders.

This GPT is trained using __character-level tokenization__ and hence, generates only characters by looking at a specific length of text (known as block). These blocks are stacked together as batches and fed to the model for training.

My GPU supports batch size and block size of 32, and the model is trained for 1000 iterations. At the end, the training and validation loss valued between 2 and 3. On prompting the model with input, it generates maximum of 100 tokens/characters.  
The model is saved as a pickle file (also present in the repository). Model can be loaded directly and further be trained using other hyperparameters, or tested with input prompt.

The dataset used for training was initially __Wizard\_of\_Oz.txt__, but later was trained on a large corpus - [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/). I have added a file known as __Data\_extraction.py__ to generate train, validation datasets and vocabulary that contains unique characters from this corpus.

Since this was done in a virtual environment, I have also added the python module requirements for this project.

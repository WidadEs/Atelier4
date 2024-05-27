# Atelier4
## Part1
### SCRAPPING
#### Cette partie collecte des articles de la BBC en arabe et évalue leur contenu en fonction d'une liste de mots clés pertinents. Les résultats sont enregistrés dans un fichier CSV avec les textes extraits et leurs scores respectifs.
### Text Data Preprocessing
#### This part outlines the steps taken for preprocessing text data in the code. It describes loading data from a CSV file, initializing NLTK tools for tokenization, lemmatization, and removing stopwords in Arabic, as well as splitting the data into training and testing sets.
### Training Neural Network Models
#### This part explains the process of training different neural network models present in the code. It details the configuration of LSTM models for classification and regression, the simple RNN model for regression, and the Bidirectional GRU model for regression. Each model is explained with its layers, optimizer, and associated metrics.
### Performance Evaluation of Models
#### In summary, the LSTM and Bidirectional RNN models with GRU showed better performance in predicting continuous values compared to the simpler RNN model. The slight differences in performance metrics between these models highlight the importance of model architecture and complexity in regression tasks.
## Part2
### Model Description
#### This is the model card for MedSymptomGPT, a model trained to understand and generate medical symptoms based on disease names. It uses the model (GPT2) architecture and has been fine-tuned on a dataset containing various diseases and their corresponding symptoms. The model is intended to assist in generating symptom lists for given diseases, aiding in medical research and educational purposes.

### Training Details
#### Training Data: The model was trained on a dataset containing disease names and their corresponding symptoms. The dataset used for training was QuyenAnhDE/Diseases_Symptoms.
### Training Procedure:
#### Preprocessing: The dataset was preprocessed to combine disease names and symptoms into a single string format.
#### Hyperparameters:
#### Batch size: 8
#### Learning rate: 5e-4
#### Number of epochs: 10

## Part 3
#### This part snippet performs sentiment analysis using the BERT (Bidirectional Encoder Representations from Transformers) model. It begins by loading necessary libraries like Torch and Transformers, as well as importing required modules such as BertModel and BertTokenizer. Data is read from a JSON file containing Amazon fashion reviews, converted into a DataFrame, and filtered to include only rows with valid "reviewText" entries.

#### The text data is tokenized using the BERT tokenizer, and the labels (ratings) are prepared for classification. The dataset is then split into training and evaluation sets using random_split from PyTorch. DataLoader objects are created for efficient batch processing during training and evaluation.

#### A BERT model for sequence classification is initialized, specifying the number of output labels (1 in this case for a regression task). AdamW optimizer is used for training the model with a learning rate of 2e-5. The training loop runs for three epochs, where the model is trained on batches of data, and evaluation loss is computed using the evaluation DataLoader.

# NLP: Salary predictions based on the job description

### Project Goal
To predict the salary level based on the vacancy description.

### Variables Description
* ```name``` - название вакансии
* ```experience``` - опыт работы, требуемый от соискателя [категориальный]
* ```schedule``` - расписание [категориальный]
* ```employment``` - вид занятости [категориальный]
* ```description``` - непосредственно описание вакансии
* ```key_skills``` - ключевые навыки в формате python-списка (много пропусков)
* ```salary``` - целевая переменная (зарплата)

### Project Methodology

There are many NLP models available that can be used for salary prediction based on job description. However, the choice of model will depend on various factors such as the size and complexity of your dataset, the type of prediction task you want to perform, and the computational resources available to you.

Here are a few popular NLP models:

- *Bag-of-Words (BoW):* BoW is a simple and commonly used model for text classification tasks. It represents each document as a bag of its words, ignoring their order but keeping track of their frequency. You can use BoW with a variety of machine learning algorithms such as logistic regression, SVM, or naive Bayes.

- *TF-IDF: Term Frequency-Inverse Document Frequency (TF-IDF)* is another widely used model for text classification tasks. It is similar to BoW, but it weights each word by its inverse document frequency, which helps to downweight common words and upweight rare ones.

- *Word Embeddings:* Word embeddings are dense vector representations of words that capture their semantic and syntactic meaning. You can train your own word embeddings using algorithms like Word2Vec, GloVe, or FastText, or you can use pre-trained embeddings like BERT, RoBERTa, or GPT-2.

- *Transformers:* Transformers are a type of deep neural network architecture that has shown excellent performance in various NLP tasks. Models like BERT, RoBERTa, and GPT-2 are pre-trained transformers that can be fine-tuned for specific downstream tasks, such as salary prediction based on job descriptions.

- *fastText* can also be used for salary prediction based on job descriptions. fastText is an extension of the Word2Vec model that can efficiently train word embeddings for large datasets. It can handle out-of-vocabulary words and can also model subword information.

To use fastText for salary prediction, you can train word embeddings on your job description data using the fastText algorithm. You can then use these embeddings as features in a machine learning model to predict salaries. For example, you can use logistic regression or SVM with fastText embeddings as input features to predict salary ranges.

One advantage of fastText is its efficiency in handling large datasets. It can handle millions or billions of words and can train embeddings quickly, making it a good choice for large-scale salary prediction projects.

Overall, fastText can be a good choice for salary prediction based on job descriptions, especially if you have a large dataset and want to train word embeddings efficiently.

### Data
Link to the dataset is [here](https://drive.google.com/drive/folders/1u-9QPp7107NYN729nisHSnQ9guT8-5kT?usp=sharing).
- ```vacancies_train.csv``` - the training set
- ```vacancies_test.csv``` - the test set
- ```sample_submission.csv``` - a sample submission file in the correct format

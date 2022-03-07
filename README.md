# Arabic-Accent-Classification

### 3 models have been created to predict arabic accent of corpus of documents

-Our data has 18 classes and nearly 458000 documents. 

-Bag of words has been leveraged in order to encode our words since the existence of specific words in a document is the main reason of belonging to a given accent.

-Logistic regression has shown better results than random forest which needed too high depth to start learning.

- A deep learning model has been developed as well which used an embedding layer as the first layer, however due to the high number of unique words made it so big and complex model which caused overfitting.

- Sequential modeling is not the best solution for our case neither word embedding because capturing contextual meaning will not help in distinguishing between accents.




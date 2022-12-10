# %%
import numpy as np 
import pandas as pd

import spacy
nlp = spacy.load('en_core_web_sm')
from nltk.corpus import stopwords
stop = stopwords.words('english')
from textblob import TextBlob

import seaborn as sns
import matplotlib.style as style 
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import set_config
set_config(display="diagram")
import time


# %%
#loading data
df = pd.read_csv("../data/train-balanced-sarcasm.csv")


# %%
#loading embeddings
emb = np.load('../data/user_gcca_embeddings.npz', )

# %%
# emb['ids'][4].decode('UTF-8')[1:-1] in df['author']
l = []
for val in emb['ids']:
    l.append(val.decode('UTF-8')[1:-1])


# %%
userincommon = list(set(l) & set(set(df['author'])))

# %%
df.index = df['author']
uniquedf = df.loc[userincommon]

# %%
embdic = {v: k for k, v in enumerate(l)}

# %%
userindexes = []

for val in uniquedf.iterrows():

    userindexes.append(embdic[val[0]])
uniquedf['userindexes'] = userindexes

# %% [markdown]
# 

X_train,X_test,Y_train,Y_test = train_test_split(uniquedf['comment', 'parent_comment', 'userindexes'],random_state=42)
datasettouse = None
datasettype = 'parent' # can be either parent, or parentresponse
if datasettype == 'parent':
    X_train['text'] = df['parent_comment'].astype(str)
elif datasettype == 'parentresponse':
    X_train['text'] =df['parent_comment'].astype(str)+" "+df['comment'].astype(str)


# %%
# !pip install transformers
from transformers import AutoTokenizer, AutoModel

import random
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig, AutoModelForSequenceClassification


# %%
deviceno = 0
modelname = 'bert-base-uncased'

max_length = 128
batch_size = 8
epochs = 1
import pandas as pd
# df = pd.read_csv('balancedSpaceSep.csv')

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(modelname)

import pandas as pd
import torch
from torch.utils.data import TensorDataset
# Load the dataset into a pandas dataframe.



# Create sentence and label lists
train_sentences = X_train['text'].astype(str)
test_sentences = X_test['text'].astype(str)
train_labels = Y_train
test_labels = Y_test

# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []

# %%

# For every sentence...
input_ids = []
attention_masks = []

for sent in train_sentences:

    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = max_length,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(list(train_labels))
trainuserindexes = torch.tensor(X_train['userindexes'])
labels = labels.to(torch.int64)
trainuserindexes = trainuserindexes.to(torch.int64)
# labels = F.one_hot(labels.to(torch.int64))

train_dataset = TensorDataset(input_ids, attention_masks, labels,trainuserindexes)




# %%

input_ids = []
attention_masks = []

reduceto=len(test_labels)
test_labels = test_labels[:reduceto]
test_sentences = test_sentences[:reduceto]
testuserindexes = torch.tensor(X_train['userindexes'])[:reduceto]
for sent in test_sentences:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 64,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(list(test_labels))
labels = labels.to(torch.int64)
testuserindexes = testuserindexes.to(torch.int64)

# labels = F.one_hot(labels.to(torch.int64))
val_dataset = TensorDataset(input_ids, attention_masks, labels, testuserindexes)


# import torch.nn.functional as F
# # lab = torch.FloatTensor(labels.shape[0], 2)
# # lab.scatter_(1, labels.int() ,1)

# from torch.utils.data import TensorDataset, random_split

# # Combine the training inputs into a TensorDataset.
# dataset = TensorDataset(input_ids, attention_masks, labels)

# # Create a 90-10 train-validation split.

# # Calculate the number of samples to include in each set.
# train_size = int(0.9 * len(dataset))
# val_size = len(dataset) - train_size

# # Divide the dataset by randomly selecting samples.
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# print('{:>5,} training samples'.format(train_size))
# print('{:>5,} validation samples'.format(val_size))
val_dataset


# The DataLoader needs to know our batch size for training, so we specify it 
# here. For fine-tuning BERT on a specific task, the authors recommend a batch 
# size of 16 or 32.

# %%
import torch.nn as nn
class BERTCascade(nn.Module):
    def __init__(self,
                 bert_encoder: nn.Module,
                 user_emb,
                 enc_hid_dim=768, #default embedding size
                 outputs=2,
                 dropout=0.1):
        super().__init__()

        self.bert_encoder = bert_encoder

        self.enc_hid_dim = enc_hid_dim
        self.useremb = nn.Embedding(user_emb.shape[0], user_emb.shape[1])
        self.useremb.weight = user_emb
        
        
        
        ### YOUR CODE HERE ### 
        self.fc1 = nn.Linear(self.enc_hid_dim, self.enc_hid_dim)
        self.fc2 = nn.Linear(self.enc_hid_dim+user_emb.shape[1], outputs)
        self.dropout = nn.Dropout(dropout)




    def forward(self,
                src,
                mask, user_indexes):
        bert_output = self.bert_encoder(src, mask)

        ### YOUR CODE HERE ###
        hidden_state = bert_output.last_hidden_state
        pooled_output = hidden_state[:,0]
        pooled_output = self.fc1(pooled_output)  
        user_emb = self.useremb(user_indexes)
        pooled_output = torch.cat((pooled_output, user_emb), dim=1)
        pooled_output = nn.ReLU()(pooled_output)  
        pooled_output = self.dropout(pooled_output) 
        logits = self.fc2(pooled_output)
        return logits


        

# %%
model2 = AutoModel.from_pretrained(modelname)

# %%
# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
# train_dataset = train_dataset[:int(len(list(train_dataset))/50)]
# model2 = AutoModel.from_pretrained(modelname)
batch_size = 24
train_dataloader = DataLoader(
#             train_dataset,  # The training samples.
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )


# val_dataset =    val_dataset[:int(len(list(val_dataset))/50)]

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'+':'+str(deviceno)
model2 = model2.to(device)
print(device)
model = BERTCascade(model2, nn.Parameter(torch.tensor(emb['G'].astype(np.float32)).to(device)))
model = model.to(device)
# Tell pytorch to run this model on the GPU.
# model.cuda()


optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

from transformers import get_linear_schedule_with_warmup

# Number of training epochs. The BERT authors recommend between 2 and 4. 
# We chose to run for 4, but we'll see later that this may be over-fitting the
# training data.


# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

import numpy as np



# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))





# %%

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
epochs = 1
# Set the seed value all over the place to make this reproducible.
seed_val = 42
import torch.nn.functional as F
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# We'll store a number of quantities such as training and validation loss, 
# validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

# For each epoch...
for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0


    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 1000 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))


        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_userindexes = batch[3].to(device)
   
        optimizer.zero_grad()


        vals = model(b_input_ids, 
                            #  token_type_ids=None, 
                             b_input_mask, 
                            b_userindexes)
        loss = F.cross_entropy(vals, b_labels)
#         loss = vals.loss
#         logits = vals.logits

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        # print(loss, logits, vals)
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)            
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))
        
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

        # Evaluate data for one epoch
    for batch in validation_dataloader:

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_userindexes = batch[3].to(device)


        with torch.no_grad():        


            # (loss, logits) = model(b_input_ids, 
            #                       #  token_type_ids=None, 
            #                        attention_mask=b_input_mask,
            #                        labels=b_labels)
            vals = model(b_input_ids, 
                            #  token_type_ids=None, 
                             b_input_mask, 
        #                              labels=b_labels,
                            b_userindexes)
        #         loss = vals.loss
        #         logits = vals.logits


        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = vals.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)



        # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)

    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

import pandas as pd

# Display floats with two decimal places.
pd.set_option('precision', 2)

# Create a DataFrame from our training statistics.
df_stats = pd.DataFrame(data=training_stats)

# Use the 'epoch' as the row index.
df_stats = df_stats.set_index('epoch')

# A hack to force the column headers to wrap.
#df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

# Display the table.
df_stats

# %%
total_eval_accuracy = 0
total_eval_loss = 0
nb_eval_steps = 0

# Evaluate data for one epoch
model.eval()
for i, batch in enumerate(validation_dataloader):



#
# `batch` contains three pytorch tensors:
#   [0]: input ids 
#   [1]: attention masks
#   [2]: labels 
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)
    b_userindexes = batch[3].to(device)



    with torch.no_grad():        


        # (loss, logits) = model(b_input_ids, 
        #                       #  token_type_ids=None, 
        #                        attention_mask=b_input_mask,
        #                        labels=b_labels)
        vals = model(b_input_ids, 

                         b_input_mask, 

                        b_userindexes)



    # Accumulate the validation loss.
    total_eval_loss += loss.item()

    # Move logits and labels to CPU
    logits = vals.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    # Calculate the accuracy for this batch of test sentences, and
    # accumulate it over all batches.
    pred_flat = np.argmax(logits, axis=1).flatten()
    labels_flat = label_ids.flatten()
    l.append(pred_flat == labels_flat)
    print(total_eval_accuracy/(i+1))

    total_eval_accuracy += flat_accuracy(logits, label_ids)



    # Report the final accuracy for this validation run.
avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)

print("  VALIDATION ACCURACY: {0:.2f}".format(avg_val_accuracy))

# Calculate the average loss over all of the batches.
avg_val_loss = total_eval_loss / len(validation_dataloader)

# Measure how long the validation run took.
validation_time = format_time(time.time() - t0)

print("  Validation Loss: {0:.2f}".format(avg_val_loss))
print("  Validation took: {:}".format(validation_time))

# Record all statistics from this epoch.


# %%

#Train Logistic Regression Model
lr_clf = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2))),('clf',  LogisticRegression(random_state= 42, solver='liblinear'))])
lr_clf.fit(train_sentences,train_labels)
# print(f"The accuracy on the training set is: {lr_clf.score(X_train,Y_train)}")
print(f"The accuracy on the test set is:  {lr_clf.score(test_sentences,test_labels)}")
Y_preds = lr_clf.predict(test_sentences)
equals = test_labels==Y_preds

# %%
first = np.sum((np.vstack((equals, val2)).T==np.array([True, True])).all(axis=1))
second = np.sum((np.vstack((equals, val2)).T==np.array([True, False])).all(axis=1))
third = np.sum((np.vstack((equals, val2)).T==np.array([False, True])).all(axis=1))
fourth = np.sum((np.vstack((equals, val2)).T==np.array([False, False])).all(axis=1))

# %%
# !pip install statsmodels
from statsmodels.stats.contingency_tables import mcnemar
# define contingency table
table = [[first, second],[third, fourth]]
# calculate mcnemar test
result = mcnemar(table, exact=True)
# summarize the finding
print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
# interpret the p-value
alpha = 0.05
if result.pvalue > alpha:
    print('Same proportions of errors (fail to reject H0)')
else:
    print('Different proportions of errors (reject H0)')



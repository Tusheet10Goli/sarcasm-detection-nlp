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
# !pip install ijson
# import ijson
import json
# import eli5

# %%


    

# %%
import numpy as np
import pickle
from collections import defaultdict
import sys, re
import pandas as pd
import json

# noinspection PyCompatibility
from builtins import range

COMMENTS_FILE = "../data/comments.json"
TRAIN_MAP_FILE = "../data/my_train_balanced.csv"
TEST_MAP_FILE = "../data/my_test_balanced.csv"

def build_data_cv(data_folder, cv=10, clean_string=True):
    """
    Loads data
    """
    revs = []

    sarc_train_file = data_folder[0]
    sarc_test_file = data_folder[1]
    
    train_data = np.asarray(pd.read_csv(sarc_train_file, header=None))
    test_data = np.asarray(pd.read_csv(sarc_test_file, header=None))
    print('starting to load json')
#     comments = json.loads(open(COMMENTS_FILE).read())
    f = open('../data/comments0.json')
    comments0 = json.load(f)
    f = open('../data/comments1.json')
    comments1 = json.load(f)
    f = open('../data/comments2.json')
    comments2 = json.load(f)
    f = open('./data/comments3.json')
    comments3 = json.load(f)
    comments  = {}
    comments.update(comments0)
    comments0 = ""
    comments.update(comments1)
    comments1 = ""
    comments.update(comments2)
    comments2 = ""
    comments.update(comments3)
    comments3 = ""
#     vocab = defaultdict(float)


    print('done loading comment json')
    for line in train_data: 
        rev = []
        label_str = line[2]
        if( label_str == 0):
            label = 0
        else:
            label = 1
        rev.append(comments[line[0]]['text'].strip())
        if clean_string:
            orig_rev = clean_str(" ".join(rev))
        else:
            orig_rev = " ".join(rev).lower()
#         words = set(orig_rev.split())
#         for word in words:
#             vocab[word] += 1
        orig_rev = (orig_rev.split())[0:100]
        orig_rev = " ".join(orig_rev)
        datum  = {"y":int(1), 
                  "id":line[0],
                  "text": orig_rev,
                  "author": comments[line[0]]['author'],
                  "topic": comments[line[0]]['subreddit'],
                  "label": label,
                  "num_words": len(orig_rev.split()),
                  "split": int(1)}
        revs.append(datum)
    print('done train')

    for line in test_data:
        rev = []
        label_str = line[2]
        if( label_str == 0):
            label = 0
        else:
            label = 1
        rev.append(comments[line[0]]['text'].strip())
        if clean_string:
            orig_rev = clean_str(" ".join(rev))
        else:
            orig_rev = " ".join(rev).lower()
#         words = set(orig_rev.split())
#         for word in words:
#             vocab[word] += 1
        orig_rev = (orig_rev.split())[0:100]
        orig_rev = " ".join(orig_rev)
        datum  = {"y":int(1),
                  "id": line[0], 
                  "text": orig_rev,  
                  "author": comments[line[0]]['author'],
                  "topic": comments[line[0]]['subreddit'],
                  "label": label,
                  "num_words": len(orig_rev.split()),                      
                  "split": int(0)}
        revs.append(datum)
        
    comments = ""
    return revs


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()



# w2v_file = sys.argv[1]    
data_folder = [TRAIN_MAP_FILE,TEST_MAP_FILE] 
print("loading data...")
revs = build_data_cv(data_folder,  cv=10, clean_string=True)
max_l = np.max(pd.DataFrame(revs)["num_words"])
print("data loaded!")
print("number of sentences: " + str(len(revs)))
# print("vocab size: " + str(len(vocab)))
print("max sentence length: " + str(max_l))
print("loading word2vec vectors...")

print("dataset created!")

# %%
wgcca_embeddings = np.load('../data/user_gcca_embeddings.npz')


# %%
import csv
ids = np.concatenate((np.array(["unknown"]), wgcca_embeddings['ids']), axis=0)
user_embeddings = wgcca_embeddings['G']
unknown_vector = np.random.normal(size=(1,100))
user_embeddings = np.concatenate((unknown_vector, user_embeddings), axis=0)
user_embeddings = user_embeddings.astype(dtype='float32')

wgcca_dict = {}
for i in range(len(ids)):
    wgcca_dict[ids[i]] = int(i)

csv_reader = csv.reader(open("../data/discourse.csv"))
topic_embeddings = []
topic_ids = []
for line in csv_reader:
    topic_ids.append(line[0])
    topic_embeddings.append(line[1:])
topic_embeddings = np.asarray(topic_embeddings)
topic_embeddings_size = len(topic_embeddings[0])
topic_embeddings = topic_embeddings.astype(dtype='float32')
print("topic emb size: ",topic_embeddings_size)

topics_dict = {}
for i in range(len(topic_ids)):
    try:
        topics_dict[topic_ids[i]] = int(i)
    except TypeError:
        print(i)

max_l = 100

x_text = []
author_text_id = []
topic_text_id = []
y = []

test_x = []
test_topic = []
test_author = []
test_y = []

for i in range(len(revs)):
    if revs[i]['split']==1:
        x_text.append(revs[i]['text'])
        try:
            author_text_id.append(wgcca_dict['"'+revs[i]['author']+'"'])
        except KeyError:
            try:
                author_text_id.append(wgcca_dict[revs[i]['author']])
            except KeyError:
                author_text_id.append(0)
        try:
            topic_text_id.append(topics_dict[revs[i]['topic']])
        except KeyError:
            topic_text_id.append(0)
        temp_y = revs[i]['label']
        y.append(temp_y)
    else:
        test_x.append(revs[i]['text'])
        try:
            test_author.append(wgcca_dict['"'+revs[i]['author']+'"'])
        except:
            test_author.append(0)
        try:
            test_topic.append(topics_dict[revs[i]['topic']])
        except:
            test_topic.append(0)
        test_y.append(revs[i]['label'])  

y = np.asarray(y)
test_y = np.asarray(test_y)

# get word indices
# x = []
# for i in range(len(x_text)):
# 	x.append(np.asarray([word_idx_map[word] for word in x_text[i].split()]))
    
# x_test = []
# for i in range(len(test_x)):
#     x_test.append(np.asarray([word_idx_map[word] for word in test_x[i].split()]))

# # padding
# for i in range(len(x)):
#     if( len(x[i]) < max_l ):
#     	x[i] = np.append(x[i],np.zeros(max_l-len(x[i])))		
#     elif( len(x[i]) > max_l ):
#     	x[i] = x[i][0:max_l]
# x = np.asarray(x)

# for i in range(len(x_test)):
#     if( len(x_test[i]) < max_l ):
#         x_test[i] = np.append(x_test[i],np.zeros(max_l-len(x_test[i])))        
#     elif( len(x_test[i]) > max_l ):
#         x_test[i] = x_test[i][0:max_l]
# x_test = np.asarray(x_test)
y_test = test_y

topic_train = np.asarray(topic_text_id)
topic_test = np.asarray(test_topic)
author_train = np.asarray(author_text_id)
author_test = np.asarray(test_author)


# shuffle_indices = np.random.permutation(np.arange(len(y)))
# x_shuffled = x[shuffle_indices]
# y_shuffled = y[shuffle_indices]
# topic_train_shuffled = topic_train[shuffle_indices]
# author_train_shuffled = author_train[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation

# dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
# x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
# topic_train, topic_dev = topic_train_shuffled[:dev_sample_index], topic_train_shuffled[dev_sample_index:]
# author_train, author_dev = author_train_shuffled[:dev_sample_index], author_train_shuffled[dev_sample_index:]
# y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
# print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
# x_train = np.asarray(x_train)
# x_dev = np.asarray(x_dev)
# author_train = np.asarray(author_train)
# author_dev = np.asarray(author_dev)
# topic_train = np.asarray(topic_train)
# topic_dev = np.asarray(topic_dev)
# y_train = np.asarray(y_train)
# y_dev = np.asarray(y_dev)
# word_idx_map["@"] = 0
# rev_dict = {v: k for k, v in word_idx_map.items()}

# %%
# np.sum(topic_train==0), len(topic_train)
# np.sum(author_train==0), len(author_train)
# np.sum(author_train==0), len(author_train)
# # emb['ids'][4].decode('UTF-8')[1:-1] in df['author']
# l = []
# for val in emb['ids']:
#     l.append(val.decode('UTF-8')[1:-1])
# # commentauthors = [comments[key]['author'] for key in comments]
# # commentauthorsunique = np.unique(np.array(list(filter(lambda v: v==v, commentauthors))).astype(str))
# userincommon = list(set(l) & set(set(df['author'])))
# df.index = df['author']
# uniquedf = df.loc[userincommon]
# embdic = {v: k for k, v in enumerate(l)}
# userindexes = []

# for val in uniquedf.iterrows():
# #     print(val[0])
# #     print(val['author'])
# #     embdic[val['author']]
#     userindexes.append(embdic[val[0]])
# uniquedf['userindexes'] = userindexes

# %%
# !pip install transformers
from transformers import AutoTokenizer, AutoModel

import random
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig, AutoModelForSequenceClassification
# topic_train = np.asarray(topic_text_id)
# topic_test = np.asarray(test_topic)
# author_train = np.asarray(author_text_id)
# author_test = np.asarray(test_author)

topic_train.shape, topic_test.shape, author_train.shape, author_test

# %%
deviceno = 0
modelname = 'bert-base-uncased'
#modelname = "cardiffnlp/twitter-roberta-base-offensive"
#modelname = 'microsoft/deberta-base'
# modelname = 'facebook/bart-large'
#modelname = 'unitary/toxic-bert'
max_length = 128
batch_size = 8
epochs = 4
import pandas as pd
# df = pd.read_csv('balancedSpaceSep.csv')

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(modelname)

import pandas as pd
import torch
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.nn.functional as F
# Load the dataset into a pandas dataframe.

# Report the number of sentences.
# print('Number of test sentences: {:,}\n'.format(df.shape[0]))

# Create sentence and label lists



train_sentences = x_text
test_sentences = test_x
train_labels = y
test_labels = test_y

# train_sentences = X_train['text'].astype(str)
# test_sentences = X_test['text'].astype(str)
# train_labels = Y_train
# test_labels = Y_test

# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []

# %%
len(train_labels)# train_labels = train_labels[:int(len(train_labels)/50)]


# %%

# For every sentence...
input_ids = []
attention_masks = []
# train_labels = train_labels[:int(len(train_labels)/50)]
# train_sentences = train_sentences[:int(len(train_sentences)/50)]
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
labels = labels.to(torch.int64)
trainuserindexes = torch.tensor(author_train)
trainuserindexes = trainuserindexes.to(torch.int64)
traintopicindexes = torch.tensor(topic_train)
traintopicindexes = traintopicindexes.to(torch.int64)
# labels = F.one_hot(labels.to(torch.int64))

train_dataset = TensorDataset(input_ids, attention_masks, labels,trainuserindexes, traintopicindexes)




# %%


# %%

input_ids = []
attention_masks = []
reduceto = len(test_labels)
test_labels = test_labels[:reduceto]
test_sentences = test_sentences[:reduceto]
# testuserindexes = torch.tensor(X_train['userindexes'])[:reduceto]
testuserindexes = author_test[:reduceto]
testtopicindexes = topic_test[:reduceto]
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

testuserindexes = torch.tensor(testuserindexes)
testuserindexes = testuserindexes.to(torch.int64)
testtopicindexes = torch.tensor(testtopicindexes)
testtopicindexes = testtopicindexes.to(torch.int64)


# labels = F.one_hot(labels.to(torch.int64))
val_dataset = TensorDataset(input_ids, attention_masks, labels, testuserindexes, testtopicindexes)


val_dataset


# The DataLoader needs to know our batch size for training, so we specify it 
# here. For fine-tuning BERT on a specific task, the authors recommend a batch 
# size of 16 or 32.

# %%

class BERTCascade(nn.Module):
    def __init__(self,
                 bert_encoder: nn.Module,
                 user_emb,
                 topic_emb,
                 enc_hid_dim=768, #default embedding size
                 outputs=2,
                 dropout=0.1):
        super().__init__()

        self.bert_encoder = bert_encoder

        self.enc_hid_dim = enc_hid_dim
        self.useremb = nn.Embedding(user_emb.shape[0], user_emb.shape[1])
        self.useremb.weight = user_emb
        self.topicemb = nn.Embedding(topic_emb.shape[0], topic_emb.shape[1])
        self.topicemb.weight = user_emb
        
        
        ### YOUR CODE HERE ### 
        self.fc1 = nn.Linear(self.enc_hid_dim, self.enc_hid_dim)
        self.fc2 = nn.Linear(self.enc_hid_dim+user_emb.shape[1]+topic_emb.shape[1], outputs)
        self.dropout = nn.Dropout(dropout)




    def forward(self,
                src,
                mask, user_indexes, topic_indexes):
        bert_output = self.bert_encoder(src, mask)

        ### YOUR CODE HERE ###
        hidden_state = bert_output.last_hidden_state
        pooled_output = hidden_state[:,0]
        pooled_output = self.fc1(pooled_output)  
        user_emb = self.useremb(user_indexes)
        topic_emb = self.topicemb(topic_indexes)
        pooled_output = torch.cat((pooled_output, user_emb, topic_emb), dim=1)
        pooled_output = nn.ReLU()(pooled_output)  
        pooled_output = self.dropout(pooled_output) 
        logits = self.fc2(pooled_output)
        return logits


        

# %%
model2 = AutoModel.from_pretrained(modelname)

# %%
batch_size = 50

# %%
# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
# train_dataset = train_dataset[:int(len(list(train_dataset))/50)]
# model2 = AutoModel.from_pretrained(modelname)
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
# model = BERTCascade(model2, nn.Parameter(torch.tensor(emb['G'].astype(np.float32)).to(device)))
torchuseremb = nn.Parameter(torch.tensor(user_embeddings.astype(np.float32)).to(device))
torchtopicemb = nn.Parameter(torch.tensor(topic_embeddings.astype(np.float32)).to(device))
model = BERTCascade(model2, torchuseremb, torchtopicemb)

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

print(device)

# %%

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
epochs = 3
# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# We'll store a number of quantities such as training and validation loss, 
# validation accuracy, and timings.
training_stats = []
epochs = 1
# Measure the total training time for the whole run.
total_t0 = time.time()
l = []
# For each epoch...
for epoch_i in range(0, epochs):
    

    
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
        b_topicindexes = batch[4].to(device)
        # print(b_labels.shape)
        # print(b_input_ids.shape) 
        optimizer.zero_grad()

        vals = model(b_input_ids, 
                            #  token_type_ids=None, 
                             b_input_mask, 
                            b_userindexes,
                            b_topicindexes)
        loss = F.cross_entropy(vals, b_labels)

        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
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
        b_topicindexes = batch[4].to(device)


        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        
            
            vals = model(b_input_ids, 
                            #  token_type_ids=None, 
                             b_input_mask, 
        #                              labels=b_labels,
                            b_userindexes,
                            b_topicindexes)
        #         loss = vals.loss



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



# Display the table.
df_stats

# %%
t0 = time.time()
l = []
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
    b_topicindexes = batch[4].to(device)


    # Tell pytorch not to bother with constructing the compute graph during
    # the forward pass, since this is only needed for backprop (training).
    with torch.no_grad():        

        vals = model(b_input_ids, 
                        #  token_type_ids=None, 
                         b_input_mask, 
    #                              labels=b_labels,
                        b_userindexes,
                        b_topicindexes)
    #         loss = vals.loss



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
    total_eval_accuracy += flat_accuracy(logits, label_ids)


    # Report the final accuracy for this validation run.
avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

# Calculate the average loss over all of the batches.
avg_val_loss = total_eval_loss / len(validation_dataloader)

# Measure how long the validation run took.
validation_time = format_time(time.time() - t0)

# %%
t0 = time.time()
l = []
preds = []
label = []
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
    b_topicindexes = batch[4].to(device)


    # Tell pytorch not to bother with constructing the compute graph during
    # the forward pass, since this is only needed for backprop (training).
    with torch.no_grad():        

        vals = model(b_input_ids, 
                        #  token_type_ids=None, 
                         b_input_mask, 
    #                              labels=b_labels,
                        b_userindexes,
                        b_topicindexes)
    #         loss = vals.loss


    # Accumulate the validation loss.
    total_eval_loss += loss.item()

    # Move logits and labels to CPU
    logits = vals.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    # Calculate the accuracy for this batch of test sentences, and
    # accumulate it over all batches.
    pred_flat = np.argmax(logits, axis=1).flatten()
    labels_flat = label_ids.flatten()
    preds.append(pred_flat)
    label.append(labels_flat)
    l.append(pred_flat == labels_flat)
    total_eval_accuracy += flat_accuracy(logits, label_ids)


    # Report the final accuracy for this validation run.
avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

# Calculate the average loss over all of the batches.
avg_val_loss = total_eval_loss / len(validation_dataloader)

# Measure how long the validation run took.
validation_time = format_time(time.time() - t0)

# %%
preds = np.hstack(preds)
label = np.hstack(label)
from sklearn.metrics import confusion_matrix, plot_confusion_matrix


# %%
cf_matrix = confusion_matrix(label, preds, normalize='all')

import seaborn as sns
ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')

ax.set_title('BERT+CASCADE Response Comment Confusion Matrix\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()

# %%
val2 = np.hstack(l)

lr_clf = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2))),('clf',  LogisticRegression(random_state= 42, solver='liblinear'))])
lr_clf.fit(train_sentences,train_labels)
# print(f"The accuracy on the training set is: {lr_clf.score(X_train,Y_train)}")
print(f"The accuracy for our Logistic Regression model is on the test set is:  {lr_clf.score(test_sentences,test_labels)}")
Y_preds = lr_clf.predict(test_sentences)
equals = test_labels==Y_preds

# %%
first = np.sum((np.vstack((equals, val2)).T==np.array([True, True])).all(axis=1))
second = np.sum((np.vstack((equals, val2)).T==np.array([True, False])).all(axis=1))
third = np.sum((np.vstack((equals, val2)).T==np.array([False, True])).all(axis=1))
fourth = np.sum((np.vstack((equals, val2)).T==np.array([False, False])).all(axis=1))

# %%
table = [[first, second],[third, fourth]]
print('Contingency table', np.array(table)/len(equals))

# %%
# !pip install statsmodels
from statsmodels.stats.contingency_tables import mcnemar
# define contingency table

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

# %%




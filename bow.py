import pandas as pd
import numpy as np
import torch.nn as nn
import torch

##
COMMENT = 'comment_text'
'''
train_comments = pd.read_csv('C:\\Users\\Theo Delemazure\\Documents\\GitHub\\ToxicCommentProject\\data\\clean\\train_comments.csv')[COMMENT].to_numpy()
test_comments = pd.read_csv('C:\\Users\\Theo Delemazure\\Documents\\GitHub\\ToxicCommentProject\\data\\clean\\test_comments.csv')[COMMENT].to_numpy()

train_labels = pd.read_csv('C:\\Users\\Theo Delemazure\\Documents\\GitHub\\ToxicCommentProject\\data\\clean\\train_labels.csv').to_numpy()[:,1:]
test_labels = pd.read_csv('C:\\Users\\Theo Delemazure\\Documents\\GitHub\\ToxicCommentProject\\data\\clean\\test_labels.csv').to_numpy()[:,1:]
'''


comments = pd.read_csv('C:\\Users\\Theo Delemazure\\Documents\\GitHub\\ToxicCommentProject\\data\\clean\\comments.csv')[COMMENT].to_numpy()

labels = pd.read_csv('C:\\Users\\Theo Delemazure\\Documents\\GitHub\\ToxicCommentProject\\data\\clean\\labels.csv').to_numpy()[:,1:]
##


comments_split = [com.split() for com in comments]

##


vocab = dict()
pos = 0
txtidx = []
maxlength = 0

for comment in comments_split:
    isent = []
    maxlength = max(maxlength,len(comment))
    for w in comment:
        if w not in vocab:
            vocab[w] = pos
            pos +=1
        isent.append(vocab[w])
    txtidx.append(torch.LongTensor(list(set(isent))))

print("Number of words : ",len(vocab))
print("Maximum length of comment : ",maxlength)
print("Number of comments :",len(comments_split))
print(txtidx[0])
            
##

vocab_size = len(vocab)
vector_size = 100
embed = nn.Embedding(vocab_size, vector_size)

##
lookup_tensor = torch.tensor([vocab["hello"]], dtype=torch.long)
hello_embed = embed(lookup_tensor)
print(hello_embed)

##

labl = torch.from_numpy(labels).float()

## FIRST MODEL

class CBOW_classifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(CBOW_classifier, self).__init__()
        self.embeddings = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim)
        self.lin = nn.Linear(embedding_dim,6)
        self.activation = nn.Sigmoid()

    def forward(self, inputs):
        out = self.embeddings(inputs)
        out = out.sum(dim=0)
        return self.activation(self.lin(out))
        #return self.lin(out)
        
        
##

classifier = CBOW_classifier(vocab_size=vocab_size,embedding_dim=40)
print(classifier(txtidx[1]))

##

loss_function = nn.MultiLabelSoftMarginLoss()
optimizer = torch.optim.Adagrad(classifier.parameters(),lr=1e-1)

##


total = int(100)
randomidx = list(range(total))
preds = torch.zeros((total,6))

for epoch in range(10):
    total_loss = torch.Tensor([0])
    random.shuffle(randomidx)
    for index in randomidx:
        x = txtidx[index]
        classifier.zero_grad()
        probs = classifier(x)
        loss = loss_function(probs[None,:], labl[index][None,:])
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    print(epoch, total_loss[0])

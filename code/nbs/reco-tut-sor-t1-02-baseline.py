#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
project_name = "reco-tut-sor"; branch = "main"; account = "sparsh-ai"
project_path = os.path.join('/content', project_name)


# In[2]:


if not os.path.exists(project_path):
    get_ipython().system(u'cp /content/drive/MyDrive/mykeys.py /content')
    import mykeys
    get_ipython().system(u'rm /content/mykeys.py')
    path = "/content/" + project_name; 
    get_ipython().system(u'mkdir "{path}"')
    get_ipython().magic(u'cd "{path}"')
    import sys; sys.path.append(path)
    get_ipython().system(u'git config --global user.email "recotut@recohut.com"')
    get_ipython().system(u'git config --global user.name  "reco-tut"')
    get_ipython().system(u'git init')
    get_ipython().system(u'git remote add origin https://"{mykeys.git_token}":x-oauth-basic@github.com/"{account}"/"{project_name}".git')
    get_ipython().system(u'git pull origin "{branch}"')
    get_ipython().system(u'git checkout main')
else:
    get_ipython().magic(u'cd "{project_path}"')


# In[28]:


get_ipython().system(u'git status')


# In[29]:


get_ipython().system(u'git add . && git commit -m \'commit\' && git push origin "{branch}"')


# ---

# In[4]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import confusion_matrix,  accuracy_score
import itertools


# In[5]:


df = pd.read_csv('./data/silver/userdata.csv')
df.head()


# In[6]:


df.info()


# In[7]:


df.describe().round(1)


# In[8]:


fig, ax = plt.subplots(figsize=(18,12))
df.hist(ax=ax)
plt.show()


# ## User Snapshot

# In[9]:


df[df.id == 'e12aeaf2d47d42479ea1c4ac3d8286c6']


# user completed an offer 0b1e... and viewed ae26.... Offer 2906.. had been ignored twice.

# ## Train Embeddings

# In[10]:


def to_categorical(df, columns):
    for col in columns:
        df[col] = pd.Categorical(df[col])
        df[col] = df[col].cat.codes
    return df


# In[11]:


# Set embedding sizes
N = len(df['id'].unique())
M = len(df['offer_id'].unique())

# Set embedding dimension
D = 100


# In[12]:


# Create a neural network
class Model(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, output_dim, layers=[1024], p=0.4):
        super(Model, self).__init__()
        self.N = n_users
        self.M = n_items
        self.D = embed_dim

        self.u_emb = nn.Embedding(self.N, self.D)
        self.m_emb = nn.Embedding(self.M, self.D)
        
        layerlist = []
        n_in = 2 * self.D
        for i in layers:
            layerlist.append(nn.Linear(n_in,i)) 
            layerlist.append(nn.ReLU())
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1],output_dim))
        self.layers = nn.Sequential(*layerlist)
        
        nn.init.xavier_uniform_(self.layers[0].weight)
        nn.init.zeros_(self.layers[0].bias)
        nn.init.xavier_uniform_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, u, m):
        u = self.u_emb(u) # output is (num_samples, D)
        m = self.m_emb(m) # output is (num_samples, D)

        # merge
        out = torch.cat((u, m), 1) # output is (num_samples, 2D)

        x = self.layers(out)
        return x


# In[13]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = Model(N, M, D, output_dim=df['event'].nunique(), layers=[512, 256])
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.08, momentum=0.9)


# In[14]:


df = to_categorical(df, ['id','offer_id'])


# In[15]:


# Convert to tensors
user_ids_t = torch.from_numpy(df['id'].values).long()
offer_ids_t = torch.from_numpy(df['offer_id'].values).long()
ratings_t = torch.from_numpy(df['event'].values).long()


# In[16]:


# Make datasets
N_train = int(0.8 * len(df['event'].values))
N_test = 1000
train_dataset = torch.utils.data.TensorDataset(
    user_ids_t[:N_train],
    offer_ids_t[:N_train],
    ratings_t[:N_train],
)

val_dataset = torch.utils.data.TensorDataset(
    user_ids_t[N_train:-N_test],
    offer_ids_t[N_train:-N_test],
    ratings_t[N_train:-N_test],
)
test_df = df[-N_test:]


# In[17]:


# Data loaders
batch_size = 32
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

validation_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)


# In[18]:


# A function to encapsulate the training loop
def batch_gd(model, criterion, optimizer, train_iter, test_iter, epochs):
    
    train_losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    acc_list = []
    
    for it in range(epochs):
        t0 = datetime.now()
        train_loss = []
        for users, offer, targets in train_loader:
            

            # move data to GPU
            users, offer, targets = users.to(device), offer.to(device), targets.to(device)
            #targets = targets.view(-1, 1).long()
            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(users, offer)
            
            loss = criterion(outputs, targets.squeeze())

            # Backward and optimize
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            
            # Track the accuracy
            total = targets.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == targets).sum().item()
            acc = correct / total
            acc_list.append(acc)

        # Get train loss and test loss
        train_loss = np.mean(train_loss) # a little misleading
        
        val_loss = []
        
        # validation
        with torch.no_grad():
            model.eval()
            
            for users, offer, targets in validation_loader:
                users, offer, targets = users.to(device), offer.to(device), targets.to(device)
                #targets = targets.view(-1, 1).long()
                outputs = model(users, offer)
                loss = criterion(outputs, targets.squeeze())
                val_loss.append(loss.item())
        
        val_loss = np.mean(val_loss)
        # Save losses
        train_losses[it] = train_loss
        val_losses[it] = val_loss

        dt = datetime.now() - t0
        print(f'Epoch {it+1}/{epochs}, Validation Loss: {train_loss:.4f}, '
              f'Test Loss: {val_loss:.4f}, Accuracy: {acc}, Duration: {dt}')

    return train_losses, val_losses


# In[19]:


train_losses, val_losses = batch_gd(model, criterion, optimizer, train_loader, validation_loader, 25)


# In[20]:


# Plot the train loss and test loss per iteration
plt.plot(train_losses, label='train loss')
plt.plot(val_losses, label='test loss')
plt.legend()
plt.show()


# In[21]:


start_ix = 10
end_ix = 20
test_X =  torch.from_numpy(test_df.iloc[start_ix:end_ix]['id'].values).long()
test_y = torch.from_numpy(test_df.iloc[start_ix:end_ix]['event'].values).long()

with torch.no_grad():
    model.to('cpu')
    model.eval()
    pred = model(test_X, test_y)
    print(pred)

_, predicted = torch.max(pred.data, 1)
print(predicted)


# In[22]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# #### Plot confusion matrix and baseline accuracy

# In[23]:


cm = confusion_matrix(test_y, predicted)
classes = [0,1,2]
plot_confusion_matrix(cm, classes)


# In[24]:


print("Accuracy so far: " + str(100*accuracy_score(test_y, predicted))+ "%" )


# Results are decent so far and almost twice better than random quessing.
# 
# #### Show some misclassified examples

# In[25]:


data = test_df.iloc[start_ix:end_ix][['age', 'became_member_on', 'gender', 'id', 'income', 'memberdays', 'event']]#['offer_id'].values
pred_values = pd.DataFrame(predicted, columns=['predicted'], index=data.index)
pd.concat([data, pred_values], axis=1)


# Now let's save the model for future reference

# In[27]:


def save_model(model, model_name, model_info):
    # Save the parameters used to construct the model
    with open(model_name, 'wb') as f:
        torch.save(model_info, f)

    # Save the model parameters
    
    with open(model_name, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)

model_info = {
         'n_users': M, 
          'n_items': N, 
          'embed_dim': D, 
          'output_dim': df['event'].nunique(), 
          'layers': [512, 256], 
          'p': 0.4
    }
save_model(model, './artifacts/models/BaselineModel.pth', model_info)


# During the next step we improve the model to take additional paramenters that describe each user and each offer, which should hopefully, give the model insigths on why a particular customer may like or not like given offer.

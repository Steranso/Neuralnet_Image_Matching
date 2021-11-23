#!/usr/bin/env python
# coding: utf-8

# In[2]:


import urllib.request
urllib.request.urlretrieve('http://thedatadoctor.io/wp-content/uploads/2019/11/lfw.tgz','lfw.tgz')


# In[3]:


get_ipython().system('tar xzf lfw.tgz')


# In[4]:


get_ipython().system('ls -l lfw')


# In[5]:


get_ipython().system('ls -l lfw/Vicente_Fox')


# In[6]:


get_ipython().system('ls -l lfw/Halle_Berry')


# In[7]:


#for question 3
get_ipython().system('ls -l lfw/Brent_Coles')






# In[9]:


from IPython.display import Image
Image(filename = './lfw/Vicente_Fox/Vicente_Fox_0022.jpg')


# In[10]:


get_ipython().system('conda install -y pillow')


# In[12]:


import sys
from IPython.display import Image
from PIL import Image as pilImage
from PIL import ImageMath
from PIL import ImageOps

face1 = pilImage.open('./lfw/Vicente_Fox/Vicente_Fox_0001.jpg')
face2 = pilImage.open('./lfw/Vicente_Fox/Vicente_Fox_0002.jpg')

face1 = ImageOps.equalize(face1).convert('LA')
face2 = ImageOps.equalize(face2).convert('LA')

face1 = ImageOps.fit(face1, (224,224), bleed=0.1, centering=(0.5,0.5))
face2 = ImageOps.fit(face2, (224,224), bleed=0.1, centering=(0.5,0.5))

total_width = face1.size[0] + face2.size[0]
max_height = max(face1.size[1], face2.size[1])

joined_image = pilImage.new('RGB', (total_width, max_height))
joined_image.paste(face1)
joined_image.paste(face2, (face1.size[0], 0))

joined_image.save('example.jpg')
Image('example.jpg')


# In[20]:


from IPython.display import Image
from PIL import Image as pilImage
import sys
import os
import PIL

count_twins = 0
for root, dirs, files in os.walk('./lfw'):
    path =root.split(os.sep)
    
    if (len(files)> 1):
        
        if (len(files)>5):
            files = files[:5]
        for file_i in files:
            for file_j in files:
                count_twins = count_twins +1
                face1 = pilImage.open(root +'/'+ file_i)
                face2 = pilImage.open(root +'/'+ file_j)
                #convert to grayscale (equalize)
                face1 = ImageOps.equalize(face1).convert('LA')
                face2 = ImageOps.equalize(face2).convert('LA')
                
                #fit to a similar frame
                face1 = ImageOps.fit(face1, (224,224), bleed=0.1, centering=(0.5,0.5))
                face2 = ImageOps.fit(face2, (224,224), bleed=0.1, centering=(0.5,0.5))

                total_width = face1.size[0] + face2.size[0]
                max_height = max(face1.size[1], face2.size[1])

                joined_image = pilImage.new('RGB', (total_width, max_height))
                joined_image.paste(face1)
                joined_image.paste(face2, (face1.size[0], 0))

                #save the image
                joined_image.save('./twins/' + str(count_twins) + ".jpg")


# In[13]:


Image('./twins/19300.jpg')


# In[34]:


from IPython.display import Image
from PIL import Image as pilImage
import sys
import os
import PIL
from pathlib import Path
import random

count_diff = 0
pass_count = 0

#19,302 is the number of twins we were able to contruct
#we'll use the same number of differnet faces (non-twins)
while count_diff <= 19302:
    face_a_name = random.choice(os.listdir("./lfw"))
    face_a_file = random.choice(os.listdir("./lfw/" + face_a_name))
    face_a_path = "./lfw/" + face_a_name + "/" + face_a_file
    
    face_b_name = random.choice(os.listdir("./lfw"))
    face_b_file = random.choice(os.listdir("./lfw/" + face_b_name))
    face_b_path = "./lfw/" + face_b_name + '/' + face_b_file
    
    #Do nothing in the case they are the same
    if (face_a_name == face_b_name):
        pass_count = pass_count + 1
    else:
        #Create 
        face1 = pilImage.open(face_b_path)
        face2 = pilImage.open(face_a_path)
        
        #convert to grayscale (equalize)
        face1 = ImageOps.equalize(face1).convert('LA')
        face2 = ImageOps.equalize(face2).convert('LA')
                
        #fit to a similar frame
        #print
        face1 = ImageOps.fit(face1, (224,224), bleed=0.1, centering=(0.5,0.5))
        face2 = ImageOps.fit(face2, (224,224), bleed=0.1, centering=(0.5,0.5))

        total_width = face1.size[0] + face2.size[0]
        max_height = max(face1.size[1], face2.size[1])
        joined_image = PIL.Image.new('RGB', (total_width, max_height))
        joined_image.paste(face1)
        joined_image.paste(face2, (face1.size[0], 0))
        #save the image
        joined_image.save('./diff/' + str(count_diff) + ".jpg")
        count_diff = count_diff + 1
    


# In[14]:


Image('./diff/15.jpg')


# In[16]:


import shutil 
import os
import random

shutil.rmtree("./model/train/")
shutil.rmtree("./model/test/")

get_ipython().system('mkdir model/train')
get_ipython().system('mkdir model/train/twins')
get_ipython().system('mkdir model/train/diff')

get_ipython().system('mkdir model/test')
get_ipython().system('mkdir model/test/twins')
get_ipython().system('mkdir model/test/diff')

all_twin_files = os.listdir('./twins/')
random.shuffle(all_twin_files)
train_twins = all_twin_files[:100]
test_twins = all_twin_files[101:201]

for file in train_twins:
    shutil.copy2('./twins/' + file, "./model/train/twins")
    
for file in test_twins:
    shutil.copy2("./twins/" + file, "./model/test/twins/")
    

all_diff_files = os.listdir('./diff/')
random.shuffle(all_diff_files)
train_diff = all_diff_files[:100]
test_diff = all_diff_files[101:201]

for file in train_diff:
    shutil.copy2('./diff/' + file, "./model/train/diff/")
    
for file in test_twins:
    shutil.copy2("./diff/" + file, "./model/test/diff/")


# In[31]:


import numpy as np 
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import time
import copy
import os
import sys

np.random.seed(1693)
torch.manual_seed(1693)

data_transforms = {
    "train" : transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ]),
    
    'test' : transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
}


data_dir = './model'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                         data_transforms[x])
                 for x in ['train', 'test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = 1, shuffle = True, num_workers = 0)
              for x in ['train', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_name = ['diff', 'twins']

since = time.time()

#=============================
faceNet = models.resnet18(pretrained = True)

faceNet.fc = nn.Linear(in_features = 512, out_features = 2)

faceNet = faceNet.to('cpu')

loss_function = nn.CrossEntropyLoss()

optimizer = optim.SGD(faceNet.parameters(), lr=0.001, momentum =0.9)

faceNet.train()

count_epochs = 0

while(count_epochs < 2):
    running_loss = 0
    running_corrects = 0
    for inputs, labels in dataloaders['train']:
        inputs = inputs.to('cpu')
        labels = labels.to('cpu')
        
        optimizer.zero_grad()
        
        outputs = faceNet(inputs)
        _, preds = torch.max(outputs, 1)
        
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
    count_epochs = count_epochs +1
    epoch_acc = running_corrects.double() / dataset_sizes['train']
    print('Epoch: ')
    print(epoch_acc)
    
torch.save(faceNet.state_dict(), 'bestmodel.torch')


# In[15]:


pred_model = models.resnet18(pretrained =True)
pred_model.fc = nn.Linear(512,2)
pred_model = pred_model.to('cpu')

pred_model.load_state_dict(torch.load('bestmodel.torch'))


# In[17]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pathlib import Path
from matplotlib.pyplot import imshow

face1 = pilImage.open('./lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg')
face2 = pilImage.open('./lfw/Vicente_Fox/Vicente_Fox_0010.jpg')

face1 = ImageOps.equalize(face1).convert('LA')
face2 = ImageOps.equalize(face2).convert('LA')

face1 = ImageOps.fit(face1, (224,224), bleed=0.1, centering=(0.5,0.5))
face2 = ImageOps.fit(face2, (224,224), bleed=0.1, centering=(0.5,0.5))


total_width = face1.size[0] + face2.size[0]
max_height = max(face1.size[1], face2.size[1])

joined_image = pilImage.new('RGB', (total_width, max_height))
joined_image.paste(face1)
#paste image 2 to the right of image 1:
joined_image.paste(face2, (face1.size[0], 0))

imshow(joined_image)

img_trans = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ]) 

cnn_input = img_trans(joined_image).unsqueeze(0)

pred = pred_model(cnn_input)
print(pred.softmax(1))


# In[18]:


import random
from matplotlib.pyplot import imshow

test_image = './diff/' +str(random.randint(1,19000)) +'.jpg'
imshow(pilImage.open(test_image))

cnn_input = img_trans(pilImage.open(test_image)).unsqueeze(0)

pred = pred_model(cnn_input)
print(pred.softmax(1))


# In[41]:


import random
from matplotlib.pyplot import imshow

test_image = './twins/' +str(random.randint(1,19000)) +'.jpg'
imshow(pilImage.open(test_image))

cnn_input = img_trans(pilImage.open(test_image)).unsqueeze(0)

pred = pred_model(cnn_input)
print(pred.softmax(1))


# In[42]:


import urllib.request
urllib.request.urlretrieve('https://thedatadoctor.io/wp-content/uploads/2019/11/SciClone_bestmodel.torch',
                           'SciClone_bestmodel.torch')


# In[43]:


import numpy as np 
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import time
import copy
import os
import sys

pred_model = models.resnet18(pretrained =True)
pred_model.fc = nn.Linear(512,2)
pred_model = pred_model.to('cpu')

pred_model.load_state_dict(torch.load('SciClone_bestmodel.torch'))


# In[44]:


import random
from matplotlib.pyplot import imshow

test_image = './diff/' +str(random.randint(1,19000)) +'.jpg'
imshow(pilImage.open(test_image))


img_trans = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

cnn_input = img_trans(pilImage.open(test_image)).unsqueeze(0)

pred = pred_model(cnn_input)
print(pred.softmax(1))


# In[45]:


import random
from matplotlib.pyplot import imshow

test_image = './twins/' +str(random.randint(1,19000)) +'.jpg'
imshow(pilImage.open(test_image))


img_trans = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

cnn_input = img_trans(pilImage.open(test_image)).unsqueeze(0)

pred = pred_model(cnn_input)
print(pred.softmax(1))


# In[51]:


from IPython.display import Image
Image(filename='./lfw/Halle_Berry/Halle_Berry_0003.jpg')


# In[73]:


image_to_identify = './lfw/Halle_Berry/Halle_Berry_0003.jpg'


from IPython.display import Image
from PIL import Image as pilImage
import os
import sys
import random
from IPython.display import Image
from PIL import Image as pilImage
from PIL import ImageMath
from PIL import ImageOps
import pandas as pd
import shutil

shutil.rmtree('./face_match/')
get_ipython().system('mkdir face_match')

count_faceMatch = 0

records =[]

for root, dirs, files in os.walk('./lfw'):
    path =root.split(os.sep)
    if (len(files)>5):
            files = files[:5]
    for file_i in files:
        if(random.random() > 0.995):
                face1 = pilImage.open(image_to_identify)
                face2 = pilImage.open(root +'/'+ file_i)
                #convert to grayscale (equalize)
                face1 = ImageOps.equalize(face1).convert('LA')
                face2 = ImageOps.equalize(face2).convert('LA')
                #fit to a similar frame
                face1 = ImageOps.fit(face1, (224,224), bleed=0.1, centering=(0.5,0.5))
                face2 = ImageOps.fit(face2, (224,224), bleed=0.1, centering=(0.5,0.5))

                total_width = face1.size[0] + face2.size[0]
                max_height = max(face1.size[1], face2.size[1])

                joined_image = pilImage.new('RGB', (total_width, max_height))
                joined_image.paste(face1)
                
                joined_image.paste(face2, (face1.size[0], 0))

                joined_image.save('./face_match/' + str(count_faceMatch) + ".jpg")
                records.append([str(count_faceMatch), file_i])
                count_faceMatch = count_faceMatch + 1
                

face1 = pilImage.open(image_to_identify)
face2 = pilImage.open('./lfw/Halle_Berry/Halle_Berry_0007.jpg')
face1 = ImageOps.equalize(face1).convert('LA')
face2 = ImageOps.equalize(face2).convert('LA')
face1 = ImageOps.fit(face1, (224,224), bleed=0.1, centering=(0.5,0.5))
face2 = ImageOps.fit(face2, (224,224), bleed=0.1, centering=(0.5,0.5))

total_width = face1.size[0] + face2.size[0]
max_height = max(face1.size[1], face2.size[1])
joined_image = pilImage.new('RGB', (total_width, max_height))
joined_image.paste(face1)  
joined_image.paste(face2, (face1.size[0], 0))

joined_image.save('./face_match/true_match.jpg')
records.append([str(count_faceMatch), 'Halle_Berry_Needle'])

records_Dataframe = pd.DataFrame(records, columns=['ID', 'File Name'])


# In[74]:


#why does this yield 12(?) jpegs when dans only gives 5?
get_ipython().system('ls face_match')


# In[75]:


test_img = './face_match/true_match.jpg'
imshow(pilImage.open(test_img))


# In[68]:


test_img = './face_match/2.jpg'
imshow(pilImage.open(test_img))

cnn_input = img_trans(pilImage.open(test_image)).unsqueeze(0)
pred = pred_model(cnn_input)
print(pred.softmax(1))
print(pred.softmax(1)[0].data.cpu().numpy()[1])


# In[76]:


from IPython.display import Image
from PIL import Image as pilImage
import os
import random
from PIL import ImageMath
from PIL import ImageOps
import pandas as pd
import shutil
import numpy as np 
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import time
import copy


pred_model = models.resnet18(pretrained =True)
pred_model.fc = nn.Linear(512,2)
pred_model = pred_model.to('cpu')

pred_model.load_state_dict(torch.load('SciClone_bestmodel.torch'))

similarities = []

img_trans = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

for root, dirs, files in os.walk('./face_match'):
    path = root.split(os.sep)
    for file_i in files:
        test_img = pilImage.open(root + '/' + file_i)
        cnn_input = img_trans(test_img).unsqueeze(0)
        pred = pred_model(cnn_input)
        similarities.append([file_i, pred.softmax(1)[0].data.cpu().numpy()[1]])
        sys.stdout.write('/r')
        sys.stdout.write('Processing file ' + file_i + '(' +str(len(similarities)) + '/' +str(len(files))+ ')')
        sys.stdout.flush()
        
sim_dataframe = pd.DataFrame(similarities, columns = ['File', 'Similarity'])


# In[77]:


print(sim_dataframe.sort_values(by=['Similarity'], ascending=False))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[79]:


#for question 1
get_ipython().run_line_magic('matplotlib', 'inline')
from pathlib import Path
from matplotlib.pyplot import imshow

face1 = pilImage.open('./lfw/Martin_Sheen/Martin_Sheen_0001.jpg')
face2 = pilImage.open('./lfw/Martin_Short/Martin_Short_0001.jpg')

face1 = ImageOps.equalize(face1).convert('LA')
face2 = ImageOps.equalize(face2).convert('LA')

face1 = ImageOps.fit(face1, (224,224), bleed=0.1, centering=(0.5,0.5))
face2 = ImageOps.fit(face2, (224,224), bleed=0.1, centering=(0.5,0.5))


total_width = face1.size[0] + face2.size[0]
max_height = max(face1.size[1], face2.size[1])

joined_image = pilImage.new('RGB', (total_width, max_height))
joined_image.paste(face1)
#paste image 2 to the right of image 1:
joined_image.paste(face2, (face1.size[0], 0))

imshow(joined_image)

img_trans = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ]) 

cnn_input = img_trans(joined_image).unsqueeze(0)

pred = pred_model(cnn_input)
print(pred.softmax(1))


# In[81]:


#for question 2
import numpy as np 
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import time
import copy
import os
import sys

pred_model = models.resnet18(pretrained =True)
pred_model.fc = nn.Linear(512,2)
pred_model = pred_model.to('cpu')

pred_model.load_state_dict(torch.load('SciClone_bestmodel.torch'))


face1 = pilImage.open('./dennis.jpg')
face2 = pilImage.open('./dennis_old.jpg')

face1 = ImageOps.equalize(face1).convert('LA')
face2 = ImageOps.equalize(face2).convert('LA')

face1 = ImageOps.fit(face1, (224,224), bleed=0.1, centering=(0.5,0.5))
face2 = ImageOps.fit(face2, (224,224), bleed=0.1, centering=(0.5,0.5))


total_width = face1.size[0] + face2.size[0]
max_height = max(face1.size[1], face2.size[1])

joined_image = pilImage.new('RGB', (total_width, max_height))
joined_image.paste(face1)
#paste image 2 to the right of image 1:
joined_image.paste(face2, (face1.size[0], 0))

imshow(joined_image)

img_trans = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ]) 

cnn_input = img_trans(joined_image).unsqueeze(0)

pred = pred_model(cnn_input)
print(pred.softmax(1))


# In[83]:


#for question 5
import numpy as np 
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import time
import copy
import os
import sys

pred_model = models.resnet18(pretrained =True)
pred_model.fc = nn.Linear(512,2)
pred_model = pred_model.to('cpu')

pred_model.load_state_dict(torch.load('SciClone_bestmodel.torch'))


face1 = pilImage.open('./Reveley.jpg')
face2 = pilImage.open('./King_William')

face1 = ImageOps.equalize(face1).convert('LA')
face2 = ImageOps.equalize(face2).convert('LA')

face1 = ImageOps.fit(face1, (224,224), bleed=0.1, centering=(0.5,0.5))
face2 = ImageOps.fit(face2, (224,224), bleed=0.1, centering=(0.5,0.5))


total_width = face1.size[0] + face2.size[0]
max_height = max(face1.size[1], face2.size[1])

joined_image = pilImage.new('RGB', (total_width, max_height))
joined_image.paste(face1)
#paste image 2 to the right of image 1:
joined_image.paste(face2, (face1.size[0], 0))

imshow(joined_image)

img_trans = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ]) 

cnn_input = img_trans(joined_image).unsqueeze(0)

pred = pred_model(cnn_input)
print(pred.softmax(1))


# In[20]:


#for question 7
import numpy as np 
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import time
import copy
import os
import sys

pred_model = models.resnet18(pretrained =True)
pred_model.fc = nn.Linear(512,2)
pred_model = pred_model.to('cpu')

pred_model.load_state_dict(torch.load('SciClone_bestmodel.torch'))


face2 = pilImage.open('./lfw/Martin_Sheen/Martin_Sheen_0001.jpg')
face1 = pilImage.open('./lfw/Halle_Berry/Halle_Berry_0007.jpg')

face1 = ImageOps.equalize(face1).convert('LA')
face2 = ImageOps.equalize(face2).convert('LA')

face1 = ImageOps.fit(face1, (224,224), bleed=0.1, centering=(0.5,0.5))
face2 = ImageOps.fit(face2, (224,224), bleed=0.1, centering=(0.5,0.5))


total_width = face1.size[0] + face2.size[0]
max_height = max(face1.size[1], face2.size[1])

joined_image = pilImage.new('RGB', (total_width, max_height))
joined_image.paste(face1)
#paste image 2 to the right of image 1:
joined_image.paste(face2, (face1.size[0], 0))

imshow(joined_image)

img_trans = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ]) 

cnn_input = img_trans(joined_image).unsqueeze(0)

pred = pred_model(cnn_input)
print(pred.softmax(1))


# In[21]:


#for question 7
import numpy as np 
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import time
import copy
import os
import sys

pred_model = models.resnet18(pretrained =True)
pred_model.fc = nn.Linear(512,2)
pred_model = pred_model.to('cpu')

pred_model.load_state_dict(torch.load('SciClone_bestmodel.torch'))


face1 = pilImage.open('./lfw/Martin_Sheen/Martin_Sheen_0001.jpg')
face2 = pilImage.open('./lfw/Halle_Berry/Halle_Berry_0007.jpg')

face1 = ImageOps.equalize(face1).convert('LA')
face2 = ImageOps.equalize(face2).convert('LA')

face1 = ImageOps.fit(face1, (224,224), bleed=0.1, centering=(0.5,0.5))
face2 = ImageOps.fit(face2, (224,224), bleed=0.1, centering=(0.5,0.5))


total_width = face1.size[0] + face2.size[0]
max_height = max(face1.size[1], face2.size[1])

joined_image = pilImage.new('RGB', (total_width, max_height))
joined_image.paste(face1)
#paste image 2 to the right of image 1:
joined_image.paste(face2, (face1.size[0], 0))

imshow(joined_image)

img_trans = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ]) 

cnn_input = img_trans(joined_image).unsqueeze(0)

pred = pred_model(cnn_input)
print(pred.softmax(1))


# In[ ]:





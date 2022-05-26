import os
import numpy as np
import random
import time
from scipy import ndimage
import torch
import torchvision
import torchvision.transforms
import matplotlib.pyplot as plt
import matplotlib

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


mnist_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()])
mnist_train = torchvision.datasets.MNIST(
        root='./mnist_data', train=True, 
        transform=mnist_transforms, download=True)
mnist_test = torchvision.datasets.MNIST(
        root='./mnist_data', train=False, 
        transform=mnist_transforms, download=True)

bs =32
train_loader = torch.utils.data.DataLoader(
        mnist_train, batch_size=bs, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(
        mnist_test, batch_size=bs, shuffle=True, drop_last=True)

'''
Augment training data with rotated digits
images: training images
labels: training labels
'''
def expand_training_data_normal(data_loader, seq_len=30):

    expanded_seq_images = []
    expanded_seq_labels = []
    p_change = 0.6
    k = 0 # counter
    for batch in data_loader:
        x, y = batch
        batch_size = x.size(0)
        x = x.numpy()
        y = y.numpy()
        k = k+1
        if k%500==0:
            print ('expanding data : %03d' % k)

        bg_value = 0.0 # this is regarded as background's value black
        image = x.reshape(-1, 28, 28)
        label = y.reshape(-1)
        expanded_images = []
        expanded_labels = []
        image_list = [image[k].reshape(28,28) for k in range(image.shape[0])]
        curr_image_idx = 0
        curr_image = image_list[curr_image_idx]
        curr_label = label[curr_image_idx]
        expanded_images.append(curr_image)
        expanded_labels.append(curr_label)
        curr_class_cnt = 1
        for i in range(seq_len):
            # rotate the image with random degree
            # angle = np.random.randint(-90,90,1)
            angle = 30
            new_img = ndimage.rotate(curr_image,angle,reshape=False, cval=bg_value)

            if curr_class_cnt < 2: # ensure each image rotates at least once
                curr_image = new_img
                curr_class_cnt += 1
                # same label
            else:
                if np.random.rand() <= p_change:
                    curr_image_idx += 1
                    curr_image_idx %= batch_size
                    curr_image_idx = int(curr_image_idx)
                    if curr_label != label[curr_image_idx]:
                        new_img = image_list[curr_image_idx] #unrotated
                        for j in range(i+1):
                            new_img = ndimage.rotate(new_img,angle,reshape=False, cval=bg_value)
                        curr_image = new_img  
                        curr_label = label[curr_image_idx]
                        curr_class_cnt = 1
                    else:
                        curr_image = new_img
                        curr_class_cnt += 1
                else:
                    curr_image = new_img
                    curr_class_cnt += 1

            # register new training data
            expanded_images.append(curr_image)
            expanded_labels.append(curr_label)
        expanded_seq_images.append(expanded_images)
        expanded_seq_labels.append(expanded_labels)

    # return them as arrays
    print ('total expanded data : %03d' % k)

    expandedX=np.array(expanded_seq_images)
    expandedY=np.array(expanded_seq_labels)

    # clip
    expandedX[expandedX < 0.] = 0.
    expandedX[expandedX > 1.] = 1.
    
    expandedX = np.expand_dims(expandedX, axis=2)
    print(expandedX.shape, expandedY.shape)
    return expandedX, expandedY

def expand_training_data_balanced(data_loader, seq_len=5):

    expanded_seq_images = []
    expanded_seq_labels = []
    p_change = 0.6
    num_class = 10
    k = 0 # counter
    n_skip = 0
    for batch in data_loader:
        x, y = batch
        batch_size = x.size(0)
        x = x.numpy()
        y = y.numpy()
        k = k+1
        if k%500==0:
            print ('expanding data : %03d' % k)

        bg_value = 0.0 # this is regarded as background's value black
        image = x.reshape(-1, 28, 28)
        label = y.reshape(-1)
        
        skip_batch = False
        for l in range(num_class):
            if not (l in label):
                skip_batch = True
        if skip_batch:
            print('skip batch for lack of all labels')
            n_skip += 1
            continue    

        expanded_images_b = [[] for m in range(num_class)]
        expanded_labels_b = [[] for m in range(num_class)]
        image_list = [image[k].reshape(28,28) for k in range(image.shape[0])]
        curr_image_idx = 0
        curr_image = image_list[curr_image_idx]
        curr_label = label[curr_image_idx]

        start_angle = np.random.choice([0, 20, 40, 60, 80, 100, 120, 140, 160, 180], 1)[0]
        curr_image = ndimage.rotate(curr_image,start_angle,reshape=False, cval=bg_value)

        # keep duplicated seq for each digits
        for b_idx in range(num_class):
            expanded_images_b[b_idx].append(curr_image)
            expanded_labels_b[b_idx].append(curr_label)
        curr_class_cnt = 1
        have_switch = False
        for i in range(seq_len):
            # rotate the image with random degree
            angle = 30

            if have_switch or curr_class_cnt < 2: # ensure each image rotates at least once
                for b_idx in range(num_class):
                    new_image = expanded_images_b[b_idx][-1]
                    new_label = expanded_labels_b[b_idx][-1]
                    new_image = ndimage.rotate(new_image,angle,reshape=False, cval=bg_value)
                    expanded_images_b[b_idx].append(new_image)
                    expanded_labels_b[b_idx].append(new_label)
                curr_class_cnt += 1
                # same label
            else:
                if (np.random.rand() <= p_change) or (i == (seq_len - 1)):
                    have_switch = True
                    same_img_idx = int(expanded_labels_b[b_idx][-1])
                    new_image = expanded_images_b[same_img_idx][-1]
                    new_label = expanded_labels_b[same_img_idx][-1]
                    new_image = ndimage.rotate(new_image,angle,reshape=False, cval=bg_value)
                    expanded_images_b[same_img_idx].append(new_image)
                    expanded_labels_b[same_img_idx].append(new_label)

                    for b_idx in range(num_class):
                        if not (b_idx == same_img_idx):
                            new_label = b_idx
                            expanded_labels_b[b_idx].append(new_label)
                            idx_arr = np.where(np.array(label) == b_idx)[0]
                            idx_chosen = np.random.choice(idx_arr, 1, replace=False)[0]
                            new_image = image_list[int(idx_chosen)]
                            new_image = ndimage.rotate(new_image,start_angle,reshape=False, cval=bg_value)
                            for j in range(i+1):
                                new_image = ndimage.rotate(new_image,angle,reshape=False, cval=bg_value)
                            expanded_images_b[b_idx].append(new_image)
                else:
                    for b_idx in range(num_class):
                        new_image = expanded_images_b[b_idx][-1]
                        new_label = expanded_labels_b[b_idx][-1]
                        new_image = ndimage.rotate(new_image,angle,reshape=False, cval=bg_value)
                        expanded_images_b[b_idx].append(new_image)
                        expanded_labels_b[b_idx].append(new_label)
                    curr_class_cnt += 1

        expanded_seq_images += expanded_images_b
        expanded_seq_labels += expanded_labels_b

    # return them as arrays
    print ('total expanded data : %03d' % k)

    expandedX=np.array(expanded_seq_images)
    expandedY=np.array(expanded_seq_labels)

    # clip
    expandedX[expandedX < 0.] = 0.
    expandedX[expandedX > 1.] = 1.
    
    expandedX = np.expand_dims(expandedX, axis=2)
    print(expandedX.shape, expandedY.shape)
    print(n_skip)
    input('press any key to continue')
    return expandedX, expandedY

def expand_training_data_noswitch(data_loader, seq_len=5):

    expanded_seq_images = []
    expanded_seq_labels = []
    bg_value = 0.0 # this is regarded as background's value black
    angle = 30
    
    for batch in data_loader:
        x, y = batch
        batch_size = x.size(0)
        x = x.numpy()
        y = y.numpy()
        
        image = x.reshape(-1, 28, 28)
        label = y.reshape(-1)

        expanded_images = []
        expanded_labels = np.stack([label]*(seq_len+1),axis=1)
        expanded_images.append(image)
        for i in range(seq_len):
            image = np.stack([ndimage.rotate(image[j],angle,reshape=False, cval=bg_value) for j in range(image.shape[0])])
            expanded_images.append(image)

        expanded_images = np.stack(expanded_images,axis=1)
        
        
        expanded_seq_images.append(expanded_images)
        expanded_seq_labels.append(expanded_labels)

    # return them as arrays

    expandedX=np.concatenate(expanded_seq_images,axis=0)
    expandedY=np.concatenate(expanded_seq_labels,axis=0)

    # clip
    expandedX[expandedX < 0.] = 0.
    expandedX[expandedX > 1.] = 1.
    
    expandedX = np.expand_dims(expandedX, axis=2)
    print(expandedX.shape, expandedY.shape)
    return expandedX, expandedY

def expand_training_data_noswitch_10(data_loader, seq_len=11):

    expanded_seq_images = []
    expanded_seq_labels = []
    bg_value = 0.0 # this is regarded as background's value black
    angle = 15
    num_class = 10
    n_skip = 0

    for batch in data_loader:
        x, y = batch
        batch_size = x.size(0)
        x = x.numpy()
        y = y.numpy()
        
        image = x.reshape(-1, 28, 28)
        label = y.reshape(-1)

        skip_batch = False
        for l in range(num_class):
            if not (l in label):
                skip_batch = True
        if skip_batch:
            print('skip batch for lack of all labels')
            n_skip += 1
            continue
        
        # pick one example from each class
        examples_lbl = []
        examples = []
        for l in range(num_class):
            examples_lbl.append(label[label == l][0])
            examples.append(image[label == l][0])

        image=np.stack(examples,axis=0)
        label=np.stack(examples_lbl,axis=0)


        expanded_images = []
        expanded_labels = np.stack([label]*(seq_len+1),axis=1)
        expanded_images.append(image)
        for i in range(seq_len):
            image = np.stack([ndimage.rotate(image[j],angle,reshape=False, cval=bg_value) for j in range(image.shape[0])])
            expanded_images.append(image)

        expanded_images = np.stack(expanded_images,axis=1)
        
        break

    # return them as arrays
    expandedX = expanded_images
    expandedY = expanded_labels


    # clip
    expandedX[expandedX < 0.] = 0.
    expandedX[expandedX > 1.] = 1.
    
    expandedX = np.expand_dims(expandedX, axis=2)
    print(expandedX.shape, expandedY.shape)
    return expandedX, expandedY

def save_npy(cfilename, data):
    with open(cfilename, mode='wb') as f:
        np.save(f, data)

enable_balanced = True
switch = False
just10 = False # False
dump_dir='./data/rdigit_data/'
if not switch:
    if just10:
        expand_training_data = expand_training_data_noswitch_10
        dump_dir='./data/rdigit_data_noswitch_10/'
    else:
        expand_training_data = expand_training_data_noswitch
        dump_dir='./data/rdigit_data_noswitch/'
else:
    if enable_balanced:
        expand_training_data = expand_training_data_balanced
    else:
        expand_training_data = expand_training_data_normal

train_X, train_Y = expand_training_data(train_loader)
print(np.diff(train_Y))
print(np.sum(np.diff(train_Y) == 0.))
print(np.sum(np.diff(train_Y) != 0.))
print(np.diff(train_Y).shape)

val_X, val_Y = train_X[(-train_X.shape[0]//6):], train_Y[(-train_X.shape[0]//6):]
train_split_X, train_split_Y = train_X[:(-train_X.shape[0]//6)], train_Y[:(-train_X.shape[0]//6)]
test_X, test_Y = expand_training_data(test_loader)

print(np.diff(test_Y))
print(np.sum(np.diff(test_Y) == 0.))
print(np.sum(np.diff(test_Y) != 0.))
print(np.diff(test_Y).shape)

print('dataset size')
print('train', train_split_X.shape, train_split_Y.shape)
print('valid', val_X.shape, val_Y.shape)
print('test', test_X.shape, test_Y.shape)


train_dump_dir = dump_dir + 'train'
if not os.path.exists(train_dump_dir):
    os.makedirs(train_dump_dir)
save_npy(os.path.join(train_dump_dir, 'images_data.npy'), train_split_X)
save_npy(os.path.join(train_dump_dir, 'digits_class.npy'), train_split_Y)

val_dump_dir = dump_dir + 'valid'
if not os.path.exists(val_dump_dir):
    os.makedirs(val_dump_dir)
save_npy(os.path.join(val_dump_dir, 'images_data.npy'), val_X)
save_npy(os.path.join(val_dump_dir, 'digits_class.npy'), val_Y)

test_dump_dir = dump_dir + 'test'
if not os.path.exists(test_dump_dir):
    os.makedirs(test_dump_dir)
save_npy(os.path.join(test_dump_dir, 'images_data.npy'), test_X)
save_npy(os.path.join(test_dump_dir, 'digits_class.npy'), test_Y)

print('finished generating, now plotting visualizations for you')

# visualize testing set
for img_s, lab_s in zip(test_X, test_Y):
    print(lab_s)
    img_s[img_s < 0] = 0
    img_s[img_s > 1] = 1 
    expanded_img = img_s[0]
    for m in range(1, img_s.shape[0]):
        expanded_img = np.concatenate((expanded_img, img_s[m]), axis=2)
    print(expanded_img.shape)
    plt.imshow(np.squeeze(expanded_img, axis=0)*255.,cmap=matplotlib.cm.Greys_r)
    plt.show()
    input()
import matplotlib.pyplot as plt
import numpy as np
import re


# function for getting a value
def get_default(val_name, default):
    name = input('give {}\n'.format(val_name)) or default
    return name


# extract data from file
def get_text():
    filename = get_default('filename', '../checkpoints/test_labelup_train_421_3/loss_log.txt')
    with open(filename, 'r') as f:
        # get all text
        text = f.read()
    return text


# create regex processor
def get_loss_array(text):
    loss = get_default('loss type', 'G_GAN')
    pattern = re.compile(loss + '[:\s\d]*\.[\d]*')
    losses = pattern.findall(text)
    losses = [float(loss.split()[1]) for loss in losses]
    return losses


text = get_text()
l = get_loss_array(text)
plt.plot(l)

# x = 10 * (np.arange(len(losses)) + 1)

num_epochs = 300
plt.title('LabelUpGAN loss over {} epochs'.format(num_epochs))

plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.savefig('421_3_D_loss.png')
#plt.show()

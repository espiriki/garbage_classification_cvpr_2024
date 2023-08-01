#IMPORT ALL THE LIBRARIES
import glob
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoModelForCausalLM
import matplotlib.pyplot as plt

HF_DATASETS_OFFLINE=1  

#SAVE THE NAMES OF ALL IMAGES IN THE BLACK_IMAGE LIST
black_images=[]
root= '/home/chawla/scratch/imgcap_together_data/'
for infile in sorted(glob.glob(root+'*.jpg')):
  name,ext=os.path.split(infile)
  black_images.append(ext)

#SAVE THE DESCRIPTIONS OF ALL IMAGES IN THE BLACK_TEXT LIST
black_text=[]
root= '/home/chawla/scratch/imgcap_together_data/text_files/'
for infile in sorted(glob.glob(root+'*.txt')):
  with open(infile, 'r') as file:
    info = file.read().rstrip('\n')
    black_text.append(info)

#SAVING IMAGE NAME AND DESCRIPTION TO A .CSV FILE
root= '/home/chawla/scratch/imgcap_together_data/'
f=open('/home/chawla/scratch/imgcap_together_data/training/train.csv', "w")

f.write("image"+","+"text"+"\n")

for i in range(len(black_images)):
  f.write(str(black_images[i])+","+str(black_text[i])+"\n")

f.close()

#LOAD DATASET
dataset = load_dataset('csv', data_files=r'/home/chawla/scratch/imgcap_together_data/training/train.csv', split="train")

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        print(item)
        image = Image.open(os.path.join(root,item['image']))
        encoding = self.processor(images=image, text=item["text"], padding="max_length", return_tensors="pt")
        print(encoding)

        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}

        return encoding
    
#TO RUN THIS STEP ON COMPUTE CANADA YOU NEED TO MANUALLY DOWNLOAD AND ADD THE FILES (preprocessor_config.json, special_tokens_map.json, tokenizer_config.json, tokenizer.json, voacb.txt) IN A FOLDER AND GIVE IT'S PATH HERE (link to download the files: https://huggingface.co/microsoft/git-base/tree/main)
processor = AutoProcessor.from_pretrained("/home/chawla/projects/def-rmsouza/chawla/huggingface/pretrained")

train_dataset = ImageCaptioningDataset(dataset, processor)
print(len(train_dataset))

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4)

batch = next(iter(train_dataloader))

processor.decode(batch["input_ids"][0])

MEAN = np.array([123.675, 116.280, 103.530]) / 255
STD = np.array([58.395, 57.120, 57.375]) / 255

unnormalized_image = (batch["pixel_values"][0].numpy() * np.array(STD)[:, None, None]) + np.array(MEAN)[:, None, None]
unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
Image.fromarray(unnormalized_image)

#TO RUN THIS STEP ON COMPUTE CANADA YOU NEED TO MANUALLY DOWNLOAD AND ADD THE FILES (generation_config.json, config.json, model.safetensor) IN A FOLDER AND GIVE IT'S PATH HERE (link to download the files: https://huggingface.co/microsoft/git-base/tree/main)
model = AutoModelForCausalLM.from_pretrained("/home/chawla/projects/def-rmsouza/chawla/huggingface/2nd_folder")

outputs = model(input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["input_ids"])
outputs.loss

idx_plot=[]
loss_plot=[]
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.train()

for epoch in range(2):
  print("Epoch:", epoch)
  for idx, batch in enumerate(train_dataloader):
    input_ids = batch.pop("input_ids").to(device)
    pixel_values = batch.pop("pixel_values").to(device)

    outputs = model(input_ids=input_ids,
                    pixel_values=pixel_values,
                    labels=input_ids)

    loss = outputs.loss

    print("Loss:", loss.item())
    print("Iteration No", idx)

    idx_plot.append(int(idx))
    loss_plot.append(int(loss.item()))

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

#SAVING THE LOSS CURVE
plt.plot(idx_plot,loss_plot)
plt.xlabel('No of iterations')
plt.ylabel('Loss')
plt.title("Loss curve")
plt.savefig('/home/chawla/projects/def-rmsouza/chawla/results/train/loss/loss_curve.png')

#TEST OF RANDOM 10 SAMPLES
x=np.random.randint(2, 1000, size=10)

for i in range(len(x)):
  example = dataset[int(x[i])]
  image = Image.open(os.path.join(root,example['image']))

  inputs = processor(images=image, return_tensors="pt").to(device)
  pixel_values = inputs.pixel_values

  generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
  generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
  print(generated_caption)
  image.save("/home/chawla/projects/def-rmsouza/chawla/results/train/image_caption/"+generated_caption+".jpg")

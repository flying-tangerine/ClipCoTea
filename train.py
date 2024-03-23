import clip
import torch
import torch.nn as nn
import pandas as pd
from dataset import NLIVEdataset
from model import CLIPVEmodel
from utils import train_model, EarlyStopper
from sklearn.model_selection import train_test_split

df = pd.read_json('/Users/ziyixu/SNLI-VE/data/snli_ve_train.jsonl', lines=True)
df = df[['Flickr30K_ID', 'sentence2', 'gold_label']]
df.loc[:, "gold_label"] = df["gold_label"].replace({"entailment": 2, "neutral": 1, "contradiction": 0})
df = df.sample(frac=1).reset_index(drop=True)[:18000]

EPOCHS = 50
BATCH_SIZE = 64 # size of training set is 530,000 128>64
LEARNING_RATE = 1e-5

device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-B/16",device=device,jit=False) #[RN50, ViT-B/16, ViT-L/14], Must set jit=False for training

for k in model.visual.transformer.parameters():
    k.requires_grad=False
    
if device == "cpu":
    model.float()
else :
    clip.model.convert_weights(model)

image_folder = '/Users/ziyixu/SNLI-VE/data/flickr30k_images'
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42) #, shuffle=True
# train_df, val_df = torch.utils.data.random_split(df,[0.8, 0.2])
train_dataset = NLIVEdataset(image_folder, train_df, device, model, preprocess)
val_dataset = NLIVEdataset(image_folder, val_df, device, model, preprocess)

train_dataloader = train_dataset.get_loader(batch_size=BATCH_SIZE)
val_dataloader = val_dataset.get_loader(batch_size = BATCH_SIZE)

# for images, texts, labels in train_dataloader:
#     print(f"image {images}{images.size(1)}, text {texts}{texts.size()}, labels: {labels}{labels.size()}")
custom_model = CLIPVEmodel(model, 512, 256, 128).to(device)
early_stopper = EarlyStopper(patience=3, min_delta=0)

loss = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9,0.98), eps=1e-6, weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
optimizer = torch.optim.Adam(custom_model.parameters(), lr=LEARNING_RATE, weight_decay=0)

train_model(custom_model, train_dataloader, train_dataset, 
            val_dataloader, val_dataset, optimizer, loss, device,
            epochs=EPOCHS, early_stopper=early_stopper)
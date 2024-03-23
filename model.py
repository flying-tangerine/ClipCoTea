import torch
import torch.nn as nn

class CLIPVEmodel(nn.Module):
    # 同时更新clip text encoder& mlp
    def __init__(self, clip, input_dim, hidden_size1, hidden_size2):
        super().__init__()
        self.clip = clip
        self.input_size = input_dim * 4
        output_size = 3
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.bn0 = nn.BatchNorm1d(num_features=self.input_size)
        self.fc1 = nn.Linear(in_features=self.input_size, out_features=hidden_size1)
        self.fc2 = nn.Linear(in_features=hidden_size1, out_features=hidden_size2)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_size2)
        self.fc3 = nn.Linear(in_features=hidden_size2, out_features=output_size)

    def forward(self, image_features, texts): 
        text_features = self.clip.encode_text(texts) #归一化可以实现吗？
        
        image_features = image_features.view(image_features.size(0), -1)
        text_features = text_features.view(text_features.size(0), -1)
        image_features = self.dropout(image_features)
        text_features = self.dropout(text_features)
        
        combined_features = torch.cat([image_features, text_features, torch.abs(image_features - text_features), image_features * text_features], dim=1)

        x = self.dropout(self.bn0(combined_features))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(self.bn1(x))
        output = self.fc3(x)
        # output = F.softmax(x, dim=1)
        return output
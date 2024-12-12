#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import BertTokenizer
from datasets import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

#%%
def preprocess_txt_to_dataframe(file_path):
    """
    Read the txt file and convert it to a DataFrame.

    parameters:
    file_path (str): the path of txt
    
    return:
    pd.DataFrame: a DataFrame containing 'text' and 'sentiment'. 
    """
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        lines = file.readlines()

    texts = []
    sentiments = []
    
    for line in lines:
        if '@' in line:
            text, sentiment = line.rsplit('@', 1)
            text = text.strip()
            sentiment = sentiment.strip() 
            texts.append(text)
            sentiments.append(sentiment)
    
    df = pd.DataFrame({
        'text': texts,
        'sentiment': sentiments
    })
    
    return df

file_path_1 = '../financial_phrasebank/data/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt'
file_path_2 = '../financial_phrasebank/data/FinancialPhraseBank-v1.0/Sentences_75Agree.txt'
dataset_1 = preprocess_txt_to_dataframe(file_path_1)
# dataset_2 = preprocess_txt_to_dataframe(file_path_2)

# dataset = pd.concat([dataset_1, dataset_2], axis = 0)
dataset = dataset_1

dataset.loc[:, "sentiment"][dataset.loc[:, "sentiment"] == "negative"] = 0
dataset.loc[:, "sentiment"][dataset.loc[:, "sentiment"] == "neutral"] = 1
dataset.loc[:, "sentiment"][dataset.loc[:, "sentiment"] == "positive"] = 2

#%%
# Spliting the dataset
train_df, val_df = train_test_split(dataset, test_size=0.2, random_state=42)  # 80% 训练集，20% 验证集

# Converting to HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

# Tokenize datasets
train_tokenized = train_dataset.map(tokenize_function, batched=True)
val_tokenized = val_dataset.map(tokenize_function, batched=True)

# Self-defined PyTorch dataset
class MyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        input_ids = torch.tensor(item['input_ids'])
        attention_mask = torch.tensor(item['attention_mask'])
        sentiment = torch.tensor(item['sentiment'])
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': sentiment}

train_data = MyDataset(train_tokenized)
val_data = MyDataset(val_tokenized)

batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

#%%
class FeedForward(nn.Module):
    def __init__(self, d_model, expansion_factor=4):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_model * expansion_factor)
        self.fc2 = nn.Linear(d_model * expansion_factor, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, norm_type, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)
        self.feed_forward = FeedForward(d_model, expansion_factor=d_ff)
        
        self.norm_type = norm_type
        if norm_type == 'layer':
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        elif norm_type == 'batch':
            self.norm1 = nn.BatchNorm1d(d_model)
            self.norm2 = nn.BatchNorm1d(d_model)
        elif norm_type == 'none':
            self.norm1 = None
            self.norm2 = None
        else:
            raise ValueError("Invalid norm_type. Choose 'layer', 'batch', or 'none'.")
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        # Initialize Multihead Attention weights
        nn.init.kaiming_normal_(self.attention.in_proj_weight, mode='fan_in', nonlinearity='linear') 
        nn.init.kaiming_normal_(self.attention.out_proj.weight, mode='fan_in', nonlinearity='linear')
        
        # Initialize feed-forward network weights
        nn.init.kaiming_normal_(self.feed_forward.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.feed_forward.fc2.weight, mode='fan_in', nonlinearity='relu')
        
        # Initialize LayerNorm weights if using LayerNorm
        if self.norm1 is not None:
            nn.init.constant_(self.norm1.weight, 1)
            nn.init.constant_(self.norm1.bias, 0)
        if self.norm2 is not None:
            nn.init.constant_(self.norm2.weight, 1)
            nn.init.constant_(self.norm2.bias, 0)

    def forward(self, x, mask):
        # Self-attention
        attention_out, _ = self.attention(x, x, x, attn_mask=mask)
        attention_out = self.dropout1(attention_out)
        
        # Apply normalization if not 'none'
        if self.norm1 is not None:
            if isinstance(self.norm1, nn.LayerNorm):
                out1 = self.norm1(x + attention_out)
            elif isinstance(self.norm1, nn.BatchNorm1d):
                out1 = self.norm1((x + attention_out).transpose(1, 2)).transpose(1, 2)
        else:
            out1 = x + attention_out

        # Feed-forward network
        ff_out = self.feed_forward(out1)
        ff_out = self.dropout2(ff_out)
        
        # Apply normalization if not 'none'
        if self.norm2 is not None:
            if isinstance(self.norm2, nn.LayerNorm):
                out2 = self.norm2(out1 + ff_out)
            elif isinstance(self.norm2, nn.BatchNorm1d):
                out2 = self.norm2((out1 + ff_out).transpose(1, 2)).transpose(1, 2)
        else:
            out2 = out1 + ff_out

        return out2

class SimpleTransformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, num_classes, norm_type, dropout=0.1):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(30522, d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, norm_type, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        # Initialize the embedding layer
        nn.init.kaiming_normal_(self.embedding.weight, mode='fan_in', nonlinearity='linear')

        # Initialize the output layer (fc_out)
        nn.init.kaiming_normal_(self.fc_out.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, x, mask=None):
        x = self.embedding(x)
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return self.fc_out(x)

#%%
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, random_initialization=False):
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.8, verbose=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if random_initialization:
        def random_initialize(module):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.uniform_(module.weight, a=-100, b=100)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.uniform_(module.bias, a=-100, b=100)

        # Apply random initialization to all modules
        model.apply(random_initialize)

    history = {'train_loss': [], 'val_loss': []}
    step_train_losses = []
    step_val_losses = []
    step_difference_max = []
    step_gradient_difference_max = []

    for epoch in range(num_epochs):
        model.train()  # 设置为训练模式

        # 训练循环
        for step, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # 转置 input_ids 为 [seq_len, batch_size]
            input_ids = input_ids.transpose(0, 1)

            # 前向传播
            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, 3), labels.unsqueeze(1).repeat(1, 512).view(-1).long())  # 3 类别
            loss.backward()
            
            lrs = np.arange(1e-5, 1e-4, 1e-5)

            difference = 0
            gradient_difference = 0
            difference_max = 0
            gradient_difference_max = 0
            gradients = []

            for lr in lrs:
                for param in model.parameters():
                    if param.grad is not None:
                        gradient = param.grad.data
                        gradients.append(gradient)
                        param.data -= lr * gradient
                
                optimizer.zero_grad()

                outputs = model(input_ids)
                loss_new = criterion(outputs.view(-1, 3), labels.unsqueeze(1).repeat(1, 512).view(-1).long())  # 3 类别
                difference = np.abs(loss_new.item() - loss.item())
                difference_max = max(difference, difference_max)
                loss = criterion(outputs.view(-1, 3), labels.unsqueeze(1).repeat(1, 512).view(-1).long())  # 3 类别
                loss.backward()

                pointer = 0
                gradient_difference = []         
                for param in model.parameters():
                    if param.grad is not None:
                        gradient = param.grad.data
                        gradient_difference = np.linalg.norm((gradient - gradients[pointer]).cpu().detach().numpy())
                        gradient_difference_max = max(gradient_difference, gradient_difference_max)
                        pointer += 1

                pointer = 0
                for param in model.parameters():
                    if param.grad is not None:
                        gradient = gradients[pointer]
                        param.data += lr * gradient
                        pointer += 1
        
            step_difference_max.append(difference_max)
            step_gradient_difference_max.append(gradient_difference_max)
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, 3), labels.unsqueeze(1).repeat(1, 512).view(-1).long())  # 3 类别
            loss.backward()
            optimizer.step()

            print(f"Step [{step+1}/{len(train_loader)}], Train Loss: {loss.item():.4f}")

            # 记录每个 step 的训练损失
            step_train_losses.append(loss.item())

            # 验证循环
            model.eval()  # 设置为评估模式
            total_val_loss = 0

            with torch.no_grad():  # 不计算梯度
                for val_batch in val_loader:
                    val_input_ids = val_batch['input_ids'].to(device)
                    val_attention_mask = val_batch['attention_mask'].to(device)
                    val_labels = val_batch['labels'].to(device)

                    # 转置 input_ids 为 [seq_len, batch_size]
                    val_input_ids = val_input_ids.transpose(0, 1)

                    # 前向传播
                    val_outputs = model(val_input_ids)
                    val_loss = criterion(val_outputs.view(-1, 3), val_labels.unsqueeze(1).repeat(1, 512).view(-1).long())  # 3 类别
                    total_val_loss += val_loss.item()

            # 记录每个 step 的验证损失
            step_val_losses.append(total_val_loss / len(val_loader))

        # 每个 epoch 后输出训练和验证损失
        avg_train_loss = sum(step_train_losses[-len(train_loader):]) / len(train_loader)
        avg_val_loss = sum(step_val_losses[-len(val_loader):]) / len(val_loader)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")

    return history, step_train_losses, step_val_losses, step_difference_max, step_gradient_difference_max

def check_initialized_gradient(model, train_loader, criterion, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()

    batch = next(iter(train_loader))
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    optimizer.zero_grad()

    input_ids = input_ids.transpose(0, 1)
    outputs = model(input_ids)
    loss = criterion(outputs.view(-1, 3), labels.unsqueeze(1).repeat(1, 512).view(-1).long())  # 3 类别
    loss.backward()

    # Recode FOG
    for name, param in model.named_parameters():
        if param.grad is not None:
            FOG = param.grad.cpu().detach().numpy()

            if FOGs.get(name, -1) == -1:
                FOGs[name] = [np.linalg.norm(FOG)]
            else:
                FOGs[name].append(np.linalg.norm(FOG))


# 模型超参数设置
d_model = 32
num_heads = 8
d_ff = 100
num_layers = 5
num_classes = 3
dropout = 0.1

proportion = dataset.sentiment.value_counts().sort_index().to_list()
class_weights = torch.tensor(proportion, dtype=torch.float).cuda()
criterion = nn.CrossEntropyLoss(weight=class_weights)


#%%
model = SimpleTransformer(num_layers, d_model, num_heads, d_ff, num_classes, "layer", dropout)
optimizer = optim.SGD(model.parameters(), lr=1e-4)
history_layer = train(model, train_loader, val_loader, criterion, optimizer, num_epochs=3, random_initialization=False)

model = SimpleTransformer(num_layers, d_model, num_heads, d_ff, num_classes, "batch", dropout)
optimizer = optim.SGD(model.parameters(), lr=1e-4)
history_batch = train(model, train_loader, val_loader, criterion, optimizer, num_epochs=3, random_initialization=False)

model = SimpleTransformer(num_layers, d_model, num_heads, d_ff, num_classes, "none", dropout)
optimizer = optim.SGD(model.parameters(), lr=1e-4)
history_none= train(model, train_loader, val_loader, criterion, optimizer, num_epochs=3, random_initialization=False)
#%%
FOGs = {}
for seed in tqdm(range(1, 101)):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = SimpleTransformer(num_layers, d_model, num_heads, d_ff, num_classes, "layer", dropout)
    optimizer = optim.SGD(model.parameters(), lr=1e-5)
    check_initialized_gradient(model, train_loader, criterion, optimizer)


    model = SimpleTransformer(num_layers, d_model, num_heads, d_ff, num_classes, "batch", dropout)
    optimizer = optim.SGD(model.parameters(), lr=1e-5)
    check_initialized_gradient(model, train_loader, criterion, optimizer)

    model = SimpleTransformer(num_layers, d_model, num_heads, d_ff, num_classes, "none", dropout)
    optimizer = optim.SGD(model.parameters(), lr=1e-5)
    check_initialized_gradient(model, train_loader, criterion, optimizer)


#%%
steps = range(1, len(history_layer[2]) + 1)

plt.plot(steps, history_layer[3], label='Model With LayerNorm', color='blue', linewidth=2)
plt.plot(steps, history_batch[3], label='Model With BatchNorm', color='orange', linewidth=2)
plt.plot(steps, history_none[3], label='Model Without Norm', color='green', linewidth=2)

plt.xlabel('Step')
plt.ylabel('Lipschitz of Loss Function')
plt.legend(loc='upper right', fontsize=8) 
plt.show()

#%%
steps = range(1, len(history_layer[2]) + 1)

plt.plot(steps, history_layer[4], label='Model With LayerNorm', color='blue', linewidth=2)
plt.plot(steps, history_batch[4], label='Model With BatchNorm', color='orange', linewidth=2)
plt.plot(steps, history_none[4], label='Model Without Norm', color='green', linewidth=2)

plt.xlabel('Step')
plt.ylabel('Smoothness of Loss Function')
plt.legend(loc='upper right', fontsize=8)
plt.show()

#%%
def plot_history(history):
    epochs = range(1, len(history[1]) + 1)
    # plt.plot(epochs, history[1], label='Training Loss')
    # plt.plot(epochs, history[2], label='Validation Loss')
    plt.plot(epochs, history[3], label='Lipschitz of Loss Function')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_history(history_layer)
# %%
epochs = range(1, len(history_layer[2]) + 1)

# 绘制 LayerNorm 的训练和验证损失
plt.plot(epochs, history_layer[1], label='Training Loss of Model With LayerNorm', 
         color='blue', linestyle='-', linewidth=2)
plt.plot(epochs, history_layer[2], label='Validation Loss of Model With LayerNorm', 
         color='blue', linestyle='--', linewidth=2)

# 绘制 BatchNorm 的训练和验证损失
plt.plot(epochs, history_batch[1], label='Training Loss of Model With BatchNorm', 
         color='orange', linestyle='-', linewidth=2)
plt.plot(epochs, history_batch[2], label='Validation Loss of Model With BatchNorm', 
         color='orange', linestyle='--', linewidth=2)

# 绘制 Without Norm 的训练和验证损失
plt.plot(epochs, history_none[1], label='Training Loss of Model Without Norm', 
         color='green', linestyle='-', linewidth=2)
plt.plot(epochs, history_none[2], label='Validation Loss of Model Without Norm', 
         color='green', linestyle='--', linewidth=2)

# 设置 x 轴和 y 轴标签
plt.xlabel('Step')
plt.ylabel('Loss')

# 添加图例
plt.legend(loc='upper right', fontsize=8)

# 显示图形
plt.show()

#%%
keys = []
layer_mean = []
layer_std = []
batch_mean = []
batch_std = []
none_mean = []
none_std = []

for key, value in FOGs.items():
    if len(value) == 300:
        keys.append(key)
        list1 = value[::3]
        list2 = value[1::3]
        list3 = value[2::3]
        layer_mean.append(np.mean(list1))
        layer_std.append(np.std(list1))
        batch_mean.append(np.mean(list2))
        batch_std.append(np.std(list1))
        none_mean.append(np.mean(list3))
        none_std.append(np.std(list1))

#%%
plt.plot(range(len(keys)), layer_mean, label='LayerNorm', color='blue', linewidth=2)
plt.plot(range(len(keys)), batch_mean, label='BatchNorm', color='orange', linewidth=2)
plt.plot(range(len(keys)), none_mean, label='No Norm', color='green', linewidth=2)
plt.xlabel('Layers')
plt.ylabel('Expected Gradient Norm Magnitude')
plt.legend(loc='upper right')
plt.show()

# %%

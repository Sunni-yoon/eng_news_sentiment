import pandas as pd
import re
import pandas as pd
import os
import PIL
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from PIL import Image
from sklearn.metrics import f1_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
import fasttext
from sklearn.model_selection import train_test_split

df10000_okt = pd.read_csv('./korean_sentiment/df10000_okt.csv')

df10000_okt = df10000_okt[:10000]

prd_text = []

for text in df10000_okt['Review'].values :
    text = re.sub("[^가-힣a-zA-Z0-9]+", " ", text)
    prd_text.append(text)
    
df10000_okt['Review_prd'] = prd_text

df10000_okt['Review_prd'].str.strip()

review = df10000_okt['Review_prd_okt']
rating = df10000_okt['Rating']

model = fasttext.load_model("./korean_sentiment/cc.ko.300.bin")

encoder = LabelEncoder()
rating = encoder.fit_transform(rating)

# x_train, x_test, y_train, y_test = train_test_split(review, rating, test_size=0.2, shuffle=True, stratify=rating, random_state=34)

x_train_val, x_test, y_train_val, y_test = train_test_split(
    review, rating, test_size=0.2, shuffle=True, stratify=rating, random_state=34
)

x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_val, y_train_val, test_size=0.25, shuffle=True, stratify=y_train_val, random_state=34
)

class Custom_dataset(Dataset):
    def __init__(self, reviews, ratings, ft_model, max_length=128):
        """
        reviews: 각 리뷰가 토큰 리스트 형태로 저장되어 있음. 예: [['숙성', '돼지고기', ...], ...]
        ratings: 각 리뷰의 라벨 리스트
        ft_model: fastText 모델 (cc.ko.300.bin)
        max_length: 고정 시퀀스 길이 (예: 256)
        """
        self.reviews = reviews
        self.ratings = ratings
        self.ft_model = ft_model
        self.max_length = max_length

    def __getitem__(self, idx):
        vectors = []
        tokens = self.reviews[idx]
        for token in tokens :
            vectors_ = self.ft_model.get_word_vector(token)
            vectors.append(vectors_)
        # vectors = [self.ft_model.get_word_vector(token) for token in tokens]
        vectors = np.array(vectors)  # (현재 리뷰의 단어 수, 300)

        # 고정 길이(max_length)로 패딩 또는 자르기
        seq_length = vectors.shape[0]
        if seq_length < self.max_length:
            pad_length = self.max_length - seq_length
            pad = np.zeros((pad_length, vectors.shape[1]))  # (pad_length, 300)
            vectors = np.concatenate([vectors, pad], axis=0)
        else:
            vectors = vectors[:self.max_length]

        review_tensor = torch.tensor(vectors, dtype=torch.float)  # (max_length, 300)
        rating_tensor = torch.tensor(self.ratings[idx], dtype=torch.long)
        return review_tensor, rating_tensor

    def __len__(self):
        return len(self.ratings)
    
train_dataset = Custom_dataset(x_train.tolist(), y_train.tolist(), model)
valid_dataset = Custom_dataset(x_valid.tolist(), y_valid.tolist(), model)    
test_dataset = Custom_dataset(x_test.tolist(), y_test.tolist(), model)    
    
# train_dataset = Custom_dataset(x_train, y_train, model) 
# test_dataset = Custom_dataset(x_test, y_test, model)

# train_dataset = Custom_dataset(x_train.values, y_train.values, model) # key 오류나면 무조건 .values
# test_dataset = Custom_dataset(x_test.values, y_test.values, model)

train_dataloader = DataLoader(train_dataset, batch_size=256,  shuffle=True)
valid_dataloader = DataLoader(train_dataset, batch_size=256,  shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=256,  shuffle=False)

class TextLSTM(nn.Module):
  def __init__(self, input, output_dim, hidden_size, num_layers, dropout_rate=0.3): 
    # input은 임베딩값, output은 라벨 갯수, hidden_size는 lstm 내부차원, seq_length는 차원 값에 영향을 안 받음 !!! 
    super(TextLSTM, self).__init__()
    
    self.input = input
    self.output_dim = output_dim
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    # self.seq_length = seq_length

    self.lstm = nn.LSTM(input_size=self.input, hidden_size=self.hidden_size, num_layers=num_layers)
    self.dropout = nn.Dropout(dropout_rate)
    self.linear1 = nn.Linear(hidden_size, 128)
    self.linear2 = nn.Linear(128, output_dim)
    self.relu = nn.ReLU()

  def forward(self, x):
    x, _ = self.lstm(x) # batch, 시퀀스, 임베딩
    x = x[:, -1, :] # 배치전체, 제일 마지막 레이어의 마지막 단어, 전체 임베딩 값 (원래 outputs이 가장 마지막 레이어를 뜻함)
    # print(x[0])
    x = self.relu(x)
    x = self.linear1(x)
    x = self.relu(x)
    output = self.dropout(x)
    logit = self.linear2(output)
    return logit

gpu_ids = [0,1,2,3,4]
device = torch.device('cuda:{}'.format(gpu_ids[0]))
model = TextLSTM(300, 3, 256, 2).to(device)
model = nn.DataParallel(model, device_ids=gpu_ids)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

'''
for epoch in range(10):

    total_loss = 0
    correct = 0  
    total_samples = 0  

    model.train()
    for reviews, rating in train_dataloader:
        reviews = reviews.to(device)
        rating = rating.to(device)

        optimizer.zero_grad()

        outputs = model(reviews) # 학습할 때는 softmax 사용 x. 크로스앤트로피만 사용하기. test할때 softmax 거쳐서 나온 값으로 분류하기
        loss = criterion(outputs, rating)  # 크로스앤트로피 안에 softmax가 있음
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        soft_logit = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, dim=1)

        # ✅ 정확도 계산
        correct += (prediction == rating).sum().item()  # 예측값과 정답 비교하여 맞춘 개수 저장
        total_samples += rating.size(0)  # 전체 샘플 개수 업데이트

        # print('soft_logit :', soft_logit[0])

    accuracy = correct / total_samples * 100  # 퍼센트(%) 단위 변환
    print(f'예측 label : {prediction[:5].tolist()}')
    print(f'실제 label : {rating[:5].tolist()}')

    print(f'{epoch} 에폭 loss: {total_loss:.4f}, 정확도: {accuracy:.2f}%')

    # torch.save(model.state_dict(), f'./dogs/resnet50_dogs2_classification{epoch}.pth')
    # print(f'{epoch} 에폭 모델 저장 완료')
'''

patience = 5  
best_val_loss = float('inf')
patience_counter = 0

# --- Training loop with early stopping --- 여기 참고하기
for epoch in range(50):
    model.train()
    total_train_loss = 0
    for reviews, labels in train_dataloader:
        reviews = reviews.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(reviews)  # 모델 출력 (logits)
        print(outputs[0])
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
    
    # --- Validation phase ---
    model.eval()
    total_val_loss = 0
    correct = 0
    total_samples = 0
    all_val_preds = []
    all_val_labels = []
    with torch.no_grad():
        for reviews, labels in valid_dataloader:
            reviews = reviews.to(device)
            labels = labels.to(device)
            
            outputs = model(reviews)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
            
            # 예측값: 로짓에서 argmax (Softmax를 거치지 않아도 argmax는 동일)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            all_val_preds.extend(preds.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())
    
    val_accuracy = correct / total_samples * 100
    val_loss_avg = total_val_loss / len(valid_dataloader)
    f1 = f1_score(all_val_labels, all_val_preds, average='weighted')
    
    print(f"Epoch {epoch}: Train Loss = {total_train_loss:.4f}, Val Loss = {val_loss_avg:.4f}, Val Acc = {val_accuracy:.2f}%, F1 = {f1:.4f}")
    
    # --- Early Stopping 체크 ---
    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        patience_counter = 0
        torch.save(model.state_dict(), './korean_sentiment/best_resnet_model.pth')
        print("Best model saved.")
    else:
        patience_counter += 1
        print(f'{patience_counter}번 참았다.')
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

# --- 모델 불러와서 최종 평가 ---
model.load_state_dict(torch.load('./korean_sentiment/best_resnet_model.pth'))
model.eval()
total_correct = 0
total_samples = 0
all_preds = []
all_labels = []
with torch.no_grad():
    for reviews, labels in test_dataloader:
        reviews = reviews.to(device)
        labels = labels.to(device)
        outputs = model(reviews)
        preds = torch.argmax(outputs, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

final_accuracy = total_correct / total_samples * 100
final_f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"Final Validation Accuracy: {final_accuracy:.2f}%")
print(f"Final Validation F1-Score: {final_f1:.4f}")

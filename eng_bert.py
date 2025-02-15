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
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from PIL import Image
from sklearn.metrics import f1_score
import nltk
from nltk.tokenize import word_tokenize


df = pd.read_csv('./eng_bert/train.csv')

from nltk.tokenize import word_tokenize

final_word_list = []

def lemmatization(POS_list, DROP_1WORD):
    lemma_list = []
    lemmatizer = nltk.stem.WordNetLemmatizer()
    
    for sentence in POS_list:
        #Lemmatize 함수를 사용할 때, 해당 토큰이 어떤 품사인지 알려줄 수 있다. 만약 품사를 알려주지 않는다면 올바르지 않은 결과가 반환될 수 있다.
        #Lemmatize 함수가 입력받는 품사는 동사, 형용사, 명사, 부사 뿐이다. ===>  각각 v, a, n, r로 입력받는다.
        #nltk.pos_tag로 반환받는 품사 중, 형용사는 J로 시작하기 때문에 lemmatize 전에 a로 바꿔줘야 한다.

        func_j2a = lambda x : x if x != 'j' else 'a'
        pos_contraction = [(token, func_j2a(POS.lower()[0])) for token, POS in sentence if POS[0] in ['V', 'J', 'N', 'R']]
        
        if DROP_1WORD: ### 글자수 1인 경우에는 drop 
            for token, POS in pos_contraction :
                if len(token) != 1: 
                    lemma_list.append(lemmatizer.lemmatize(token, POS))
        else: 
            lemma_list.append([lemmatizer.lemmatize(token, POS) for token, POS in pos_contraction])
    return lemma_list

def remove_stopwords(lemma_list):
    no_stopword_list = list()
    stop_words = set(nltk.corpus.stopwords.words('english')) # 불용어 불러오기 
    for lemma_word in lemma_list:
        if lemma_word not in stop_words:
            no_stopword_list.append(lemma_word)
    return no_stopword_list

def text_prep(text): # input shape - sentence 

    DROP_1WORD = True
    # text preprocessing 
    text = text.lower()
    text = text.replace('<br />', ' ')
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n', ' ', text)

    # pos checking - lemmatization
    pos_text = nltk.pos_tag(text.split())
    pos_text_list = [pos_text]
    lem_text_list = lemmatization(pos_text_list, DROP_1WORD)
    no_stopword_list = remove_stopwords(lem_text_list)

    ## save word list 
    final_word_list.append(no_stopword_list)
    # print(no_stopword_list)

    return ' '.join(no_stopword_list)

from tqdm import tqdm
tqdm.pandas()
word_series = df['text'].progress_apply(text_prep) 



from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", output_hidden_states=True)
model = AutoModelForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", output_hidden_states=True)

class total_customdataset(Dataset): # left와 right 모두 같은 데이터 사용

    def __init__(self, df, bert_tokenizer, label):
        self.bert_tokenizer = bert_tokenizer # 버트 토크나이저와
        self.df = df.reset_index(drop = True) # index 초기화
        self.text = self.df['text']
        self.label = label


    def __getitem__(self, idx): # 필요한 인수 3개. 유저 리뷰, 아이템 리뷰, 평점 3개의 인덱스 반환
        tokenized = self.bert_tokenizer(self.text[idx], padding='max_length', truncation=True)

        # right model bert input
        bert_ids = torch.tensor(tokenized['input_ids'])
        bert_attention = torch.tensor(tokenized['attention_mask'])
        label = torch.tensor(self.label[idx])

        return bert_ids, bert_attention, label


    def __len__(self):
        return self.label.shape[0]
    
dataset = total_customdataset(df, tokenizer, label)
    
total_dlr = DataLoader(dataset, batch_size=2, shuffle=True)

class bert_model(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.linear1 = nn.Linear(768, 256)
        self.linear2 = nn.Linear(256, 2)
        self.Tanh = nn.Tanh()
        
    def forward(self, ids, attention, output_hidden_states=True):
        feature = self.bert(ids, attention)
        feature = feature.hidden_states[-1][:,0,:]
        feature = self.Tanh(self.linear1(feature))
        output = self.linear2(feature)
        
        return output
    
gpu_ids = [0,1,2,3,4]
device = torch.device('cuda:{}'.format(gpu_ids[0]))
model = bert_model(model).to(device)
model = nn.DataParallel(model, device_ids=gpu_ids)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)


patience = 5  
best_val_loss = float('inf')
patience_counter = 0

# --- Training loop with early stopping --- 여기 참고하기
for epoch in range(50):
    model.train()
    total_train_loss = 0
    for batch in total_dlr:
        bert_ids, bert_attention, label = batch
        bert_ids = bert_ids.to(device)
        bert_attention = bert_attention.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(bert_ids, bert_attention)
        # print(outputs[0])
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
    
    torch.save(model.state_dict(), './eng_bert/best_resnet_model.pth')
    
    '''
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
        '''

# 모델 불러오기 및 평가 모드 전환
model.load_state_dict(torch.load('./eng_bert/best_resnet_model.pth'))
model.eval()

all_preds_val = []
all_labels_val = []
all_preds_test = []

with torch.no_grad():
    # Validation 데이터에 대한 평가
    total_correct = 0
    total_samples = 0
    for reviews, labels in val_dlr:
        reviews = reviews.to(device)
        labels = labels.to(device)
        outputs = model(reviews)
        preds = torch.argmax(outputs, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        all_preds_val.extend(preds.cpu().numpy())
        all_labels_val.extend(labels.cpu().numpy())
    
    # Test 데이터에 대한 예측
    for reviews, _ in test_dlr:
        reviews = reviews.to(device)
        outputs = model(reviews)
        preds = torch.argmax(outputs, dim=1)
        all_preds_test.extend(preds.cpu().numpy())

# Validation 지표 계산
final_accuracy = total_correct / total_samples * 100
final_f1 = f1_score(all_labels_val, all_preds_val, average='weighted')
final_recall = recall_score(all_labels_val, all_preds_val, average='weighted')
final_precision = precision_score(all_labels_val, all_preds_val, average='weighted')

print(f"Final Validation Accuracy: {final_accuracy:.2f}%")
print(f"Final Validation F1-Score: {final_f1:.4f}")
print(f"Final Validation Recall: {final_recall:.4f}")
print(f"Final Validation Precision: {final_precision:.4f}")

# Test 예측 결과 저장
df_inference = pd.DataFrame({'prediction': all_preds_test})
df_inference.to_csv('./test_predictions.csv', index=False)
print("Test predictions saved to 'test_predictions.csv'")

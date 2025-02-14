import torch
import pandas as pd
import numpy as np
import random
import time
import datetime
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim


import csv
import os

from torch.utils.data import Dataset, DataLoader 

# BERT 사용을 위함
from transformers import AutoTokenizer, AutoModelForSequenceClassification #right

#left
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler



def load_embedding(word2vec_file):
    
    with open(word2vec_file, encoding='utf-8') as f:
        
        word_emb = list() # list
        word_dict = dict() # dict({emb:단어})
        
        word_emb.append([0])
        word_dict['<UNK>'] = 0  # Unknown token는 0 벡터로 매핑
        
        for line in tqdm(f.readlines()):
            tokens = line.split(' ')
            word_emb.append([float(i) for i in tokens[1:]])
            word_dict[tokens[0]] = len(word_dict)
            
        word_emb[0] = [0] * len(word_emb[1])
        
    print(f'#### len_word_emb = {len(word_emb)}, len_word_dict = {len(word_dict)}')
        
    return word_emb, word_dict
  

word_emb, word_dict = load_embedding('glove.6B.50d.txt')

df = pd.read_csv('total_test_original2.csv')
# df = df.drop('label', axis=1)


import pandas as pd
import torch
from torch.utils.data import Dataset


class DeepCoNNDataset(Dataset):
    
    # basic
    def __init__(self, df, word_dict):
        
        
        self.word_dict = word_dict # {'워드':'차례 번호'}
        
        # self.config = config
        
        self.PAD_WORD_idx = self.word_dict['<UNK>']
        self.review_length =  40   # max review length
        self.review_count  =  10   # max review count

        df.columns=['userID', 'itemID', 'review', 'rating'] 
        df['review'] = df['review'].apply(self._review2id)  # 리뷰 > 단어 인덱스
    
        
        self.user_reviews = self._get_reviews(df, 'userID', 'itemID')                      # 리뷰를 갖고 오기 
        self.item_reviews = self._get_reviews(df, 'itemID', 'userID')  # 반대가 됩니다.
        
        self.rating = torch.Tensor(df['rating'].to_list()).view(-1, 1) # make (51764, 1)



    def __getitem__(self, idx):
        return self.user_reviews[idx], self.item_reviews[idx], self.rating[idx]

    def __len__(self):
        return self.rating.shape[0]
    
    # 추가정의 ---- 리뷰 전처리를 여기서 한 번 더 진행 ----
    # (이미 단어사전 idx화 되어있는) 리뷰 가져오기
    def _get_reviews(self, df, sub, obj):
       
    
        reviews_by_lead = dict(list(df[[obj, 'review']].groupby(by=df[sub])))  # dict({유저ID:판다스(아이템|리뷰)})
        # print(list(df[[costar, 'review']].groupby(by=df[lead]))[0])
        
        subject_reviews = []
       
        print(f'processing of subject : {sub}')
        
        for idx, (sub_id, obj_id) in enumerate((zip(tqdm(df[sub]), df[obj]))):
            df_data = reviews_by_lead[sub_id]  # 위 dict(해당 ID의 판다스 가져옴)에서 가져옴

            reviews = df_data['review'].to_list()  # 판다스의 리뷰<들> : list로 보냄
                
            # 아래 함수를 호출함
            reviews = self._adjust_review_list(reviews, self.review_length, self.review_count)
            subject_reviews.append(reviews)
            
        return torch.LongTensor(subject_reviews)
    
    
    # 리뷰 개수 및 길이 조정
    def _adjust_review_list(self, reviews, r_length, r_count):
        reviews = reviews[:r_count] + [[self.PAD_WORD_idx] * r_length] * (r_count - len(reviews)) 
        reviews = [r[:r_length] + [0] * (r_length - len(r)) for r in reviews]            
        return reviews

    # 리뷰를 단어사전 idx로 변경
    def _review2id(self, review):  
        
        if not isinstance(review, str):
            return []  # bug fix
        
        wids = []
        for word in review.split():
            if word in self.word_dict:
                wids.append(self.word_dict[word])  # 변환
                
            else: # 단어가 없으면 <UNK>, 0  # Unknown token
                wids.append(self.PAD_WORD_idx)
                
                
        return wids


train_dataset = DeepCoNNDataset(df, word_dict)
train_dlr = DataLoader(train_dataset, batch_size=32, shuffle=False)


class CNN(nn.Module):
    
    def __init__(self, word_dim):
        super().__init__()

        self.kernel_count  = 100 # config.kernel_count
        self.review_count  = 10 # config.review_count : max review count
        self.kernel_size   = 3
        self.review_length = 40
        self.dropout_prob  = 0.5
        self.cnn_out_dim = 50
        
        
        # [?,40,50] > permute(0,2,1) > [?,50,40] > [?,100,40] > pooling [?,100,1] 
        
        self.conv = nn.Sequential( # torch 문법. 쓸 conv layer 선언
            nn.Conv1d(
                in_channels= word_dim,
                out_channels= self.kernel_count,
                kernel_size= self.kernel_size,
                padding=(self.kernel_size - 1) // 2),  # out shape(new_batch_size, kernel_count, review_length) 
            
            nn.ReLU(),  # Relu
            nn.MaxPool2d(kernel_size=(1, self.review_length)),  # out shape(new_batch_size,kernel_count,1)
            nn.Dropout(p=self.dropout_prob))

        self.linear = nn.Sequential(
                           # 100           # 10
            nn.Linear(self.kernel_count * self.review_count, self.cnn_out_dim),  # self.cnn_out_dim : 출력차원
            nn.ReLU(),
            nn.Dropout(p=self.dropout_prob))

    def forward(self, vec):  # input shape(new_batch_size, review_length, word2vec_dim)
        
        
        # conv1d 적용하기 위해 permute
        latent = self.conv(vec.permute(0, 2, 1))   # out(new_batch_size, kernel_count, 1) 
        latent = self.linear(latent.reshape(-1, self.kernel_count * self.review_count)) # reshape    :  1  * 1000
        return latent  # out shape(batch_size(원래갯수 51750), cnn_out_dim)

class FactorizationMachine(nn.Module):
    
    # 드랍아웃을 전체적으로 적용했음
    
    def __init__(self, p, k):  # p = cnn_out_dim, k = 잠재변수( 보통 8, 10)
        
        super().__init__()
        
        self.v = nn.Parameter(torch.rand(p, k) / 10)   # -0.1 ~ 0.1 
        self.linear  = nn.Linear(p, 1, bias=True) 
        self.dropout = nn.Dropout(0.5)

        
        
    def forward(self, x):
        
        linear_part = self.linear(x)  # input shape(batch_size, cnn_out_dim), out shape(batch_size, 1)
        
        inter_part1 = torch.mm(x, self.v) ** 2
        inter_part2 = torch.mm(x ** 2, self.v ** 2)
        
        pair_interactions = torch.sum(inter_part1 - inter_part2, dim=1, keepdim=True)
        
        pair_interactions = self.dropout(pair_interactions)
        
        output = linear_part + 0.5 * pair_interactions
        return output  # out shape(batch_size, 1)
      

class DeepCoNN(nn.Module):
    
    def __init__(self, word_emb):
        
        super().__init__() # 매개변수 생성
        
        self.cnn_out_dim = 50
        
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(word_emb))
        self.cnn_u = CNN(word_dim=self.embedding.embedding_dim)  # cnn_u
        self.cnn_i = CNN(word_dim=self.embedding.embedding_dim)  # cnn_i
        self.fm = FactorizationMachine(self.cnn_out_dim * 2, 8)

    def forward(self, user_review, item_review):  # input shape(batch_size, review_count, review_length)
        
        new_batch_size = user_review.shape[0] * user_review.shape[1]
        user_review = user_review.reshape(new_batch_size, -1)
        item_review = item_review.reshape(new_batch_size, -1)

        # 임베딩 처리
        u_vec = self.embedding(user_review)
        i_vec = self.embedding(item_review)

        # CNN
        user_latent = self.cnn_u(u_vec)
        item_latent = self.cnn_i(i_vec)

        concat_latent = torch.cat((user_latent, item_latent), dim=1)
        
        # FM
        prediction = self.fm(concat_latent)
        
        return prediction
      
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('using device:', device)

'''
model = DeepCoNN(word_emb).to(device)

optim = torch.optim.Adam(model.parameters(), lr=0.001)
mse = nn.MSELoss()



for epoch in range(20):
    
    cost = 0
    rating_cost = 0
    cost_total=0
    
    for i, batch in (enumerate(train_dlr)):
        user_reviews, item_reviews, ratings = batch
        user_reviews=user_reviews.to(device)
        item_reviews=item_reviews.to(device)
        ratings=ratings.to(device)

        predict = model(user_reviews, item_reviews)
        
        loss = mse(predict, ratings)

        optim.zero_grad()
        loss.backward()
        optim.step()
        
        # tensor형 변환을 위한 코드(scalr)
        cost += loss.item() 
    
        # if i % 200 == 0:
            # print('{}에폭의 {} 배치 마다의 distance:'.format(epoch,i),root_distance)
        
        
        # avg = (sum(cos_s))/16 # 배치 사이즈로 나누기
        # avg += avg.item()
        
    cost_total = cost/(i+1)
        
    print('epoch_batch_loss : {}'.format(cost_total)) # scalr
    print('rating : ',predict, 'label : ', ratings) # tensor
    
torch.save(model.state_dict(), 'Deepconn/DM2_loss:{:.2f}.pt'.format(cost_total))

'''

param = torch.load('Deepconn/MI_loss:0.65.pt')
model = DeepCoNN(word_emb).to(device)
model.load_state_dict(param)


rating_list=[]

total_cost=0
rating_cost=0

with torch.no_grad():
    model.eval()
    
    for i, batch in tqdm(enumerate(train_dlr)):
        user_reviews, item_reviews, ratings = batch
        user_reviews=user_reviews.to(device)
        item_reviews=item_reviews.to(device)
        # print(user_reviews.size())
        # print(user_reviews.unsqueeze(1).size()) 
        # user_reviews = user_reviews.unsqueeze(1).to(device)
        # item_reviews = item_reviews.unsqueeze(1).to(device) 
        ratings=ratings.to(device)
        
        predict_rating = model(user_reviews, item_reviews)
        
        # tensor_predict = predict_rating.to('cpu').tolist()
        tensor_predict = predict_rating.squeeze(1)
        tensor_predict = tensor_predict.tolist()
        rating_list.append(tensor_predict)
        
        '''tensor_predict = predict_rating.to('cpu').tolist()
        predict_rating = predict_rating.tolist()
        rating_list.append(tensor_predict)'''
        
    rating_list_l = sum(rating_list, [])
    
    # total_cost = rating_cost/(i+1)

    # print('test rating mse: {}'.format(total_cost))

    '''print(rating_list_l)'''



total_data2 = pd.DataFrame(
            {'deep rating':rating_list_l})

'''total_data2 = pd.DataFrame(
            {'refine rating':rating_list})'''

print(total_data2[:10])

total_data2.to_csv('notna_modeloutput/MI_test_modeloutput_deepconn.csv', index=False)

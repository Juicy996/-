"""
import requests
import pandas as pd
import numpy as np
import json
import html
import urllib
#import sys
import re
#import imp
import random
import time
from threading import Timer
from bs4 import BeautifulSoup
 
 
#imp.reload(sys)
#sys.setdefaultencoding('utf8')
headers ={'user-agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 BIDUBrowser/8.7 Safari/537.36'}
 
def get_html1(i):
    url = 'https://blockchain.info/rawblock/0000000000000bae09a7a393a8acded75aa67e46cb81f7acaa5ad94f9eacd103'
    html = requests.get(url.format(i,random.randint(1501050773102,1501051774102)),headers=headers)
    ceshi1=html.content
    data = json.loads(ceshi1)
    return data
    #return(data['PackageList']['Data'])
 
 
data_ceshi=pd.DataFrame([])
html_list=[]
for i in range(100):
    html_list.append(get_html1(i))
for i,heml_avg in enumerate(html_list):
    tmp=pd.DataFrame(heml_avg)
    tmp["page_id"]=i
    data_ceshi=data_ceshi.append(tmp)
 
 
print (data_ceshi)
data_ceshi.to_csv('./data.csv',encoding='gbk')

"""
import requests

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36",
}
newUrl ="https://blockchain.info/rawblock/0000000000000bae09a7a393a8acded75aa67e46cb81f7acaa5ad94f9eacd103"
response = requests.get(newUrl)
#当爬取的界面需要用户名密码登录时候，构建的请求需要包含auth字段
#response = requests.get(newUrl,headers=headers,auth=('username','passsword'))
print(response.content.decode("utf-8"))#打印网页内容
#print(response.status_code)#浏览器返回的错误码，200表示成功



"""
import tensorflow as tf
init = tf.contrib.layers.xavier_initializer()
num_layer = 4
linear_q = []
linear_k = []
linear_v = []
for i in range(num_layer):
    linear_q.append(tf.keras.layers.Dense(300, kernel_initializer = init, bias_initializer = init, activation = 'tanh'))
    linear_k.append(tf.keras.layers.Dense(300, kernel_initializer = init, bias_initializer = init, activation = 'tanh'))
    linear_v.append(tf.keras.layers.Dense(300, kernel_initializer = init, bias_initializer = init, activation = 'tanh'))
print(linear_k)

import numpy as np
import tensorflow as tf
img = np.array([[1,1,1],
                [1,1,1],
                [1,1,1,]]).astype(np.float32)

img = tf.constant(img)  # 将numpy转换为tensorflow
img = tf.expand_dims(img,2)  # 在最后一维度,添加chanel通道,默认值为1.0
img = tf.expand_dims(img,0)  # 在最前一维度,添加图片,用来表示图片


conv = tf.keras.layers.Conv2D(filters=1,kernel_size=2,strides=(1,1),padding='same')
conv(img)
print(img)
"""
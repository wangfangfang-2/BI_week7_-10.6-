import pandas as pd 
from collections import defaultdict
from datetime import  datetime,timedelta
import matplotlib.pyplot as plt
 #移动推荐系统
 #数据加载
df = pd.read_csv('./tianchi_fresh_comp_train_user.csv') 
 
#df = pd.read_csv('./sample_train_user.csv')
print(df.head())

 #计算CVR behavior_type中count4代表购买
 
count_all,count_4 = 0,0
count_user= df['behavior_type'].value_counts()
count_all = count_user[1] + count_user[2] + count_user[3] + count_user[4]
count_4 += count_user[4]

cvr = count_4/count_all
#用户的点击购买率CVR 百分比*100
print('CXVRJ+{}%'.format(cvr*100))

 #将time字段设置为pandas中的datetime类型
df['time'] = pd.to_datetime(df['time'])
df.index = df['time']
#print(df.head())


#

""" #时间规律统计
def show_count_day(df):
  
    count_day = defaultdict(int)
    #从2014-11-18遍历到2014-12-18
    str1 = '2014-11-17'

    temp_date = datetime.strptime(str1,'%Y-%m-%d')
    delta = timedelta(days=1)
    for i in range(31):
        temp_date = temp_date + delta
        print(temp_date)
        #将时间转化为字符串类型
        temp_str= temp_date.strftime('%Y-%m-%d')
        #df的index为时间
        count_day[temp_str] += df[temp_str].shape[0]
    print(count_day)

    #时间绘制
    #封装df_count_day

    df_count_day = pd.DataFrame.from_dict(count_day, orient='index',columns=['count'])
    df_count_day['count'].plot(kind = 'bar')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
show_count_day(df)"""
#属于商品子集P的个数
df_p = pd.read_csv('./tianchi_fresh_comp_train_item.csv')
#print(df_p.head())

    #df = df.reset_index()
#print(df.head())
#使用reset_index将index还原 做拼接
df = pd.merge(df, df_p, on=['item_id']).set_index('time')
#print(df.shape)
#print(df.head())

#

def show_count_hour(date1):
    count_hour = {}
#设置初始值
    for i in range(24):
        time_str = date1 + ' %02.d' % i
    #print(time_str)
        count_hour[time_str] = [0,0,0,0]
    #print(time_str) 
        temp = df[time_str]['behavior_type'].value_counts()
    #print(count_hour) 
        for j in range(len(temp)):
            count_hour[time_str][temp.index[j]-1] += temp[temp.index[j]]
   # print(count_hour)
        #从字典类型生成DataFrame
    df_count_hour = pd.DataFrame.from_dict(count_hour, orient = 'index')
    df_count_hour.plot(kind = 'bar')
    plt.legend(loc = 'best')
    plt.grid(True)
    plt.show()

show_count_hour('2014-12-12')

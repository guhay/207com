import pandas as pd

def feature_cut(df_feature,cut_bins):
    return pd.cut(df_feature, cut_bins,right=True, labels=range(len(cut_bins)-1))

def cut_bins(df):
    f_col=df.columns.drop('user_id')
    for i in f_col:
        df[i] = df[i].replace(r"\N",0).astype(float)

    money_features=['1_total_fee','2_total_fee','3_total_fee','4_total_fee','pay_num']
    money_bins=[-10,0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,250,300,400,5000]
    for f in money_features:
        print(f)
        df[f+'_cut']=feature_cut(df[f],money_bins)

    traffic_features=['month_traffic','last_month_traffic','local_trafffic_month']
    traffic_bins=[-100,0,50,100,200,300,400,500,600,700,800,900,1024,1524,2048,2548,3072,3572,4096,4596,5120,6144,7168,8192,9216,10240,100000]
    for f in traffic_features:
        print(f)
        df[f+'_cut']=feature_cut(df[f],traffic_bins)
        # feature_one_hot=pd.get_dummies(df[f],prefix=f)
        # df=df.drop([f],axis=1)
        # df=pd.concat([df,feature_one_hot],axis=1)


    call_features=['local_caller_time','service1_caller_time','service2_caller_time']
    call_bins=[-100,0,1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,150,200,300,500,10000]
    for f in call_features:
        print(f)
        df[f+'_cut']=feature_cut(df[f],call_bins)
    return df

def get_feature(df):
    f_col=df.columns.drop('user_id')
    for i in f_col:
        df[i] = df[i].replace(r"\N",0).astype(float)
    df['fee-pay']=df['1_total_fee']-df['pay_num']
    df['pay_every']=df['pay_num']/df['pay_times']
    df['pay_1_2']=df['1_total_fee']-df['2_total_fee']
    df['pay_2_3']=df['2_total_fee']-df['3_total_fee']
    df['pay_3_4']=df['3_total_fee']-df['4_total_fee']
    df['pay_mean']=(df['1_total_fee']+df['2_total_fee']+df['3_total_fee']+df['4_total_fee'])/4
    df['pay1_mean']=df['1_total_fee']-df['pay_mean']
    df['pay2_mean'] = df['2_total_fee'] - df['pay_mean']
    df['pay3_mean'] = df['3_total_fee'] - df['pay_mean']
    df['pay4_mean'] = df['4_total_fee'] - df['pay_mean']
    df['pay_mean_mean']=(df['pay1_mean']+df['pay2_mean']+df['pay3_mean']+df['pay4_mean'])/4
    df['pay_1_divide_online']=df['1_total_fee']/df['online_time']
    df['pay_mean_divide_online']=df['pay_mean']/df['online_time']

    df['month_traffic_rate']=df['month_traffic']/(df['month_traffic']+df['local_trafffic_month']+0.01)
    df['local_traffic_rate']=df['local_trafffic_month']/(df['month_traffic']+df['local_trafffic_month']+0.01)
    df['use_this_month_traffic']=df['month_traffic']-df['last_month_traffic']
    df['traffic_all']=df['month_traffic']+df['local_trafffic_month']
    df['last_traffic_ratr']=df['last_month_traffic']/(df['month_traffic']+0.01)
    df['last_traffic_pay_now']=df['last_month_traffic']/df['1_total_fee']
    df['last_traffic_pay_last']=df['last_month_traffic']/df['2_total_fee']
    df['traffic_pay_now']=df['month_traffic']/df['1_total_fee']
    df['traffic_pay_last']=df['month_traffic']/df['2_total_fee']

    df['pre_fee_traffic']=df['month_traffic']/df['1_total_fee']
    df['pre_fee_local_traffic']=df['local_trafffic_month']/df['1_total_fee']
    df['pre_fee_traffic_last_month']=df['last_month_traffic']/df['2_total_fee']

    df['local_caller_rate']=df['local_caller_time']/(df['local_caller_time']+df['service1_caller_time']+df['service2_caller_time']+0.01)
    df['service_caller_rate']=df['service1_caller_time']/(df['local_caller_time']+df['service1_caller_time']+df['service2_caller_time']+0.01)
    df['service2_caller_rate']=df['service2_caller_time']/(df['local_caller_time']+df['service1_caller_time']+df['service2_caller_time']+0.01)
    df['caller2_1']=df['service2_caller_time']-df['service1_caller_time']
    df['caller2_local']=df['service2_caller_time']-df['local_caller_time']
    df['local_caller_rate_devide_pay1'] = df['local_caller_time'] / df['1_total_fee']
    df['local_caller_rate_devide_pay2'] = df['local_caller_time'] / df['2_total_fee']
    df['local_caller_rate_devide_pay3'] = df['local_caller_time'] / df['3_total_fee']
    df['local_caller_rate_devide_pay4'] = df['local_caller_time'] / df['4_total_fee']
    df['local_caller_rate_devide_pay_mean'] = df['local_caller_time'] / df['pay_mean']
    df['service_caller_rate_devide_pay1'] = df['service1_caller_time'] / df['1_total_fee']
    df['service_caller_rate_devide_pay2'] = df['service1_caller_time'] / df['2_total_fee']
    df['service_caller_rate_devide_pay3'] = df['service1_caller_time'] / df['3_total_fee']
    df['service_caller_rate_devide_pay4'] = df['service1_caller_time'] / df['4_total_fee']
    df['service_caller_rate_devide_pay_mean'] = df['service1_caller_time'] / df['pay_mean']
    df['service2_caller_rate_devide_pay1']=df['service2_caller_time']/df['1_total_fee']
    df['service2_caller_rate_devide_pay2'] = df['service2_caller_time'] / df['2_total_fee']
    df['service2_caller_rate_devide_pay3'] = df['service2_caller_time'] / df['3_total_fee']
    df['service2_caller_rate_devide_pay4'] = df['service2_caller_time'] / df['4_total_fee']
    df['service2_caller_rate_devide_pay_mean'] = df['service2_caller_time'] / df['pay_mean']

    df['pre_fee_local_caller']=df['local_caller_time']/(df['1_total_fee']+0.01)
    df['pre_fee_service_caller']=df['service1_caller_time']/(df['1_total_fee']+0.01)
    df['pre_fee_service2_caller']=df['service2_caller_time']/(df['1_total_fee']+0.01)
    return df

df_train=pd.read_csv('files/train.csv')
df_test=pd.read_csv('files/test.csv')
# df_train=cut_bins(df_train)
# df_test=cut_bins(df_test)
df_train=get_feature(df_train)
df_test=get_feature(df_test)
df_train.to_csv('files/train_feature.csv',index=False)
df_test.to_csv('files/test_feature.csv',index=False)

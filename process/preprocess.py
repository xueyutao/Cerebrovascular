import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正确显示中文标签
plt.rcParams['font.serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正确显示负号

database_src = '../database/随访导出数据.xlsx'
# database_src = './database/patient.xlsx'
out_src = '../database/patient.csv'


def drop_col(df, col_name, cutoff=0.5):
    n = len(df)
    cnt = df[col_name].count()
    if (float(cnt) / n) < cutoff:
        df.drop(col_name, axis=1, inplace=1)


''' 患者基本信息 '''


def info_preprocess(data):
    print('患者信息数据预处理中...')
    if '性别' in data.columns:
        data.loc[data['性别'] == '男', '性别'] = 1  # “男”替换为“1”
        data.loc[data['性别'] == '女', '性别'] = 0
        data = data.dropna(subset=['性别'])
        # data.loc[data['性别'].isnull()] = None
    if '年龄' in data.columns:
        data.loc[data['年龄'] <= 10] = None
        data.loc[data['年龄'] >= 100] = None
        data = data.dropna(subset=['年龄'])
    if 'dn_体重' in data.columns:
        x = data["dn_体重"].mean()
        data.loc[data['dn_体重'] <= 35, 'dn_体重'] = None
        data.loc[data['dn_体重'] >= 150, 'dn_体重'] = None
        data.loc[data['dn_体重'].isna().values == True, 'dn_体重'] = x
    if 'dn_身高' in data.columns:
        data['dn_身高'] = np.where(data['dn_身高'] > 3, data['dn_身高'] / 100, data['dn_身高'])  # 身高异常值处理，不能低于0和大于10（m）
        x = data["dn_身高"].mean()
        data['dn_身高'] = np.where(data['dn_身高'] <= 1, x, data['dn_身高'])
        data.loc[data['dn_身高'].isna().values == True, 'dn_身高'] = x
    if '_体重指数' in data.columns:
        data['_体重指数'] = data['_体重指数'].astype(float)
        # data.loc[data['_体重指数'] <= 10] = None
        data.loc[data['_体重指数'] >= 50] = None
        data['_体重指数'] = np.where(data['_体重指数'] <= 5, data['dn_体重'] / (data['dn_身高'] ** 2), data['_体重指数'])
        # data['_体重指数'] = np.where(data['_体重指数'] >= 50, data['dn_体重'] / (data['dn_身高']**2), data['_体重指数'])
        data.loc[data['_体重指数'].isna().values == True, '_体重指数'] = data['dn_体重'] / (data['dn_身高'] ** 2)
    data.to_csv(out_src, header=True, index=False, encoding='utf_8_sig')
    data2 = data.loc[:, ['_患者ID', '性别', '年龄', 'dn_身高', 'dn_体重', '_体重指数']]
    data2.rename(columns={'_患者ID': 'ID'})
    # mid = data2['ID号']   #取备注列的值
    # data2.pop('ID号')  #删除备注列
    # data2.insert(0, 'ID号', mid) #插入备注列
    data2.to_csv('../database/info.csv', header=True, index=False, encoding='utf_8_sig', float_format='%.2f')


''' 既往病史 '''


def disease_preprocess():
    print('患者病史数据预处理中...')
    data = pd.read_csv('../database/patient.csv', encoding='utf-8', low_memory=False)
    if 'CABG史（冠脉搭桥术）' in data.columns:
        data.loc[data['CABG史（冠脉搭桥术）'] == '是', 'CABG史（冠脉搭桥术）'] = 1
        data.loc[data['CABG史（冠脉搭桥术）'] == '否', 'CABG史（冠脉搭桥术）'] = 0
        data.loc[data['CABG史（冠脉搭桥术）'].isna().values == True, 'CABG史（冠脉搭桥术）'] = 2
    if 'dt_PCI史' in data.columns:
        data.loc[data['dt_PCI史'] == '是', 'dt_PCI史'] = 1
        data.loc[data['dt_PCI史'] == '否', 'dt_PCI史'] = 0
        data.loc[data['dt_PCI史'] == '无', 'dt_PCI史'] = 0
        data.loc[(data['dt_PCI史'] == '阿托伐他汀') | (data['dt_PCI史'] == '瑞舒伐他汀') | (data['dt_PCI史'] == '瑞苏伐他汀'), 'dt_PCI史'] = 1
        data.loc[data['dt_PCI史'].isna().values == True, 'dt_PCI史'] = 2
    if 'dt_冠心病家族史' in data.columns:
        data.loc[data['dt_冠心病家族史'] == '是', 'dt_冠心病家族史'] = 1
        data.loc[data['dt_冠心病家族史'] == '否', 'dt_冠心病家族史'] = 0
        data.loc[data['dt_冠心病家族史'] == '无', 'dt_冠心病家族史'] = 0
        data.loc[data['dt_冠心病家族史'] == '华法林', 'dt_冠心病家族史'] = 1
        data.loc[data['dt_冠心病家族史'].isna().values == True, 'dt_冠心病家族史'] = 2
    if 'dt_吸烟' in data.columns:
        # data['dt_吸烟'] = np.where(data['dt_吸烟'].astype("float") <= 20, 1, data['dt_吸烟'])
        data.loc[data['dt_吸烟'] == '当前吸烟', 'dt_吸烟'] = 1
        data.loc[data['dt_吸烟'] == '无', 'dt_吸烟'] = 0
        data.loc[data['dt_吸烟'] == '已戒烟', 'dt_吸烟'] = 0
        data.loc[data['dt_吸烟'].isna().values == True, 'dt_吸烟'] = 2
        data['dt_吸烟'] = data['dt_吸烟'].astype("float")
        data['dt_吸烟'] = np.where((data['dt_吸烟'] > 2) & (data['dt_吸烟'] < 20), 1, data['dt_吸烟'])
    if 'dt_周围血管病史' in data.columns:
        data.loc[data['dt_周围血管病史'] == '是', 'dt_周围血管病史'] = 1
        data.loc[data['dt_周围血管病史'] == '否', 'dt_周围血管病史'] = 0
        data.loc[data['dt_周围血管病史'].isna().values == True, 'dt_周围血管病史'] = 2
    if 'dt_消化道疾病史' in data.columns:
        data.loc[data['dt_消化道疾病史'] == '是', 'dt_消化道疾病史'] = 1
        data.loc[data['dt_消化道疾病史'] == '否', 'dt_消化道疾病史'] = 0
        data.loc[data['dt_消化道疾病史'] == '3', 'dt_消化道疾病史'] = 2
        data.loc[data['dt_消化道疾病史'].isna().values == True, 'dt_消化道疾病史'] = 2
    if 'dt_痛风' in data.columns:
        data.loc[data['dt_痛风'] == '是', 'dt_痛风'] = 1
        data.loc[data['dt_痛风'] == '否', 'dt_痛风'] = 0
        data.loc[data['dt_痛风'].isna().values == True, 'dt_痛风'] = 2
    if 'dt_糖尿病' in data.columns:
        data.loc[data['dt_糖尿病'] == '是', 'dt_糖尿病'] = 1
        data.loc[data['dt_糖尿病'] == '否', 'dt_糖尿病'] = 0
        data.loc[data['dt_糖尿病'].isna().values == True, 'dt_糖尿病'] = 2
        data['dt_糖尿病'] = data['dt_糖尿病'].astype("float")
        data['dt_糖尿病'] = np.where((data['dt_糖尿病'] > 3) & (data['dt_糖尿病'] < 10), 1, data['dt_糖尿病'])
    if 'dt_肾功能不全史' in data.columns:
        data.loc[data['dt_肾功能不全史'] == '是', 'dt_肾功能不全史'] = 1
        data.loc[data['dt_肾功能不全史'] == '否', 'dt_肾功能不全史'] = 0
        data.loc[data['dt_肾功能不全史'].isna().values == True, 'dt_肾功能不全史'] = 2
    if 'dt_肿瘤病史' in data.columns:
        data.loc[data['dt_肿瘤病史'] == '是', 'dt_肿瘤病史'] = 1
        data.loc[data['dt_肿瘤病史'] == '否', 'dt_肿瘤病史'] = 0
        data.loc[data['dt_肿瘤病史'] == '无', 'dt_肿瘤病史'] = 0
        data.loc[(data['dt_肿瘤病史'] == '硝苯地平缓释片') | (data['dt_肿瘤病史'] == '氨氯地平'), 'dt_肿瘤病史'] = 1
        data.loc[data['dt_肿瘤病史'].isna().values == True, 'dt_肿瘤病史'] = 2
    if 'dt_脑血管疾病史' in data.columns:
        data.loc[data['dt_脑血管疾病史'] == '是', 'dt_脑血管疾病史'] = 1
        data.loc[data['dt_脑血管疾病史'] == '否', 'dt_脑血管疾病史'] = 0
        data.loc[data['dt_脑血管疾病史'] == '3', 'dt_脑血管疾病史'] = 2
        data.loc[data['dt_脑血管疾病史'].isna().values == True, 'dt_脑血管疾病史'] = 2
    if 'dt_血脂异常' in data.columns:
        data.loc[data['dt_血脂异常'] == '是', 'dt_血脂异常'] = 1
        data.loc[data['dt_血脂异常'] == '否', 'dt_血脂异常'] = 0
        data.loc[data['dt_血脂异常'].isna().values == True, 'dt_血脂异常'] = 2
        data['dt_血脂异常'] = data['dt_血脂异常'].astype("float")
        data['dt_血脂异常'] = np.where((data['dt_血脂异常'] > 0) & (data['dt_血脂异常'] < 1), 1, data['dt_血脂异常'])
        data['dt_血脂异常'] = np.where((data['dt_血脂异常'] > 1) & (data['dt_血脂异常'] < 2), 1, data['dt_血脂异常'])
    if 'dt_高血压病' in data.columns:
        data.loc[data['dt_高血压病'] == '是', 'dt_高血压病'] = 1
        data.loc[data['dt_高血压病'] == '否', 'dt_高血压病'] = 0
        data.loc[data['dt_高血压病'] == '无', 'dt_高血压病'] = 0
        data.loc[(data['dt_高血压病'] == '氯沙坦') | (data['dt_高血压病'] == '缬沙坦') | (data['dt_高血压病'] == '坎地沙坦'), 'dt_高血压病'] = 1
        data.loc[data['dt_高血压病'].isna().values == True, 'dt_高血压病'] = 2
    if 'dt_心肌梗塞史' in data.columns:
        data.loc[data['dt_心肌梗塞史'] == '是', 'dt_心肌梗塞史'] = 1
        data.loc[data['dt_心肌梗塞史'] == '否', 'dt_心肌梗塞史'] = 0
        data.loc[data['dt_心肌梗塞史'] == '无', 'dt_心肌梗塞史'] = 0
        data.loc[data['dt_心肌梗塞史'].isna().values == True, 'dt_心肌梗塞史'] = 2
    data.to_csv(out_src, header=True, index=False, encoding='utf_8_sig')
    data2 = data.loc[:,
            ['_患者ID', 'dt_吸烟', 'CABG史（冠脉搭桥术）', 'dt_PCI史', 'dt_冠心病家族史', 'dt_周围血管病史', 'dt_消化道疾病史', 'dt_痛风', 'dt_糖尿病',
             'dt_肾功能不全史', 'dt_肿瘤病史', 'dt_脑血管疾病史', 'dt_血脂异常', 'dt_高血压病', 'dt_心肌梗塞史']]
    data2.to_csv('../database/disease.csv', header=True, index=False, encoding='utf_8_sig')


''' 抽血检验指标 '''


def bloodtest_preprocess():
    print('抽血检验指标数据预处理中...')
    data = pd.read_csv('../database/patient.csv', encoding='utf-8', low_memory=False)
    if 'dn_NTproBNP' in data.columns:
        data.loc[data['dn_NTproBNP'] < 0, 'dn_NTproBNP'] = 0
        data.loc[data['dn_NTproBNP'] >= 10000, 'dn_NTproBNP'] = 10000
        x = data['dn_NTproBNP'].mean()
        data.loc[data['dn_NTproBNP'].isna().values == True, 'dn_NTproBNP'] = x
    if 'dn_中性粒细胞数' in data.columns:
        data.loc[data['dn_中性粒细胞数'] < 0, 'dn_中性粒细胞数'] = 0
        data.loc[data['dn_中性粒细胞数'] >= 100, 'dn_中性粒细胞数'] = 100
        x = data['dn_中性粒细胞数'].mean()
        data.loc[data['dn_中性粒细胞数'].isna().values == True, 'dn_中性粒细胞数'] = x
    if 'dn_低密度脂蛋白' in data.columns:
        data.loc[data['dn_低密度脂蛋白'] < 0, 'dn_低密度脂蛋白'] = 0
        data.loc[data['dn_低密度脂蛋白'] >= 20, 'dn_低密度脂蛋白'] = 20
        x = data['dn_低密度脂蛋白'].mean()
        data.loc[data['dn_低密度脂蛋白'].isna().values == True, 'dn_低密度脂蛋白'] = x
    if 'dn_单核细胞数' in data.columns:
        data.loc[data['dn_单核细胞数'] < 0, 'dn_单核细胞数'] = 0
        data.loc[data['dn_单核细胞数'] >= 26, 'dn_单核细胞数'] = 26
        x = data['dn_单核细胞数'].mean()
        data.loc[data['dn_单核细胞数'].isna().values == True, 'dn_单核细胞数'] = x
    if 'dn_同型半胱氨酸' in data.columns:
        x = data['dn_同型半胱氨酸'].mean()
        data.loc[data['dn_同型半胱氨酸'].isna().values == True, 'dn_同型半胱氨酸'] = x
    if 'dn_嗜酸性粒细胞数' in data.columns:
        x = data['dn_嗜酸性粒细胞数'].mean()
        data.loc[data['dn_嗜酸性粒细胞数'].isna().values == True, 'dn_嗜酸性粒细胞数'] = x
    if 'dn_尿酸' in data.columns:
        x = data['dn_尿酸'].mean()
        data.loc[data['dn_尿酸'].isna().values == True, 'dn_尿酸'] = x
    if 'dn_总胆固醇' in data.columns:
        data.loc[data['dn_总胆固醇'] < 0, 'dn_总胆固醇'] = 0
        data.loc[data['dn_总胆固醇'] >= 100, 'dn_总胆固醇'] = 600
        x = data['dn_总胆固醇'].mean()
        data.loc[data['dn_总胆固醇'].isna().values == True, 'dn_总胆固醇'] = x
    if 'dn_总胆红素' in data.columns:
        x = data['dn_总胆红素'].mean()
        data.loc[data['dn_总胆红素'].isna().values == True, 'dn_总胆红素'] = x
    # data = data.drop(columns=['dn_极低密度脂蛋白'])
    # if 'dn_极低密度脂蛋白' in data.columns:
    #     x = data['dn_极低密度脂蛋白'].mean()
    #     data.loc[data['dn_极低密度脂蛋白'].isna().values == True, 'dn_极低密度脂蛋白'] = x
    if 'dn_淋巴细胞数' in data.columns:
        x = data['dn_淋巴细胞数'].mean()
        data.loc[data['dn_淋巴细胞数'].isna().values == True, 'dn_淋巴细胞数'] = x
    if 'dn_甘油三酯' in data.columns:
        x = data['dn_甘油三酯'].mean()
        data.loc[data['dn_甘油三酯'].isna().values == True, 'dn_甘油三酯'] = x
    if 'dn_白细胞总数' in data.columns:
        x = data['dn_白细胞总数'].mean()
        data.loc[data['dn_白细胞总数'].isna().values == True, 'dn_白细胞总数'] = x
    if 'dn_直接胆红素' in data.columns:
        x = data['dn_直接胆红素'].mean()
        data.loc[data['dn_直接胆红素'].isna().values == True, 'dn_直接胆红素'] = x
    if 'dn_空腹血糖' in data.columns:
        x = data['dn_空腹血糖'].mean()
        data.loc[data['dn_空腹血糖'].isna().values == True, 'dn_空腹血糖'] = x
    if 'dn_糖基化血红蛋白' in data.columns:
        x = data['dn_糖基化血红蛋白'].mean()
        data.loc[data['dn_糖基化血红蛋白'].isna().values == True, 'dn_糖基化血红蛋白'] = x
    if 'dn_红细胞体积分布宽度' in data.columns:
        x = data['dn_红细胞体积分布宽度'].mean()
        data.loc[data['dn_红细胞体积分布宽度'].isna().values == True, 'dn_红细胞体积分布宽度'] = x
    if 'dn_红细胞平均体积' in data.columns:
        x = data['dn_红细胞平均体积'].mean()
        data.loc[data['dn_红细胞平均体积'].isna().values == True, 'dn_红细胞平均体积'] = x
    if 'dn_红细胞总数' in data.columns:
        x = data['dn_红细胞总数'].mean()
        data.loc[data['dn_红细胞总数'].isna().values == True, 'dn_红细胞总数'] = x
    if 'dn_红细胞比积' in data.columns:
        x = data['dn_红细胞比积'].mean()
        data.loc[data['dn_红细胞比积'].isna().values == True, 'dn_红细胞比积'] = x
    if 'dn_肌酐' in data.columns:
        x = data['dn_肌酐'].mean()
        data.loc[data['dn_肌酐'].isna().values == True, 'dn_肌酐'] = x
    if 'dn_肌酸激酶' in data.columns:
        # print(data.loc[data['dn_肌酸激酶']].dtypes)
        data.loc[data['dn_肌酸激酶'].values == '?', 'dn_肌酸激酶'] = None
        # data['dn_肌酸激酶'][data['dn_肌酸激酶'] == '?'] = None
        data['dn_肌酸激酶'] = data['dn_肌酸激酶'].astype("float")
        x = data['dn_肌酸激酶'].mean()
        data.loc[data['dn_肌酸激酶'].isna().values == True, 'dn_肌酸激酶'] = x
    if 'dn_肌酸激酶同工酶' in data.columns:
        x = data['dn_肌酸激酶同工酶'].mean()
        data.loc[data['dn_肌酸激酶同工酶'].isna().values == True, 'dn_肌酸激酶同工酶'] = x
    if 'dn_肌钙蛋白I' in data.columns:
        x = data['dn_肌钙蛋白I'].mean()
        data.loc[data['dn_肌钙蛋白I'].isna().values == True, 'dn_肌钙蛋白I'] = x
    if 'dn_肌钙蛋白T' in data.columns:
        x = data['dn_肌钙蛋白T'].mean()
        data.loc[data['dn_肌钙蛋白T'].isna().values == True, 'dn_肌钙蛋白T'] = x
    if 'dn_血小板分布宽度' in data.columns:
        x = data['dn_血小板分布宽度'].mean()
        data.loc[data['dn_血小板分布宽度'].isna().values == True, 'dn_血小板分布宽度'] = x
    if 'dn_血小板总数' in data.columns:
        x = data['dn_血小板总数'].mean()
        data.loc[data['dn_血小板总数'].isna().values == True, 'dn_血小板总数'] = x
    if 'dn_血小板比容' in data.columns:
        x = data['dn_血小板比容'].mean()
        data.loc[data['dn_血小板比容'].isna().values == True, 'dn_血小板比容'] = x
    if 'dn_血清总蛋白' in data.columns:
        x = data['dn_血清总蛋白'].mean()
        data.loc[data['dn_血清总蛋白'].isna().values == True, 'dn_血清总蛋白'] = x
    if 'dn_血清白蛋白' in data.columns:
        x = data['dn_血清白蛋白'].mean()
        data.loc[data['dn_血清白蛋白'].isna().values == True, 'dn_血清白蛋白'] = x
    if 'dn_血红蛋白' in data.columns:
        x = data['dn_血红蛋白'].mean()
        data.loc[data['dn_血红蛋白'].isna().values == True, 'dn_血红蛋白'] = x
    if 'dn_谷丙转氨酶' in data.columns:
        x = data['dn_谷丙转氨酶'].mean()
        data.loc[data['dn_谷丙转氨酶'].isna().values == True, 'dn_谷丙转氨酶'] = x
    if 'dn_超敏C反应蛋白' in data.columns:
        x = data['dn_超敏C反应蛋白'].mean()
        data.loc[data['dn_超敏C反应蛋白'].isna().values == True, 'dn_超敏C反应蛋白'] = x
    if 'dn_载脂蛋白A' in data.columns:
        x = data['dn_载脂蛋白A'].mean()
        data.loc[data['dn_载脂蛋白A'].isna().values == True, 'dn_载脂蛋白A'] = x
    if 'dn_载脂蛋白B' in data.columns:
        x = data['dn_载脂蛋白B'].mean()
        data.loc[data['dn_载脂蛋白B'].isna().values == True, 'dn_载脂蛋白B'] = x
    if 'dn_间接胆红素' in data.columns:
        x = data['dn_间接胆红素'].mean()
        data.loc[data['dn_间接胆红素'].isna().values == True, 'dn_间接胆红素'] = x
    if 'dn_高密度脂蛋白' in data.columns:
        x = data['dn_高密度脂蛋白'].mean()
        data.loc[data['dn_高密度脂蛋白'].isna().values == True, 'dn_高密度脂蛋白'] = x

    data.to_csv(out_src, header=True, index=False, encoding='utf_8_sig')
    data2 = data.loc[:,
            ['_患者ID', 'dn_NTproBNP', 'dn_中性粒细胞数', 'dn_低密度脂蛋白', 'dn_单核细胞数', 'dn_同型半胱氨酸', 'dn_嗜酸性粒细胞数', 'dn_尿酸', 'dn_总胆固醇',
             'dn_总胆红素',
             'dn_淋巴细胞数', 'dn_甘油三酯', 'dn_白细胞总数', 'dn_直接胆红素', 'dn_空腹血糖', 'dn_糖基化血红蛋白', 'dn_红细胞体积分布宽度', 'dn_红细胞平均体积',
             'dn_红细胞总数', 'dn_红细胞比积', 'dn_肌酐', 'dn_肌酸激酶', 'dn_肌酸激酶同工酶', 'dn_肌钙蛋白I', 'dn_肌钙蛋白T', 'dn_血小板分布宽度', 'dn_血小板总数',
             'dn_血小板比容',
             'dn_血清总蛋白', 'dn_血清白蛋白', 'dn_血红蛋白', 'dn_谷丙转氨酶', 'dn_超敏C反应蛋白', 'dn_载脂蛋白A', 'dn_载脂蛋白B', 'dn_间接胆红素',
             'dn_高密度脂蛋白']]
    data2.to_csv('../database/bloodtest.csv', header=True, index=False, encoding='utf_8_sig', float_format='%.2f')


''' 药物情况 '''


def medicine_preprocess():
    print('药物情况数据预处理中...')
    data = pd.read_csv('../database/patient.csv', encoding='utf-8', low_memory=False)
    if 'dt_ACEI' in data.columns:
        data.loc[(data['dt_ACEI'] == '雷米普利') | (data['dt_ACEI'] == '洛丁新') | (data['dt_ACEI'] == '雅施达') | (
                data['dt_ACEI'] == '依那普利（伊苏）'), 'dt_ACEI'] = 1
        data.loc[(data['dt_ACEI'] == '无') | (data['dt_ACEI'] == '否'), 'dt_ACEI'] = 0
        data.loc[data['dt_ACEI'].isna().values == True, 'dt_ACEI'] = 2
    if 'dt_ADP受体拮抗剂' in data.columns:
        for i in data['dt_ADP受体拮抗剂'].unique():
            if (i == '无') | (i == '否'):
                data.loc[data['dt_ADP受体拮抗剂'] == i, 'dt_ADP受体拮抗剂'] = 0
            elif i != i:
                data.loc[data['dt_ADP受体拮抗剂'].isna().values == True, 'dt_ADP受体拮抗剂'] = 2
            else:
                data.loc[data['dt_ADP受体拮抗剂'] == i, 'dt_ADP受体拮抗剂'] = 1
    if 'dt_ARB' in data.columns:
        for i in data['dt_ARB'].unique():
            if (i == '无') | (i == '否'):
                data.loc[data['dt_ARB'] == i, 'dt_ARB'] = 0
            elif i != i:
                data.loc[data['dt_ARB'].isna().values == True, 'dt_ARB'] = 2
            else:
                data.loc[data['dt_ARB'] == i, 'dt_ARB'] = 1
    if 'dt_GP2b3a抑制剂' in data.columns:
        for i in data['dt_GP2b3a抑制剂'].unique():
            if (i == '无') | (i == '否'):
                data.loc[data['dt_GP2b3a抑制剂'] == i, 'dt_GP2b3a抑制剂'] = 0
            elif i != i:
                data.loc[data['dt_GP2b3a抑制剂'].isna().values == True, 'dt_GP2b3a抑制剂'] = 2
            else:
                data.loc[data['dt_GP2b3a抑制剂'] == i, 'dt_GP2b3a抑制剂'] = 1
    if 'dt_β受体阻断剂' in data.columns:
        for i in data['dt_β受体阻断剂'].unique():
            if (i == '无') | (i == '否'):
                data.loc[data['dt_β受体阻断剂'] == i, 'dt_β受体阻断剂'] = 0
            elif i != i:
                data.loc[data['dt_β受体阻断剂'].isna().values == True, 'dt_β受体阻断剂'] = 2
            else:
                data.loc[data['dt_β受体阻断剂'] == i, 'dt_β受体阻断剂'] = 1
    if 'dt_他汀' in data.columns:
        for i in data['dt_他汀'].unique():
            if (i == '无') | (i == '否'):
                data.loc[data['dt_他汀'] == i, 'dt_他汀'] = 0
            elif i != i:
                data.loc[data['dt_他汀'].isna().values == True, 'dt_他汀'] = 2
            else:
                data.loc[data['dt_他汀'] == i, 'dt_他汀'] = 1
    # if 'dt_依折麦布' in data.columns:
    #     data.loc[data['dt_依折麦布'] == 'dt_依折麦布', 'dt_依折麦布'] = 1
    #     data.loc[(data['dt_依折麦布'] == '无') | (data['dt_依折麦布'] == '否'), 'dt_依折麦布'] = 0
    #     data.loc[data['dt_依折麦布'].isna().values == True, 'dt_依折麦布'] = 2
    if 'dt_依折麦布' in data.columns:
        for i in data['dt_依折麦布'].unique():
            if (i == '无') | (i == '否'):
                data.loc[data['dt_依折麦布'] == i, 'dt_依折麦布'] = 0
            elif i != i:
                data.loc[data['dt_依折麦布'].isna().values == True, 'dt_依折麦布'] = 2
            else:
                data.loc[data['dt_依折麦布'] == i, 'dt_依折麦布'] = 1
    if 'dt_口服抗凝药' in data.columns:
        for i in data['dt_口服抗凝药'].unique():
            if (i == '无') | (i == '否'):
                data.loc[data['dt_口服抗凝药'] == i, 'dt_口服抗凝药'] = 0
            elif i != i:
                data.loc[data['dt_口服抗凝药'].isna().values == True, 'dt_口服抗凝药'] = 2
            else:
                data.loc[data['dt_口服抗凝药'] == i, 'dt_口服抗凝药'] = 1
    if 'dt_抗凝药物' in data.columns:
        for i in data['dt_抗凝药物'].unique():
            if (i == '无') | (i == '否'):
                data.loc[data['dt_抗凝药物'] == i, 'dt_抗凝药物'] = 0
            elif i != i:
                data.loc[data['dt_抗凝药物'].isna().values == True, 'dt_抗凝药物'] = 2
            else:
                data.loc[data['dt_抗凝药物'] == i, 'dt_抗凝药物'] = 1
    if 'dt_溶栓药' in data.columns:
        for i in data['dt_溶栓药'].unique():
            if (i == '无') | (i == '否'):
                data.loc[data['dt_溶栓药'] == i, 'dt_溶栓药'] = 0
            elif i != i:
                data.loc[data['dt_溶栓药'].isna().values == True, 'dt_溶栓药'] = 2
            else:
                data.loc[data['dt_溶栓药'] == i, 'dt_溶栓药'] = 1
    if 'dt_螺内酯' in data.columns:
        for i in data['dt_螺内酯'].unique():
            if (i == '无') | (i == '否'):
                data.loc[data['dt_螺内酯'] == i, 'dt_螺内酯'] = 0
            elif i != i:
                data.loc[data['dt_螺内酯'].isna().values == True, 'dt_螺内酯'] = 2
            else:
                data.loc[data['dt_螺内酯'] == i, 'dt_螺内酯'] = 1
    if 'dt_钙离子拮抗剂' in data.columns:
        for i in data['dt_钙离子拮抗剂'].unique():
            if (i == '无') | (i == '否'):
                data.loc[data['dt_钙离子拮抗剂'] == i, 'dt_钙离子拮抗剂'] = 0
            elif i != i:
                data.loc[data['dt_钙离子拮抗剂'].isna().values == True, 'dt_钙离子拮抗剂'] = 2
            else:
                data.loc[data['dt_钙离子拮抗剂'] == i, 'dt_钙离子拮抗剂'] = 1
    if 'dt_阿司匹林' in data.columns:
        data.loc[data['dt_阿司匹林'] == '是', 'dt_阿司匹林'] = 1
        data.loc[(data['dt_阿司匹林'] == '无') | (data['dt_阿司匹林'] == '否'), 'dt_阿司匹林'] = 0
        data.loc[data['dt_阿司匹林'].isna().values == True, 'dt_阿司匹林'] = 2

        # data.loc[(data['dt_ACEI'] == '雷米普利')| (data['dt_ACEI'] == '洛丁新') | (data['dt_ACEI'] == '雅施达') | (data['dt_ACEI'] == '依那普利（伊苏）'), 'dn_NTproBNP'] = 1
        # data.loc[(data['dt_ADP受体拮抗剂'] == '无') | (data['dt_ADP受体拮抗剂'] == '否'), 'dn_NTproBNP'] = 0
        # data.loc[data['dt_ADP受体拮抗剂'].isna().values == True, 'dt_ADP受体拮抗剂'] = 2

    data.to_csv(out_src, header=True, index=False, encoding='utf_8_sig')
    data2 = data.loc[:,
            ['_患者ID', 'dt_ACEI', 'dt_ADP受体拮抗剂', 'dt_ARB', 'dt_GP2b3a抑制剂', 'dt_β受体阻断剂', 'dt_他汀', 'dt_依折麦布', 'dt_口服抗凝药',
             'dt_抗凝药物',
             'dt_溶栓药', 'dt_螺内酯', 'dt_钙离子拮抗剂', 'dt_阿司匹林']]
    data2.to_csv('../database/medicine.csv', header=True, index=False, encoding='utf_8_sig', float_format='%.2f')
    # print(data['_体重指数'][data['_体重指数'] <= 5])
    # print(data['_体重指数'][data['_体重指数'] >= 50])
    # print(data['CABG史（冠脉搭桥术）'].isna().sum())
    # print(data['dt_吸烟'].dtype)
    print('==========================')
    print(data['dt_ACEI'].describe())
    # print(data.columns)
    # print(data.head())
    print(data.size)
    print(data.shape)
    print(len(data))
    print('==========================')


''' 心电图 '''


def electrocardiogram_preprocess():
    print('心电图数据预处理中')
    data = pd.read_csv('../database/patient.csv', encoding='utf-8', low_memory=False)
    # if 'dn_QRS时限' in data.columns:
    #     data.loc[(data['dt_ACEI'] == '无') | (data['dt_ACEI'] == '否'), 'dt_ACEI'] = 0
    #     data.loc[data['dt_ACEI'].isna().values == True, 'dt_ACEI'] = 2
    if 'dn_QRS时限' in data.columns:
        data.loc[data['dn_QRS时限'] < 10, 'dn_QRS时限'] = 10
        data.loc[data['dn_QRS时限'] >= 1000, 'dn_QRS时限'] = 1000
        x = data['dn_QRS时限'].mean()
        data.loc[data['dn_QRS时限'].isna().values == True, 'dn_QRS时限'] = x
    if 'dn_QT间期' in data.columns:
        data.loc[data['dn_QT间期'] < 10, 'dn_QT间期'] = 10
        data.loc[data['dn_QT间期'] >= 1000, 'dn_QT间期'] = 1000
        x = data['dn_QT间期'].mean()
        data.loc[data['dn_QT间期'].isna().values == True, 'dn_QT间期'] = x
    if 'dn_心率' in data.columns:
        data.loc[data['dn_心率'] < 0, 'dn_心率'] = 0
        data.loc[data['dn_心率'] >= 1000, 'dn_心率'] = 1000
        x = data['dn_心率'].mean()
        data.loc[data['dn_心率'].isna().values == True, 'dn_心率'] = x
    if 'dn_肺动脉压' in data.columns:
        data.loc[data['dn_肺动脉压'] < 0, 'dn_肺动脉压'] = 0
        data.loc[data['dn_肺动脉压'] >= 1000, 'dn_肺动脉压'] = 1000
        x = data['dn_肺动脉压'].mean()
        data.loc[data['dn_肺动脉压'].isna().values == True, 'dn_肺动脉压'] = x

    data.to_csv(out_src, header=True, index=False, encoding='utf_8_sig')
    data2 = data.loc[:,
            ['_患者ID', 'dn_QRS时限', 'dn_QT间期', 'dn_心率', 'dn_肺动脉压']]
    data2.to_csv('../database/electrocardiogram.csv', header=True, index=False, encoding='utf_8_sig', float_format='%.2f')



''' 心脏彩超 '''


def heartindex_preprocess():
    print('心脏彩超数据预处理中')
    data = pd.read_csv('../database/patient.csv', encoding='utf-8', low_memory=False)
    if 'A波速率' in data.columns:
        data.loc[data['A波速率'] < 0, 'A波速率'] = 0
        data.loc[data['A波速率'] >= 1000, 'A波速率'] = 1000
        x = data['A波速率'].mean()
        data.loc[data['A波速率'].isna().values == True, 'A波速率'] = x
    if 'E\'波速率' in data.columns:
        data.loc[data['E\'波速率'] < 0, 'E\'波速率'] = 0
        data.loc[data['E\'波速率'] >= 1000, 'E\'波速率'] = 1000
        x = data['E\'波速率'].mean()
        data.loc[data['E\'波速率'].isna().values == True, 'E\'波速率'] = x
    if 'E/E\'比值' in data.columns:
        data.loc[data['E/E\'比值'] < 0, 'E/E\'比值'] = 0
        data.loc[data['E/E\'比值'] >= 1000, 'E/E\'比值'] = 1000
        x = data['E/E\'比值'].mean()
        data.loc[data['E/E\'比值'].isna().values == True, 'E/E\'比值'] = x
    if 'dn_EA比值' in data.columns:
        data.loc[data['dn_EA比值'] < 0, 'dn_EA比值'] = 0
        data.loc[data['dn_EA比值'] >= 1000, 'dn_EA比值'] = 1000
        x = data['dn_EA比值'].mean()
        data.loc[data['dn_EA比值'].isna().values == True, 'dn_EA比值'] = x
    if 'dn_E波速率' in data.columns:
        data.loc[data['dn_E波速率'] < 0, 'dn_E波速率'] = 0
        data.loc[data['dn_E波速率'] >= 1000, 'dn_E波速率'] = 1000
        x = data['dn_E波速率'].mean()
        data.loc[data['dn_E波速率'].isna().values == True, 'dn_E波速率'] = x
    if 'dn_右室内径' in data.columns:
        data.loc[data['dn_右室内径'] < 0, 'dn_右室内径'] = 0
        data.loc[data['dn_右室内径'] >= 1000, 'dn_右室内径'] = 1000
        x = data['dn_右室内径'].mean()
        data.loc[data['dn_右室内径'].isna().values == True, 'dn_右室内径'] = x
    if 'dn_室间隔厚度' in data.columns:
        data.loc[data['dn_室间隔厚度'] < 0, 'dn_室间隔厚度'] = 0
        data.loc[data['dn_室间隔厚度'] >= 500, 'dn_室间隔厚度'] = 500
        x = data['dn_室间隔厚度'].mean()
        data.loc[data['dn_室间隔厚度'].isna().values == True, 'dn_室间隔厚度'] = x
    if 'dn_左室射血分数' in data.columns:
        data.loc[data['dn_左室射血分数'] < 0, 'dn_左室射血分数'] = 0
        data.loc[data['dn_左室射血分数'] >= 1000, 'dn_左室射血分数'] = 1000
        x = data['dn_左室射血分数'].mean()
        data.loc[data['dn_左室射血分数'].isna().values == True, 'dn_左室射血分数'] = x
    if 'dn_左室收缩末期内径' in data.columns:
        data.loc[data['dn_左室收缩末期内径'] < 0, 'dn_左室收缩末期内径'] = 0
        data.loc[data['dn_左室收缩末期内径'] >= 1000, 'dn_左室收缩末期内径'] = 1000
        x = data['dn_左室收缩末期内径'].mean()
        data.loc[data['dn_左室收缩末期内径'].isna().values == True, 'dn_左室收缩末期内径'] = x
    if 'dn_左室舒张末期内径' in data.columns:
        data.loc[data['dn_左室舒张末期内径'] < 0, 'dn_左室舒张末期内径'] = 0
        data.loc[data['dn_左室舒张末期内径'] >= 1000, 'dn_左室舒张末期内径'] = 1000
        x = data['dn_左室舒张末期内径'].mean()
        data.loc[data['dn_左室舒张末期内径'].isna().values == True, 'dn_左室舒张末期内径'] = x
    if 'dn_左室重量' in data.columns:
        data.loc[data['dn_左室重量'] < 0, 'dn_左室重量'] = 0
        data.loc[data['dn_左室重量'] >= 1000, 'dn_左室重量'] = 1000
        x = data['dn_左室重量'].mean()
        data.loc[data['dn_左室重量'].isna().values == True, 'dn_左室重量'] = x
    if 'dn_左室重量指数' in data.columns:
        data.loc[data['dn_左室重量指数'] < 0, 'dn_左室重量指数'] = 0
        data.loc[data['dn_左室重量指数'] >= 1000, 'dn_左室重量指数'] = 1000
        x = data['dn_左室重量指数'].mean()
        data.loc[data['dn_左室重量指数'].isna().values == True, 'dn_左室重量指数'] = x
    if 'dn_左房内径' in data.columns:
        data.loc[data['dn_左房内径'] < 0, 'dn_左房内径'] = 0
        data.loc[data['dn_左房内径'] >= 1000, 'dn_左房内径'] = 1000
        x = data['dn_左房内径'].mean()
        data.loc[data['dn_左房内径'].isna().values == True, 'dn_左房内径'] = x
    if 'dn_短轴缩短率' in data.columns:
        data.loc[data['dn_短轴缩短率'] < 0, 'dn_短轴缩短率'] = 0
        data.loc[data['dn_短轴缩短率'] >= 1000, 'dn_短轴缩短率'] = 1000
        x = data['dn_短轴缩短率'].mean()
        data.loc[data['dn_短轴缩短率'].isna().values == True, 'dn_短轴缩短率'] = x

    # 短期入院285，长期入院1952,死亡143
    if 'a患者.B短期随访::dt_短期再入院' in data.columns:
        data.loc[(data['a患者.B短期随访::dt_短期再入院'] == '是') & (data['dlt_死亡终点'] != '是'), 'dlt_死亡终点'] = '短期复发'
        # data.loc[data['a患者.B短期随访::dt_短期再入院'] == '否', 'dlt_死亡终点'] = '否'
    if 'a患者.B长期随访::dt_L再入院' in data.columns:
        data.loc[(data['a患者.B长期随访::dt_L再入院'] == '是') & (data['dlt_死亡终点'] != '是'), 'dlt_死亡终点'] = '长期复发'
        # data.loc[data['a患者.B长期随访::dt_L再入院'] == '否', 'dlt_死亡终点'] = '否'
    # if 'dlt_死亡终点' in data.columns:
    #     data.loc[data['dlt_死亡终点'] == '是', 'dlt_死亡终点'] = '是'
    #     data.loc[data['dlt_死亡终点'] == '否', 'dlt_死亡终点'] = 0
    #     data.loc[data['dlt_死亡终点'].isna().values == True, 'dlt_死亡终点'] = 2

    data.to_csv(out_src, header=True, index=False, encoding='utf_8_sig')
    data2 = data.loc[:,
            ['_患者ID', 'A波速率', 'E\'波速率', 'E/E\'比值', 'dn_EA比值', 'dn_E波速率', 'dn_右室内径',
             'dn_室间隔厚度', 'dn_左室射血分数', 'dn_左室收缩末期内径', 'dn_左室舒张末期内径', 'dn_左室重量', 'dn_左室重量指数', 'dn_左房内径', 'dn_短轴缩短率',
             'dlt_死亡终点']]
    data2.to_csv('../database/heartindex.csv', header=True, index=False, encoding='utf_8_sig',
                 float_format='%.2f')

    inputfile_dir = ['../database/info.csv', '../database/disease.csv', '../database/bloodtest.csv',
                     '../database/medicine.csv',
                     '../database/electrocardiogram.csv', '../database/heartindex.csv']
    outputfile = '../database/1.csv'
    # for inputfile in inputfile_dir:
    #     data = pd.read_csv(inputfile, header=None)
    #     data.to_csv(outputfile, mode='a', header=True, index=False, encoding='utf_8_sig')

    df1 = pd.read_csv(inputfile_dir[0])
    for file in inputfile_dir[1:]:
        df2 = pd.read_csv(file)  # 打开csv文件，注意编码问题，保存到df2中
        # df1 = pd.merge(df1, df2, on='_患者ID')
        df1 = pd.merge(df1, df2, on='_患者ID', left_index=True, right_index=True)

        # df1.append(df2.loc[:, 1:])
        # df1 = pd.concat([df1, df2.loc[:, 1:]], axis=1, ignore_index=True)  # 将df2数据与df1合并
    # df1 = df1.drop_duplicates()  # 去重
    # df1 = df1.reset_index(drop=True)  # 重新生成index
    df1.to_csv(outputfile, header=True, index=False, encoding='utf_8_sig')
    print(df1.shape)




# if '年龄' in data.columns:
#     data.loc[data['年龄']>100 || data['年龄']<100] = 0
# print(data['性别'].isnull().sum())
# print(data[data['性别'].isnull().values == True])

# mid = data['ID号']   #取备注列的值
# data.pop('ID号')  #删除备注列
# data.insert(0, 'ID号', mid) #插入备注列
# data.dropna(axis=0, how='all', inplace=True)

def data_preprocess():
    print('开始数据预处理...')
    data = pd.read_excel(database_src, sheet_name='Sheet1')

    # 去重和删除ID号不存在的数据
    data.dropna(axis=0, how='all', subset=['ID号'], inplace=True)
    print(data.describe())
    print(data.shape)
    if '_患者ID' in data.columns:
        data.loc[data['_患者ID'].isna().values == True, '_患者ID'] = 533
    # 去除列中数量不足1000个的字段
    data.dropna(axis=1, how='all', thresh=100, inplace=True)
    # data.drop_duplicates(subset=['ID号'], keep='last', inplace=True)
    data = data.drop(columns=['姓名', '出生日期', '手术日期', '住院号', '联系电话1', '_联系电话2', '住院号', 'dt_所属医院'])
    # data.to_csv(out_src, header=True, index=False, encoding='utf_8_sig')

    info_preprocess(data)
    disease_preprocess()
    bloodtest_preprocess()
    medicine_preprocess()
    electrocardiogram_preprocess()
    heartindex_preprocess()
    print('数据预处理结束...')

    # data.to_csv(out_src, header=True, index=False, encoding='utf_8_sig')


if __name__ == '__main__':
    data_preprocess()
    '''异常值检测，画箱型图'''
    # plt.figure()
    # p = data.boxplot(return_type='dict')
    # x = p['fliers'][0].get_xdata()
    # y = p['fliers'][0].get_ydata()
    # y.sort()
    #
    # for i in range(len(x)):
    #     if i>0:
    #         plt.annotate(y[i], xy=(x[i], y[i]), xytext=(x[i]+0.05 - 0.8/(y[i]-y[i-1]), y[i]))
    #     else:
    #         plt.annotate(y[i], xy=(x[i], y[i]), xytext=(x[i]+0.08, y[i]))
    # plt.show()

    # 利用四分位间距检测异常值
    # for col in ['dn_尿酸']:
    #     print('*' * 50)
    #     print(col)
    #     # 求上四分位数
    #     q_75 = data[col].quantile(q=0.75)
    #     print('上四分位数:', q_75)
    #     # 求下四分位数
    #     q_25 = data[col].quantile(q=0.25)
    #     print('下四分位数:', q_25)
    #     # 求四分位间距
    #     d = q_75 - q_25
    #     print('四分位数间距:', d)
    #     # 求数据上界和数据下界
    #     data_top = q_75 + 1.5 * d
    #     data_bottom = q_25 - 1.5 * d
    #     print('数据上界:', data_top)
    #     print('数据下界:', data_bottom)
    #     # 查看异常值的数量
    #     print('异常值的个数：', len(data[(data[col] > data_top) | (data[col] < 0)]))

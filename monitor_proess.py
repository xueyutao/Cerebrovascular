##
# import pymysql
import pandas as pd
import uuid
import time,datetime
import numpy as np
from dateutil.relativedelta import relativedelta
# from preprocess.data_entity import DataEntity

mysql_conn = {}


# 导入预处理好的csv数据文件到mysql数据库中
def import_to_mysql(cursor, file_url):
    file_df = pd.read_csv(file_url)
    total = 0
    err_cnt = 0
    print('---------开始导入-------------')
    for i, row in file_df.iterrows():
        # nan 设置为0
        nan_index = row[row != row].index
        row[nan_index] = 0

        # # 设置数值缺失标志位
        # flag_row = row.copy()
        # flag_row[:] = 0
        # flag_row[nan_index] = 1

        row_id = uuid.uuid1()
        sql = 'insert into HourData(id,siteName,time,ph,TN,TP,NH,temper,turbi,doxygen,conduct,permanga,' \
              'phFlag,TNFlag,TPFlag,NHFlag,temperFlag,turbiFlag,doxygenFlag,conductFlag,permangaFlag)' \
              'values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
        args = (row_id, row['站点名称'], row['监测时间'], row['pH值'], row['总氮'], row['总磷'], row['氨氮'],
                row['水温'], row['浑浊度'], row['溶解氧'], row['电导率'], row['高锰酸盐指数'],
                row['phFlag'], row['TNFlag'], row['TPFlag'], row['NHFlag'], row['temperFlag'],
                row['turbiFlag'], row['doxygenFlag'], row['conductFlag'], row['permangaFlag'])
        try:
            cursor.execute(sql, args)
        except Exception as e:
            print('发生一条错误，继续插入数据:', end='')
            print(e)
            total = total - 1
            err_cnt = err_cnt + 1
        total = total + 1
        if i > 0 and (i % 1000 == 0):
            mysql_conn.commit()
            print(str(i) + ' row inserted')

    print('----------导入结束------------')
    print(str(total) + '成功 ' + str(err_cnt) + '失败')

    mysql_conn.commit()
    cursor.close()
    mysql_conn.close()


# 计算数据
# compute_type 可选：D W M 分别代表 日，周，月
# 计算日数据时，begin必须是一天的起始时间
# 计算周数据时，begin必须是周的起始时间
# 计算月数据时，begin必须是月的起始时间
def compute_data(cursor, begin, end, compute_type='D'):

    start_time = datetime.datetime.fromisoformat(begin)
    end_time = datetime.datetime.fromisoformat(end)

    # 获取所有站点
    sql = 'SELECT DISTINCT siteName FROM HourData'
    cursor.execute(sql)
    mysql_conn.commit()
    all_site = cursor.fetchall()
    for site in all_site:
        site_name = site[0]
        print(site_name)

        time_left = start_time
        while time_left.timestamp() <= end_time.timestamp():
            if compute_type == 'D':
                time_right = time_left + relativedelta(days=1)
            elif compute_type == 'W':
                time_right = time_left + relativedelta(weeks=1)
            elif compute_type == 'M':
                time_right = time_left + relativedelta(months=1)

            sql = 'SELECT * from HourData WHERE UNIX_TIMESTAMP(time) >= %s and UNIX_TIMESTAMP(time) < %s'
            arg = (time_left.timestamp() + 8 * 3600, time_right.timestamp() + 8 * 3600)
            cursor.execute(sql, arg)
            mysql_conn.commit()
            day_data_list = cursor.fetchall()
            entity_list = DataEntity.from_sql_result_list(day_data_list)

            # 计算平均值
            avg_list = DataEntity.compute_avg(entity_list)
            # 插入数据库
            sql = 'insert into ComputeAvgData(id,siteName,time,' \
                  'ph,TN,TP,NH,temper,turbi,doxygen,conduct,permanga,' \
                  'dataType)' \
                  'values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'
            # 获取uuid
            row_id = uuid.uuid1()

            args = [row_id, site_name, time_left.isoformat()]
            args = args + avg_list
            args.append(compute_type)
            arg = tuple(args)

            print(time_left.isoformat())

            cursor.execute(sql, arg)
            mysql_conn.commit()

            if compute_type == 'D':
                time_left = time_left + relativedelta(days=1)
            elif compute_type == 'W':
                time_left = time_left + relativedelta(weeks=1)
            elif compute_type == 'M':
                time_left = time_left + relativedelta(months=1)


# 预处理0 缺失时间补充
def data_time_complete():
    pass


# 预处理1，合并文件
# file_list: 文件列表
def data_fix_concat(file_list, dist):
    df_list = []
    for file in file_list:
        print('处理文件：' + file)
        file1_df = pd.read_excel(file, usecols=[1, 2, 3, 4, 5, 6], dtype=object)
        print(file1_df.columns)

        if '因子代码' in file1_df.columns:
            file1_df.loc[file1_df['因子代码'] == 'W01001', '因子名称'] = 'pH值'  # “pH”替换为“pH值”
            file1_df.loc[file1_df['因子代码'] == 'W01003', '因子名称'] = '浑浊度'  # “浊度”替换为“浑浊度”
            file1_df.loc[file1_df['因子名称'] == '浊度', '因子名称'] = '浑浊度'

        if '站点编码' in file1_df.columns:
            file1_df.loc[file1_df['站点编码'] == '3506000005WQ', '站点名称'] = '芗城水利局站'  # 将“芗城区水利局站”和“舟尾亭水闸”替换为“芗城水利局站”
            file1_df.loc[file1_df['站点编码'] == '3506000002WQ', '站点名称'] = '北京路水闸站'  # 规范“北京路水闸”为“北京路水闸站”
            file1_df.loc[file1_df['站点编码'] == '3506000004WQ', '站点名称'] = '中山桥水闸站'  # 规范“中山桥水闸”为“中山桥水闸站”
        # 删除两条无关数据
        if '因子代码' in file1_df.columns:
            file1_df = file1_df[~file1_df['因子代码'].isin(['W01017'])]
            file1_df = file1_df[~file1_df['因子代码'].isin(['W01018'])]

        file1_df.drop_duplicates(inplace=True)  # 删除重复数据
        df_list.append(file1_df)
    file_df = pd.concat(df_list, ignore_index=True)
    file_df.to_csv(dist, header=True, index=False, encoding='utf_8_sig')


# 预处理2 转置数据
def data_transpose(file_in='./first.csv', dataTrans="./second.csv"):
    print('转置数据...')
    data_df = pd.read_csv(file_in, encoding='utf-8', dtype=object)
    data_df.drop(['站点编码', '监测因子编码'], inplace=True, axis=1, errors='ignore')  # 删除“站点编码”、“监测因子编码”两个属性
    # 转置“因子名称”、“数值”两个属性
    data_df['监测时间'] = pd.to_datetime(data_df['监测时间'])
    data_df['数值'] = data_df['数值'].apply(pd.to_numeric, errors='coerce')
    data_df = data_df.pivot_table(index=['监测时间', '站点名称'], columns='因子名称', values='数值')
    data_df.reset_index(inplace=True)
    data_df.to_csv(dataTrans, header=True, index=False, encoding='utf_8_sig')
    print('转置数据完成...')


# 预处理3 插入缺失时间
def data_insert_time(file_in='./msecond2020.csv', file_out='./mthird.csv'):
    data2020_df = pd.read_csv(file_in, encoding='utf-8', dtype=object,
                              usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    data2020_df.head(5)
    site_list = data2020_df['站点名称'].unique().tolist()

    all_df = pd.DataFrame()
    first = True

    date_index = pd.date_range('2020-01-01 00:00:00', '2020-12-31 23:00:00', freq='H')
    for site in site_list:
        site_df = data2020_df.loc[data2020_df['站点名称'] == site]
        for i in range(len(site_df)):
            row_time = datetime.datetime.strptime(site_df.iloc[i, 0], '%Y-%m-%d %H:%M:%S')
            site_df.iloc[i, 0] = row_time
        site_df = site_df.set_index('监测时间')
        site_df = site_df.reindex(date_index)
        site_df['站点名称'][site_df['站点名称'].isna()] = site

        all_df = all_df.append(site_df) if not first else site_df
        first = False

    all_df.to_csv(file_out, header=True, index=True, encoding='utf_8_sig')


# 预处理4 数据缺失处理，异常处理
def data_clean(file_in='./mthird.csv', file_out='./mforth.csv'):
    print('数据缺失处理...')
    # 缺失值、异常值处理
    data_df = pd.read_csv(file_in, encoding='utf-8', dtype=object)
    # data_df = pd.read_csv(path + dataTrans, encoding='utf-8', dtype=object) index_col=['站点名称', '监测时间'],
    data_df['phFlag'] = 0
    data_df['TNFlag'] = 0
    data_df['TPFlag'] = 0
    data_df['NHFlag'] = 0
    data_df['temperFlag'] = 0
    data_df['turbiFlag'] = 0
    data_df['doxygenFlag'] = 0
    data_df['conductFlag'] = 0
    data_df['permangaFlag'] = 0
    data_df.sort_values(by=['站点名称', '监测时间'], inplace=True)
    # data_df = data_df.astype(float)
    data_df['pH值'] = data_df['pH值'].astype(float)
    data_df['总氮'] = data_df['总氮'].astype(float)
    data_df['总磷'] = data_df['总磷'].astype(float)
    data_df['氨氮'] = data_df['氨氮'].astype(float)
    data_df['水温'] = data_df['水温'].astype(float)
    data_df['浑浊度'] = data_df['浑浊度'].astype(float)
    data_df['溶解氧'] = data_df['溶解氧'].astype(float)
    data_df['电导率'] = data_df['电导率'].astype(float)
    data_df['高锰酸盐指数'] = data_df['高锰酸盐指数'].astype(float)

    fill_method = 'nearest'

    data_df['pH值'] = np.where(data_df['pH值'] > 50, data_df['pH值'] * 0.1, data_df['pH值'])  # 50<pH值<90：小数点左移一位
    data_df['pH值'][data_df['pH值'] > 14] = None  # pH值>14：视为缺失值，和缺失值一起处理
    data_df['phFlag'][data_df['pH值'].isna()] = 1
    # data_df['pH值'].fillna(method=fill_method, inplace=True)  # 用前一时刻的值填充缺失值
    data_df['pH值'].interpolate(method=fill_method, inplace=True)

    data_df['水温'][data_df['水温'] == 0] = None  # 水温=0：视为缺失值，和缺失值一起处理
    data_df['temperFlag'][data_df['水温'].isna()] = 1
    data_df['水温'].interpolate(method=fill_method, inplace=True)  # 用前一时刻的值填充缺失值

    data_df['turbiFlag'][data_df['浑浊度'].isna()] = 1
    data_df['浑浊度'].interpolate(method=fill_method, inplace=True)  # 用前一时刻的值填充缺失值

    data_df['溶解氧'][data_df['溶解氧'] >= 14.64] = None  # 溶解氧>14.64：视为缺失值，和缺失值一起处理
    data_df['doxygenFlag'][data_df['溶解氧'].isna()] = 1
    data_df['溶解氧'].interpolate(method=fill_method, inplace=True)  # 用前一时刻的值填充缺失值

    data_df['conductFlag'][data_df['电导率'].isna()] = 1
    data_df['电导率'].interpolate(method=fill_method, inplace=True)  # 用前一时刻的值填充缺失值

    # data_df.reset_index(inplace=True)
    data_df['总氮'][data_df['总氮'] <= 0] = None  # 总氮=0：视为缺失值，和缺失值一起处理
    data_df['总氮'][data_df['总氮'] > 100] = None  # 总氮>100：视为缺失值，和缺失值一起处理
    data_df['TNFlag'][data_df['总氮'].isna()] = 1
    data_df['总氮'].interpolate(method=fill_method, inplace=True)  # 用前一时刻的值填充缺失值

    data_df['氨氮'] = np.where(data_df['氨氮'] < 0, data_df['氨氮'] * -1, data_df['氨氮'])  # 负数转正
    data_df['氨氮'][data_df['氨氮'] == 0] = None  # 氨氮=0：视为缺失值，和缺失值一起处理
    # data_df['氨氮'] = np.where(data_df['氨氮'] > data_df['总氮'], None, data_df['氨氮'])
    data_df['NHFlag'][data_df['氨氮'].isna()] = 1
    data_df['氨氮'].interpolate(method=fill_method, inplace=True)  # 用前一时刻的值填充缺失值

    data_df['总磷'] = np.where(data_df['总磷'] < 0, data_df['总磷'] * -1, data_df['总磷'])  # 负数转正
    data_df['总磷'][data_df['总磷'] == 0] = None  # 总磷=0：视为缺失值，和缺失值一起处理
    data_df['总磷'] = np.where(data_df['总磷'] > 5, data_df['总磷'] * 0.1, data_df['总磷'])  # 总磷>5：小数点左移一位
    data_df['TPFlag'][data_df['总磷'].isna()] = 1
    data_df['总磷'].interpolate(method=fill_method, inplace=True)  # 用前一时刻的值填充缺失值

    data_df['高锰酸盐指数'] = np.where(data_df['高锰酸盐指数'] < 0, data_df['高锰酸盐指数'] * -1,
                                    data_df['高锰酸盐指数'])  # 负数转正
    data_df['高锰酸盐指数'][data_df['高锰酸盐指数'] == 0] = None  # 高锰酸盐指数=0：视为缺失值，和缺失值一起处理
    data_df['permangaFlag'][data_df['高锰酸盐指数'].isna()] = 1
    data_df['高锰酸盐指数'].interpolate(method=fill_method, inplace=True)  # 用前一时刻的值填充缺失值
    data_df.reset_index(inplace=True)

    data_df.to_csv(file_out, header=True, index=False, encoding='utf_8_sig')
    print('数据缺失处理完成')


if __name__ == '__main__':
    # 链接MySQL
    # mysql_conn = pymysql.connect(
    #     host="172.17.171.8",
    #     user="root",
    #     password="123456",
    #     database="waterAna",
    #     charset="utf8mb4")
    # mysql_cursor = mysql_conn.cursor()

    # # 预处理监测站点数据
    # import os
    # root = "E:\\project\\major\\water_analysis\\data\\data2021\\monitor"
    # file_names = [os.path.join(path, name) for path, subdirs, files in os.walk(root) for name in files]
    # # # 步骤1：
    # data_fix_concat(file_names,'./mfirst.csv')
    # # # 步骤2：
    # data_transpose('./mfirst.csv', './msecond.csv')
    # # 步骤3：
    # data_insert_time('./mthird.csv','./')

    # # 步骤4
    data_clean('./msecond4.csv', './mthird.csv')

    # 格式转换
    # data_to_ts('./mthird.csv','./mforth.csv')
    #
    # 导入到mysql数据库
    # import_to_mysql(mysql_cursor, './third.csv')

    # # 计算天数据
    # compute_data(mysql_cursor, '2018-08-03 00:00:00', '2020-06-12 00:00:00', 'D')
    # # 计算周数据,起始时间要填周一
    # compute_data(mysql_cursor, '2018-07-29 00:00:00', '2020-06-12 00:00:00', 'W')
    # # 计算月数据
    # compute_data(mysql_cursor, '2018-08-01 00:00:00', '2020-06-12 00:00:00', 'M')


"""
模拟华泰网格交易做回测

fastapi接口：

导数据：
传入Excel
返回token

回测：
传入token，网格参数
返回结果token，收益盈亏情况

获取结果：
传入结果token
下载交易明细表

获取图示：
传入结果token
返回图片
"""
import json
import os
import pprint
import traceback
import hashlib
import base64

import pandas
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl

import numpy as np
import tqdm

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi_fixer import app

# 设置字体类型
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']

_data_root = 'wp'


@app.post('/load_data')
def load_data(file: bytes = File(...), name: str = "data_token"):
    """
    加载股票数据Excel。
    ```
    :param file: Excel文件。
    :param name: 最后生成的data_token基于name。
    :return:
    ```
    """
    try:
        # file.read()
        # file.filename
        if not os.path.exists(_data_root):
            os.makedirs(_data_root)
        # token = base64.urlsafe_b64encode(hashlib.md5(b'').digest()).decode()[:-2]
        token = hashlib.md5(file).hexdigest()
        fname = f'{name}.csv'
        cnt = 1
        while os.path.exists(os.path.join(_data_root, fname)):
            fname = f'{name}.{token[:cnt]}.csv'
            cnt += 1

        fname_path = os.path.join(_data_root, fname)
        with open(fname_path, 'wb') as f:
            f.write(file)
        return dict(success=1, message='data loaded success.', data=dict(data_token=fname))
    except Exception:
        exc = traceback.format_exc()
        return dict(success=0, message=f'Failed to load data.\n{exc}')


@app.get('/do_trading')
def do_trading(data_token='data_token.csv', data_start_index='0', data_end_index='_', name='grid_token',
               touruzijin=10000, jizhunjia=0, dancifene=100,
               jiancangfene=1000,
               wanggeshangjie=12, wanggexiajie=6, wanggeleixing=1, mairuyuzhi=0.5, maichuyuzhi=0.5,
               shouxufeilv=0.0001):
    """
    网格交易回测。
    ```
    :param data_token: 股票数据的token
    :param data_start_index: 数据起始位置
    :param data_end_index: 数据结束位置
    :param name: 网格参数的token生成基于name
    :param touruzijin: 投入资金
    :param jizhunjia: 基准价格，设置为0则为回测的第一个数据
    :param dancifene: 买入卖出单次交易股数，等股交易
    :param jiancangfene: 起始建仓股数
    :param wanggeshangjie: 网格上界
    :param wanggexiajie: 网格下界
    :param wanggeleixing: 网格类型，1为差价，2为百分比
    :param mairuyuzhi: 买入阈值，差价或百分比
    :param maichuyuzhi: 卖出阈值，差价或百分比
    :param shouxufeilv: 手续费率，默认0.01%，0.1元起
    :return:
    ```
    """
    data_path = os.path.join(_data_root, data_token)
    names, data = parse_excel(data_path)
    if data_start_index.isdigit():
        data_start_index = int(data_start_index)
    else:
        data_start_index = None

    if data_end_index.isdigit():
        data_end_index = int(data_end_index)
    else:
        data_end_index = None
    names = names[data_start_index:data_end_index]
    data = data[data_start_index:data_end_index]

    # grid_path = os.path.join(_data_root, grid_token)
    # if os.path.isfile(grid_path):
    #     grid_params = json.load(open(grid_path, encoding='utf8'))
    # else:
    grid_params = dict(
        data_start_index=data_start_index,  # 起始点
        data_end_index=data_end_index,  # 结束点
        touruzijin=float(touruzijin),  # 投入总资金
        jizhunjia=float(jizhunjia),  # 基准价，0则以初始价为基准价
        dancifene=int(dancifene),  # 单次交易股数
        jiancangfene=int(jiancangfene),  # 建仓股数
        wanggeshangjie=float(wanggeshangjie),  # 网格上界
        wanggexiajie=float(wanggexiajie),  # 网格下界
        wanggeleixing=int(wanggeleixing),  # 网格类型，1为差价，2为百分比
        mairuyuzhi=float(mairuyuzhi),  # 买入阈值
        maichuyuzhi=float(maichuyuzhi),  # 卖出阈值
        shouxufeilv=float(shouxufeilv),  # 手续费率
    )

    params_json = json.dumps(grid_params, ensure_ascii=False, indent=4)
    token = hashlib.md5(params_json.encode('utf8')).hexdigest()
    fname = f'{name}.json'
    cnt = 1
    while os.path.exists(os.path.join(_data_root, fname)):
        fname = f'{name}.{token[:cnt]}.json'
        cnt += 1
    grid_token = fname

    result = grid_func(shuju=data,  # 数据
                       **grid_params)
    # 投入资金：10000，回报总价值：17342.759（其中现金：8740.759，份额价值：8602.000），盈亏比例：73.43%
    # display(shoupan=shoupan, maichu_idx=maichu_idx, mairu_idx=mairu_idx)

    fname_path = os.path.join(_data_root, grid_token)
    with open(fname_path, 'wt') as f:
        f.write(params_json)

    data = dict(data_token=data_token, grid_token=grid_token, grid_params=grid_params, data=data, names=names)
    result_path = os.path.join(_data_root, f'data_token={data_token}.grid_token={grid_token}.json')
    with open(result_path, 'wt', encoding='utf8') as fout:
        out = dict(result=result, data=data, data_token=data_token, grid_token=grid_token)
        json.dump(out, fout, ensure_ascii=False, indent=4)

    return dict(success=1, message='do trading success.',
                data=dict(result=result['report_conclusion'], grid_token=grid_token, data_token=data_token))


@app.get('/do_searching')
def do_searching(data_token='data_token.csv', data_start_index='10000', data_end_index='_', name='grid_token',
                 touruzijin: int = 10000, shouxufeilv: float = 0.0001, n_search: int = 5, topn: int = 10):
    """
    网格交易回测。
    ```
    :param data_token: 股票数据的token
    :param data_start_index: 数据起始位置
    :param data_end_index: 数据结束位置
    :param name: 网格参数的token生成基于name
    :param touruzijin: 投入资金
    :param shouxufeilv: 手续费率，默认0.01%，0.1元起
    :param n_search: 搜索密度，越大耗时越久，默认5
    :param topn: 返回最佳收益的结果数，默认10
    :return:
    ```
    """
    # jizhunjia = 0, dancifene = 100,
    # jiancangfene = 1000,
    # wanggeshangjie = 12, wanggexiajie = 6, wanggeleixing = 1, mairuyuzhi = 0.5, maichuyuzhi = 0.5,
    data_path = os.path.join(_data_root, data_token)
    names, data = parse_excel(data_path)
    if data_start_index.isdigit():
        data_start_index = int(data_start_index)
    else:
        data_start_index = None

    if data_end_index.isdigit():
        data_end_index = int(data_end_index)
    else:
        data_end_index = None
    names = names[data_start_index:data_end_index]
    data = data[data_start_index:data_end_index]

    # grid_path = os.path.join(_data_root, grid_token)
    # if os.path.isfile(grid_path):
    #     grid_params = json.load(open(grid_path, encoding='utf8'))
    # else:
    grid_params = dict(
        data_start_index=data_start_index,  # 起始点
        data_end_index=data_end_index,  # 结束点
        touruzijin=float(touruzijin),  # 投入总资金
        # jizhunjia=float(jizhunjia),  # 基准价，0则以初始价为基准价
        # dancifene=int(dancifene),  # 单次交易股数
        # jiancangfene=int(jiancangfene),  # 建仓股数
        # wanggeshangjie=float(wanggeshangjie),  # 网格上界
        # wanggexiajie=float(wanggexiajie),  # 网格下界
        # wanggeleixing=int(wanggeleixing),  # 网格类型，1为差价，2为百分比
        # mairuyuzhi=float(mairuyuzhi),  # 买入阈值
        # maichuyuzhi=float(maichuyuzhi),  # 卖出阈值
        shouxufeilv=float(shouxufeilv),  # 手续费率
    )

    params_json = json.dumps(grid_params, ensure_ascii=False, indent=4)
    token = hashlib.md5(params_json.encode('utf8')).hexdigest()
    fname = f'{name}.json'
    cnt = 1
    while os.path.exists(os.path.join(_data_root, fname)):
        fname = f'{name}.{token[:cnt]}.json'
        cnt += 1
    grid_token = fname

    result = search_params(shuju=data, param=grid_params, n_search=int(n_search), topn=int(topn))
    # result = grid_func(shuju=data,  # 数据
    #                    **grid_params)
    # 投入资金：10000，回报总价值：17342.759（其中现金：8740.759，份额价值：8602.000），盈亏比例：73.43%
    # display(shoupan=shoupan, maichu_idx=maichu_idx, mairu_idx=mairu_idx)

    fname_path = os.path.join(_data_root, grid_token)
    with open(fname_path, 'wt') as f:
        f.write(params_json)

    data = dict(data_token=data_token, grid_token=grid_token, grid_params=grid_params, data=data, names=names)
    result_path = os.path.join(_data_root, f'data_token={data_token}.grid_token={grid_token}.searching.json')
    with open(result_path, 'wt', encoding='utf8') as fout:
        out = dict(result=result, data=data, data_token=data_token, grid_token=grid_token)
        json.dump(out, fout, ensure_ascii=False, indent=4)

    return dict(success=1, message='do searching success.',
                data=dict(result=result, grid_token=grid_token, data_token=data_token))


@app.get('/download_result')
def download_result(data_token='data_token.csv', grid_token='grid_token.json'):
    """
    下载交易细节结果及其数据，下载为json文件。
    ```
    out = dict(result=result, data=data, data_token=data_token, grid_token=grid_token)

    data = dict(data_token=data_token, grid_token=grid_token, grid_params=grid_params, data=data, names=names)

    result = dict(report_conclusion=dict(report_content=report_content, report_data=report_data),
                  trading_detail=dict(maichu_jiazhi=maichu_jiazhi, maichu_idx=maichu_idx,
                                      mairu_jiazhi=mairu_jiazhi, mairu_idx=mairu_idx,
                                      jiazhi_lst=jiazhi_lst))
    index time value b/s jiazhi_before jiazhi
    ```
    ```
    :param data_token:
    :param grid_token:
    :return:
    ```
    """
    fname = f'data_token={data_token}.grid_token={grid_token}.json'
    result_path = os.path.join(_data_root, fname)
    data = json.load(open(result_path, encoding='utf8'))
    details = data['result']['trading_detail']
    lst = []
    for idx, (name, value, jiazhi_before) in enumerate(
            zip(data['data']['names'], data['data']['data'], details['jiazhi_lst'])):
        if details['maichu_idx'] and idx == details['maichu_idx'][0]:
            # 卖出
            idx = details['maichu_idx'].pop(0)
            money = details['maichu_jiazhi'].pop(0)
            dt = dict(time=name, value=value, action='sell', money=money,
                      money_before=jiazhi_before, number=idx, action_flag=-1)
        elif details['mairu_idx'] and idx == details['mairu_idx'][0]:
            # 买入
            idx = details['mairu_idx'].pop(0)
            money = details['mairu_jiazhi'].pop(0)
            dt = dict(time=name, value=value, action='buy', money=money,
                      money_before=jiazhi_before, number=idx, action_flag=1)
        elif data['result']['report_conclusion']['report_data']['kaishijiaoyidian'] == idx:
            dt = dict(time=name, value=value, action='init', money=jiazhi_before,
                      money_before=jiazhi_before, number=idx, action_flag=1)
        else:
            dt = dict(time=name, value=value, action='keep', money=jiazhi_before,
                      money_before=jiazhi_before, number=idx, action_flag=0)
        lst.append(dt)
    excel_path = result_path + '.xlsx'
    pandas.DataFrame(lst).to_excel(excel_path)
    return FileResponse(excel_path, filename=fname + '.xlsx')


@app.get('/get_result')
def get_result(data_token='data_token.csv', grid_token='grid_token.json'):
    """
    获取交易细节及其数据，返回字典。
    ```
    :param data_token:
    :param grid_token:
    :return:
    ```
    """
    fname = f'data_token={data_token}.grid_token={grid_token}.json'
    result_path = os.path.join(_data_root, fname)
    with open(result_path, 'rt', encoding='utf8') as fin:
        out = json.load(fin)
        return out


def grid_func(shuju, touruzijin, jizhunjia, dancifene, jiancangfene,
              wanggeshangjie, wanggexiajie, wanggeleixing, mairuyuzhi, maichuyuzhi,
              shouxufeilv=0.0001, **kwargs):
    """
    网格交易回测
    ```
    :param shuju: 股票所有数据
    :param touruzijin: 投入资金
    :param jizhunjia: 基准价格，设置为0则为回测的第一个数据
    :param dancifene: 买入卖出单次交易股数，等股交易
    :param jiancangfene: 起始建仓股数
    :param wanggeshangjie: 网格上界
    :param wanggexiajie: 网格下界
    :param wanggeleixing: 网格类型，1为差价，2为百分比
    :param mairuyuzhi: 买入阈值，差价或百分比
    :param maichuyuzhi: 卖出阈值，差价或百分比
    :param shouxufeilv: 手续费率，默认0.01%，0.1元起
    :return:
    ```
    """
    values = shuju
    # 暂用起始价来做基准价
    if not jizhunjia:
        jizhunjia = values[0]

    # 基准价是否限制在网格上下界
    # if wanggexiajie <= values[0] <= wanggeshangjie:
    #     jizhunjia = values[0]
    # elif values[0] < wanggexiajie:
    #     jizhunjia = wanggexiajie
    # else:
    #     jizhunjia = wanggeshangjie

    # 是否建立起始仓位
    xianjin = touruzijin - jiancangfene * jizhunjia
    # 总共份额数
    totalfene = jiancangfene
    jizhunjingzhi = jizhunjia

    maichu_jiazhi, mairu_jiazhi = [], []
    maichu_idx, mairu_idx = [], []

    # 找股价达到基准价的时刻
    kaishi_idx = 0
    for i, (v1, v2) in enumerate(zip(values, values[1:])):
        if (jizhunjia - v1) * (jizhunjia - v2) <= 0:
            # 基准价在前后两个股价之间，则开始交易
            kaishi_idx = i
            jizhunjingzhi = v1
            break

    jiazhi_lst = []
    for i in range(0, len(values), 1):
        # 记录每一天的总价值
        if i < kaishi_idx:
            jiazhi_lst.append(touruzijin)
            # 股价达到基准价时候开始交易
            continue

        # 获取当时股价净值
        dangshijingzhi = values[i]
        jiazhi_lst.append(xianjin + totalfene * dangshijingzhi)

        # 涨破上界
        if dangshijingzhi > wanggeshangjie:
            continue

        # 跌破下界
        if dangshijingzhi < wanggexiajie:
            continue

        if wanggeleixing == 1:
            # 差价网格
            maichujia = jizhunjingzhi + maichuyuzhi
        else:
            # 比例网格
            maichujia = jizhunjingzhi * (1 + maichuyuzhi / 100)
        # 卖出基金
        if dangshijingzhi >= maichujia and totalfene >= dancifene:
            totalfene -= dancifene
            shouxufei = max(0.1, shouxufeilv * dangshijingzhi * dancifene)
            xianjin += (dangshijingzhi * dancifene - shouxufei)
            maichu_idx.append(i)
            maichu_jiazhi.append(xianjin + totalfene * dangshijingzhi)

            # 用成交价更新基准价
            jizhunjingzhi = dangshijingzhi

            # 可能出现当时净值高于多个网格的情况
            gengxin_idx = 0
            continue

        if wanggeleixing == 1:
            # 差价网格
            mairujia = jizhunjingzhi - mairuyuzhi
        else:
            # 比例网格
            mairujia = jizhunjingzhi * (1 - mairuyuzhi / 100)
        # 买入基金
        if dangshijingzhi <= mairujia and xianjin >= (1 + shouxufeilv) * dancifene * dangshijingzhi:
            totalfene += dancifene
            shouxufei = max(0.1, shouxufeilv * dangshijingzhi * dancifene)
            xianjin -= (dangshijingzhi * dancifene + shouxufei)
            mairu_idx.append(i)
            mairu_jiazhi.append(xianjin + totalfene * dangshijingzhi)

            # 用成交价更新基准价
            jizhunjingzhi = dangshijingzhi

            # 可能出现当时净值低于多个网格的情况
            gengxin_idx = 0
            continue
    zongjiazhi = (xianjin + totalfene * values[-1])
    report_content = ("投入资金：{}，回报总价值：{:.3f}（其中现金：{:.3f}，份额价值：{:.3f}），盈亏比例：{:.2f}%".format(
        touruzijin, zongjiazhi, xianjin, totalfene * values[-1], (zongjiazhi / touruzijin) * 100))

    xianjin = round(xianjin, 4)
    report_data = dict(
        touruzijin=touruzijin,
        zongjiaoyicishu=len(mairu_idx) + len(maichu_idx),
        zongjiageshu=len(values) - kaishi_idx,
        kaishijiaoyidian=kaishi_idx,
        kanpanjia=jizhunjia,
        xianjin=xianjin,
        totalfene=totalfene,
        shoupanjia=values[-1],
        fenejiazhi=totalfene * values[-1],
        zongjiazhi=round(xianjin + totalfene * values[-1], 4),
        yinkuibili=round(zongjiazhi / touruzijin, 4)
    )
    # from pprint import pprint

    # pprint(report_content)
    result = dict(report_conclusion=dict(report_content=report_content, report_data=report_data),
                  trading_detail=dict(maichu_jiazhi=maichu_jiazhi, maichu_idx=maichu_idx,
                                      mairu_jiazhi=mairu_jiazhi, mairu_idx=mairu_idx,
                                      jiazhi_lst=jiazhi_lst))
    return result


def search_params(shuju, param, n_search=11, topn=10):
    """搜索参数"""
    _shangjie_p = np.linspace(np.min(shuju), np.max(shuju), n_search)
    _xiajie_p = np.linspace(np.min(shuju), np.max(shuju), n_search)
    touruzijin = param['touruzijin']
    # pprint.pprint(dict(_shangjie_p=_shangjie_p, _xiajie_p=_xiajie_p, touruzijin=touruzijin))
    params = []
    for _xiajie in _xiajie_p:
        for _shangjie in _shangjie_p:
            if _shangjie - _xiajie <= 0.001:
                continue
            _wangge_p = (np.linspace(0, _shangjie - _xiajie, n_search) ** 2) / (_shangjie - _xiajie)
            _jizhunjia_p = np.linspace(_xiajie, _shangjie, 5)
            for _wangge in _wangge_p:
                if _shangjie - _xiajie < _wangge:
                    continue
                if _wangge < 0.001:
                    continue
                for _jizhunjia in _jizhunjia_p:
                    _jc_max = int(touruzijin // _jizhunjia // 100 * 100)
                    _jc_min = 100
                    _jiancangfene_p = np.arange(_jc_min, _jc_max + 1, 100)
                    _jiancangfene_p = _jiancangfene_p[::len(_jiancangfene_p) // 5]
                    for _jiancangfene in _jiancangfene_p:
                        _dancifene_p = np.arange(100, _jiancangfene + 1, 100)
                        for _dancifene in _dancifene_p:
                            if _jiancangfene % _dancifene != 0:
                                continue
                            p = dict(wanggeshangjie=round(_shangjie, 4), wanggexiajie=round(_xiajie, 4),
                                     wanggeleixing=1,
                                     mairuyuzhi=round(_wangge, 4), maichuyuzhi=round(_wangge, 4),
                                     jizhunjia=round(_jizhunjia, 4), jiancangfene=int(_jiancangfene),
                                     dancifene=int(_dancifene))
                            pa = {**param, **p}
                            params.append(pa)

    res_lst = []
    for param in tqdm.tqdm(params):
        res = grid_func(shuju=shuju, **param)
        result = res['report_conclusion']['report_data']
        res_lst.append(dict(param=param, result=result))
    outs = list(sorted(res_lst, key=lambda x: x['result']['yinkuibili'], reverse=True))
    yinkuibili_lst = [x['result']['yinkuibili'] for x in outs]

    pprint.pprint(outs[:3])
    return dict(result_best=outs[:1], result_topn=outs[:topn], result_examples=outs[::len(outs) // topn],
                yinkuibili_examples=yinkuibili_lst[::len(yinkuibili_lst) // topn])


def display(shoupan, maichu_idx, mairu_idx):
    """
    可视化买卖点。
    :param shoupan: 收盘价列表。
    :param maichu_idx: 卖出id列表。
    :param mairu_idx: 买入id列表。
    :return:
    """
    x = [_id for _id in range(len(shoupan))]
    maichu = [shoupan[i] for i in maichu_idx]
    mairu = [shoupan[i] for i in mairu_idx]
    plt.plot(x, shoupan, color='blue', linewidth=0.5, linestyle='-')
    plt.xlabel("x - 时间点")
    plt.ylabel("y - 股票价格")
    plt.scatter(mairu_idx, mairu, s=10, color='green', label='买入交易点', marker='o', linewidths=1)
    plt.scatter(maichu_idx, maichu, s=10, color='red', label='卖出交易点', marker='x', linewidths=1)
    plt.legend()
    plt.show()


def parse_excel(data_path):
    """
    解析导出的Excel股票数据。
    :param data_path:
    :return:
    """
    df = pd.read_csv(data_path, encoding='gb2312')
    shijian, values = [], []
    for _id in range(1, len(df) - 1):
        for line in list(df.loc[_id]):
            process_line = line.split('\t')
            shijian.append(f'{process_line[0]}-kaipan')
            values.append(float(process_line[1]))

            shijian.append(f'{process_line[0]}-zuigao')
            values.append(float(process_line[2]))

            shijian.append(f'{process_line[0]}-zuidi')

            values.append(float(process_line[3]))

            shijian.append(f'{process_line[0]}-shoupan')
            values.append(float(process_line[4]))
    return shijian, values


def run_example():
    """
    跑整个流程的样例。
    :return:
    """
    data_path = 'static/000665.xls'
    shijian, shuju = parse_excel(data_path)
    result = grid_func(
        shuju=shuju,  # 数据
        touruzijin=10000,  # 投入总资金
        jizhunjia=0,  # 基准价，0则以初始价为基准价
        dancifene=100,  # 单次交易股数
        jiancangfene=1000,  # 建仓股数
        wanggeshangjie=12,  # 网格上界
        wanggexiajie=6,  # 网格下界
        wanggeleixing=1,  # 网格类型，1为差价，2为百分比
        mairuyuzhi=0.5,  # 买入阈值
        maichuyuzhi=0.5,  # 卖出阈值
        shouxufeilv=0.0001,  # 手续费率
    )
    # 投入资金：10000，回报总价值：17342.759（其中现金：8740.759，份额价值：8602.000），盈亏比例：73.43%
    display(shoupan=shuju,
            maichu_idx=result['trading_detail']['maichu_idx'],
            mairu_idx=result['trading_detail']['mairu_idx'])


if __name__ == "__main__":
    run_example()

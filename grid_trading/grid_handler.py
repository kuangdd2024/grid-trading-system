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
import datetime
import functools
import json
import os
import pathlib
import pprint
import re
import traceback
import hashlib
import base64

import pandas
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl

import numpy as np
import tqdm

from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import FileResponse
from .fastapi_fixer import app

import efinance as ef

# 设置字体类型
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']

_data_root = 'mnt/wp'


@app.post('/do_loading')
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
def do_trading(
        grid_token_evaluating: str = Query('',
                                           description='do_evaluating结果的token，如果设置grid_token_evaluating，则下面的参数不生效'),
        evaluating_result_choice: str = Query('topn 0', description='do_evaluating结果选择，grid_token_evaluating设置后才生效。\n'
                                                                    '参数设置规则：topn [int]：result_topn的第[int]个参数，'
                                                                    'examples [int]：result_examples的第[int]个参数'),
        grid_token_trading: str = Query('',
                                        description='do_trading结果的token，如果设置grid_token_trading，则下面的参数不生效'),
        data_token: str = Query('000665-101',
                                description="股票代码-时间间隔，例如：000665-5，或股票数据的token。"
                                            "股票代码例如：000665；"
                                            "行情之间的时间间隔，默认为 ``5`` ，可选示例如下："
                                            " - ``1`` : 分钟 - ``5`` : 5 分钟 - ``15`` : 15 分钟 - ``30`` : 30 分钟"
                                            " - ``60`` : 60 分钟 - ``101`` : 日 - ``102`` : 周 - ``103`` : 月"),
        data_start_index: str = Query('2022/09/01',
                                      description='数据起始位置或起始时间，时间格式例如：2022/07/30-09:30:00，2022/07/30'),
        data_end_index: str = Query('2023/07/31',
                                    description='数据结束位置或结束时间，时间格式例如：2022/07/30-15:00:00，2022/07/30'),
        name: str = Query('trading', description='网格参数的token生成基于name'),
        touruzijin: float = Query(10000.0, description='投入资金'),
        jizhunjia: float = Query(6.0, description='基准价格，设置为0则为回测的第一个数据'),
        dancifene: int = Query(100, description='买入卖出单次交易股数，等股交易'),
        jiancangfene: int = Query(1000, description='起始建仓股数'),
        wanggeshangjie: float = Query(7.0, description='网格上界'),
        wanggexiajie: float = Query(5.0, description='网格下界'),
        wanggeleixing: int = Query(1, description='网格类型，1为差价，2为百分比'),
        mairuyuzhi: float = Query(0.05, description='网格大小，买入阈值，差价或百分比'),
        maichuyuzhi: float = Query(0.05, description='网格大小，卖出阈值，差价或百分比'),
        shouxufeilv: float = Query(0.0001, description='手续费率，默认0.01%，0.1元起')
):
    """
    网格交易回测。
    """
    if grid_token_evaluating:
        # data = dict(data_token=data_token, grid_token=grid_token, grid_params=grid_params, data=data, names=names)
        result_path = os.path.join(_data_root, f'grid_token={grid_token_evaluating}.evaluating.json')
        sw, idx = evaluating_result_choice.split()
        with open(result_path, 'rt', encoding='utf8') as fin:
            # out = dict(result=result, data=data, data_token=data_token, grid_token=grid_token)
            grid_params = json.load(fin)['result'][f'result_{sw}'][int(idx)]['param']
    elif grid_token_trading:
        # data = dict(data_token=data_token, grid_token=grid_token, grid_params=grid_params, data=data, names=names)
        result_path = os.path.join(_data_root, f'grid_token={grid_token_trading}.trading.json')
        with open(result_path, 'rt', encoding='utf8') as fin:
            # out = dict(result=result, data=data, data_token=data_token, grid_token=grid_token)
            grid_params = json.load(fin)['data']['grid_params']
    else:
        grid_params = dict(
            data_token=data_token,  # 数据代号
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

    data_token = grid_params['data_token']
    names, data = parse_data(data_token, data_start_index, data_end_index)
    params_json = json.dumps(grid_params, ensure_ascii=False, indent=4)
    token = hashlib.md5(params_json.encode('utf8')).hexdigest()
    fname = f'{data_token}.{name}.json'
    cnt = 1
    while os.path.exists(os.path.join(_data_root, fname)):
        fname = f'{data_token}.{name}.{token[:cnt]}.json'
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
    result_path = os.path.join(_data_root, f'grid_token={grid_token}.trading.json')
    with open(result_path, 'wt', encoding='utf8') as fout:
        out = dict(result=result, data=data, data_token=data_token, grid_token=grid_token)
        json.dump(out, fout, ensure_ascii=False, indent=4)

    return dict(success=1, message='do trading success.',
                data=dict(grid_token=grid_token, data_token=data_token, result=result['report_conclusion']))


@app.get('/do_searching')
def do_searching(
        data_token: str = Query('000665-101', description='股票数据的token'),
        data_start_index: str = Query('2022/01/01',
                                      description='数据起始位置或起始时间，时间格式例如：2022/07/30-09:30:00，2022/07/30'),
        data_end_index: str = Query('2023/06/30',
                                    description='数据结束位置或结束时间，时间格式例如：2022/07/30-15:00:00，2022/07/30'),
        name: str = Query('searching', description='网格参数的token生成基于name'),
        touruzijin: float = Query(100000, description='投入资金'),
        # jizhunjia: int = Query(0, description='基准价格，设置为0则为回测的第一个数据'),
        # dancifene: int = Query(100, description='买入卖出单次交易股数，等股交易'),
        # jiancangfene: int = Query(1000, description='起始建仓股数'),
        # wanggeshangjie: int = Query(12, description='网格上界'),
        # wanggexiajie: int = Query(6, description='网格下界'),
        # wanggeleixing: int = Query(1, description='网格类型，1为差价，2为百分比'),
        # mairuyuzhi: float = Query(0.5, description='网格大小，买入阈值，差价或百分比'),
        # maichuyuzhi: float = Query(0.5, description='网格大小，卖出阈值，差价或百分比'),
        shouxufeilv: float = Query(0.0001, description='手续费率，默认0.01%，0.1元起'),
        n_search: int = Query(4, description='搜索密度，数量越大越精细，但耗时越久，默认4'),
        topn: int = Query(10, description='返回最佳收益的结果数，默认10')
):
    """
    网格交易参数搜索。
    """
    # jizhunjia = 0, dancifene = 100,
    # jiancangfene = 1000,
    # wanggeshangjie = 12, wanggexiajie = 6, wanggeleixing = 1, mairuyuzhi = 0.5, maichuyuzhi = 0.5,
    names, data = parse_data(data_token, data_start_index, data_end_index)

    # grid_path = os.path.join(_data_root, grid_token)
    # if os.path.isfile(grid_path):
    #     grid_params = json.load(open(grid_path, encoding='utf8'))
    # else:
    grid_params = dict(
        data_token=data_token,  # 数据代号
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
    fname = f'{data_token}.{name}.json'
    cnt = 1
    while os.path.exists(os.path.join(_data_root, fname)):
        fname = f'{data_token}.{name}.{token[:cnt]}.json'
        cnt += 1
    grid_token = fname

    result = search_params(param=params_json, n_search=int(n_search), topn=int(topn))
    # result = grid_func(shuju=data,  # 数据
    #                    **grid_params)
    # 投入资金：10000，回报总价值：17342.759（其中现金：8740.759，份额价值：8602.000），盈亏比例：73.43%
    # display(shoupan=shoupan, maichu_idx=maichu_idx, mairu_idx=mairu_idx)

    fname_path = os.path.join(_data_root, grid_token)
    with open(fname_path, 'wt') as f:
        f.write(params_json)

    data = dict(data_token=data_token, grid_token=grid_token, grid_params=grid_params, data=data, names=names)
    result_path = os.path.join(_data_root, f'grid_token={grid_token}.searching.json')
    with open(result_path, 'wt', encoding='utf8') as fout:
        out = dict(result=result, data=data, data_token=data_token, grid_token=grid_token)
        json.dump(out, fout, ensure_ascii=False, indent=4)

    return dict(success=1, message='do searching success.',
                data=dict(grid_token=grid_token, data_token=data_token, result=result))


@app.get('/do_evaluating')
def do_evaluating(
        data_token: str = Query('000665-101', description='股票数据的token'),
        data_start_index: str = Query('2022/01/01',
                                      description='训练数据起始位置或起始时间，时间格式例如：2022/07/30-09:30:00，2022/07/30'),
        data_end_index: str = Query('2022/12/31',
                                    description='训练数据结束位置或结束时间，时间格式例如：2022/07/30-15:00:00，2022/07/30'),
        data_eval_start_index: str = Query('2023/01/01',
                                           description='验证数据起始位置或起始时间，时间格式例如：2022/07/30-09:30:00，2022/07/30'),
        data_eval_end_index: str = Query('2023/06/30',
                                         description='验证数据结束位置或结束时间，时间格式例如：2022/07/30-15:00:00，2022/07/30'),
        name: str = Query('evaluating', description='网格参数的token生成基于name'),
        touruzijin: float = Query(100000, description='投入资金'),
        # jizhunjia: int = Query(0, description='基准价格，设置为0则为回测的第一个数据'),
        # dancifene: int = Query(100, description='买入卖出单次交易股数，等股交易'),
        # jiancangfene: int = Query(1000, description='起始建仓股数'),
        # wanggeshangjie: int = Query(12, description='网格上界'),
        # wanggexiajie: int = Query(6, description='网格下界'),
        # wanggeleixing: int = Query(1, description='网格类型，1为差价，2为百分比'),
        # mairuyuzhi: float = Query(0.5, description='网格大小，买入阈值，差价或百分比'),
        # maichuyuzhi: float = Query(0.5, description='网格大小，卖出阈值，差价或百分比'),
        shouxufeilv: float = Query(0.0001, description='手续费率，默认0.01%，0.1元起'),
        n_search: int = Query(4, description='搜索密度，数量越大越精细，但耗时越久，默认4'),
        topn: int = Query(10, description='返回最佳收益的结果数，默认10')

):
    """
    网格交易参数搜索和评估。
    """
    # jizhunjia = 0, dancifene = 100,
    # jiancangfene = 1000,
    # wanggeshangjie = 12, wanggexiajie = 6, wanggeleixing = 1, mairuyuzhi = 0.5, maichuyuzhi = 0.5,
    # names, data = parse_data(data_token, data_start_index, data_end_index)

    # grid_path = os.path.join(_data_root, grid_token)
    # if os.path.isfile(grid_path):
    #     grid_params = json.load(open(grid_path, encoding='utf8'))
    # else:
    grid_params = dict(
        data_token=data_token,  # 数据代号
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
    fname = f'{data_token}.{name}.json'
    cnt = 1
    while os.path.exists(os.path.join(_data_root, fname)):
        fname = f'{data_token}.{name}.{token[:cnt]}.json'
        cnt += 1
    grid_token = fname

    # 训练集搜索最佳参数
    result = search_params(param=params_json, n_search=int(n_search), topn=int(topn))

    # 评估验证集
    names, data = parse_data(data_token, data_eval_start_index, data_eval_end_index)

    for dt in result["result_topn"]:
        res = grid_func(shuju=data, **dt["param"])
        rep = res['report_conclusion']['report_data']
        dt["result_evaluating"] = rep

    for dt in result["result_examples"]:
        res = grid_func(shuju=data, **dt["param"])
        rep = res['report_conclusion']['report_data']
        dt["result_evaluating"] = rep
        # result = grid_func(shuju=data,  # 数据
    #                    **grid_params)
    # 投入资金：10000，回报总价值：17342.759（其中现金：8740.759，份额价值：8602.000），盈亏比例：73.43%
    # display(shoupan=shoupan, maichu_idx=maichu_idx, mairu_idx=mairu_idx)

    fname_path = os.path.join(_data_root, grid_token)
    with open(fname_path, 'wt') as f:
        f.write(params_json)

    data = dict(data_token=data_token, grid_token=grid_token, grid_params=grid_params, data=data, names=names)
    result_path = os.path.join(_data_root, f'grid_token={grid_token}.evaluating.json')
    with open(result_path, 'wt', encoding='utf8') as fout:
        out = dict(result=result, data=data, data_token=data_token, grid_token=grid_token)
        json.dump(out, fout, ensure_ascii=False, indent=4)

    return dict(success=1, message='do evaluating success.',
                data=dict(grid_token=grid_token, data_token=data_token, result=result))


@app.get('/download_trading_detail')
def download_trading_detail(
        data_token='data_token.csv',
        grid_token='000665-5.trading.json',
        output_format: str = Query('.xlsx',
                                   description='下载文件格式，支持格式：.xlsx、.json、data，'
                                               '如果是data则返回json的response')
):
    """
    下载交易细节结果及其数据，下载为Excel文件。
    """
    # out = dict(result=result, data=data, data_token=data_token, grid_token=grid_token)
    # data = dict(data_token=data_token, grid_token=grid_token, grid_params=grid_params, data=data, names=names)
    # result = dict(report_conclusion=dict(report_content=report_content, report_data=report_data),
    #               trading_detail=dict(maichu_jiazhi=maichu_jiazhi, maichu_idx=maichu_idx,
    #                                   mairu_jiazhi=mairu_jiazhi, mairu_idx=mairu_idx,
    #                                   jiazhi_lst=jiazhi_lst))
    # index time value b/s jiazhi_before jiazhi

    fname = f'grid_token={grid_token}.trading.json'
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
    if output_format in ['.xlsx', '.xls', '.csv', '.tsv']:
        excel_path = result_path + output_format
        pandas.DataFrame(lst).to_excel(excel_path)
        return FileResponse(excel_path, filename=fname + output_format)
    elif output_format in ['.json']:
        excel_path = result_path + output_format
        with open(result_path + output_format, 'wt', encoding='utf8') as fout:
            json.dump(lst, ensure_ascii=False, indent=4)
        return FileResponse(excel_path, filename=fname + output_format)
    else:
        return dict(success=1, message='download trading details success', data=lst)


@app.get('/get_result')
def get_result(
        grid_token: str = Query('000665-5.trading.json', description='参数token'),
        switch: str = Query('trading', description='哪一个功能的结果，选择范围：trading、searching、evaluating'),
        output_format: str = Query('data',
                                   description='下载文件格式，支持格式：.json、data，'
                                               '如果是data则返回json的response')
):
    """
    获取交易细节。
    """
    fname = f'grid_token={grid_token}.{switch}.json'
    result_path = os.path.join(_data_root, fname)
    if output_format == '.json':
        return FileResponse(result_path, filename=fname)
    else:
        with open(result_path, 'rt', encoding='utf8') as fin:
            out = json.load(fin)
            return dict(success=1, message='success', data=out)


@app.get('/get_data')
def get_data(
        data_token: str = Query('000665-5', description='股票数据token，仅当switch=loading时生效'),
        data_start_index: str = Query('2022/01/01',
                                      description='数据起始位置或起始时间，时间格式例如：2022/07/30'),
        data_end_index: str = Query('2023/06/30',
                                    description='数据结束位置或结束时间，时间格式例如：2022/07/30'),
        output_format: str = Query('data',
                                   description='下载文件格式，支持格式：.xlsx、.json、data，'
                                               '如果是data则返回json的response')
):
    """获取股票数据。"""
    names, data = parse_data(data_token, data_start_index=data_start_index, data_end_index=data_end_index)
    s = data_start_index.replace('/', '-')
    e = data_end_index.replace('/', '-')
    out = list(zip(names, data))
    if output_format in ['.xlsx', '.xls', '.csv', '.tsv']:
        fname = f'data_token={data_token}.{s}_{e}.{output_format}'
        outpath = os.path.join(_data_root, fname)
        pandas.DataFrame(out).to_excel(outpath)
        return FileResponse(outpath, filename=fname)
    elif output_format == '.json':
        fname = f'data_token={data_token}.{s}_{e}.{output_format}'
        outpath = os.path.join(_data_root, fname)
        json.dump(out, outpath, ensure_ascii=False, indent=4)
        return FileResponse(outpath, filename=fname)
    else:
        return dict(success=1, message='success', data=out)


@app.get('/get_history')
def get_history(
        pattern: str = Query('*', description='查看逻辑。'),
        token: str = Query('', description='history的token验证。')
):
    """
    获取历史记录。
    """
    if token == 'grid52128':
        result = list(sorted(pathlib.Path(_data_root).glob(pattern)))
        return dict(success=1, message='get history success.', data=result)
    else:
        return dict(success=0, message='获取历史数据失败，原因：token有误，请核对！')


# @functools.lru_cache(maxsize=128)
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
    # param = json.loads(param)
    # names, values = parse_data(param["data_token"], param["data_start_index"], param["data_end_index"])
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


@functools.lru_cache(maxsize=128)
def search_params(param, n_search=5, topn=10):
    """搜索参数"""
    param = json.loads(param)
    names, shuju = parse_data(param["data_token"], param["data_start_index"], param["data_end_index"])

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
            _jizhunjia_p = np.linspace(_xiajie, _shangjie, n_search)
            for _wangge in _wangge_p:
                if _shangjie - _xiajie < _wangge:
                    continue
                if _wangge < 0.001:
                    continue
                for _jizhunjia in _jizhunjia_p:
                    _jc_max = int(touruzijin // _jizhunjia // 100 * 100)
                    _jc_min = 100
                    _jiancangfene_p = np.arange(_jc_min, _jc_max + 1, 100)
                    _jiancangfene_p = _jiancangfene_p[::len(_jiancangfene_p) // n_search]
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

    # pprint.pprint(outs[:3])
    return dict(result_best=outs[:1], result_topn=outs[:topn], result_examples=outs[::len(outs) // topn],
                yinkuibili_examples=yinkuibili_lst[::len(yinkuibili_lst) // topn])


def parse_data_index(names, data_start_index, data_end_index):
    """
    解析起止位置。
    :param names:
    :param data_start_index:
    :param data_end_index:
    :return:
    """
    # 用时间选择区域
    times = [datetime.datetime.strptime(w, '%Y/%m/%d-%H:%M:%S') for w in names]

    if re.search(r'\d+/\d+/\d+', data_start_index):
        if not re.search(r'\d+:\d+:\d+', data_start_index):
            data_start_index = f'{data_start_index}-00:00:00'
            data_end_index = f'{data_end_index}-23:59:59'

        time_start = datetime.datetime.strptime(data_start_index, '%Y/%m/%d-%H:%M:%S')
        time_end = datetime.datetime.strptime(data_end_index, '%Y/%m/%d-%H:%M:%S')
        data_start_index = None
        data_end_index = None
        for num, time_now in enumerate(times):
            if data_start_index is None and time_now >= time_start:
                data_start_index = str(num)

            if data_end_index is None and time_now >= time_end:
                data_end_index = str(num)

    if data_start_index.lstrip('-').isdigit():
        data_start_index = int(data_start_index)
    else:
        data_start_index = None

    if data_end_index.lstrip('-').isdigit():
        data_end_index = int(data_end_index)
    else:
        data_end_index = None
    return times, data_start_index, data_end_index


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


@functools.lru_cache(maxsize=128)
def parse_excel(data_path):
    """
    解析导出的Excel股票数据。
    :param data_path:
    :return:
    """
    flag = 0
    t_min, v_shift = '', 0
    shijian, values = [], []
    with open(data_path, encoding='gbk') as fin:
        for line in fin:
            parts = re.split(r'[,\s]+', line.strip())
            if len(parts) >= 5:
                if flag == 0:
                    if parts[0] == '时间':
                        if len(parts[0].split('-')) == 2:
                            flag = 1
                        else:
                            flag = 2
                    elif parts[0] == '日期':
                        if parts[1] == '时间':
                            flag = 3
                        else:
                            flag = 4
                    else:
                        flag = -1
                    continue

                if flag == 1:
                    # 2023/02/28-10:31
                    t_min = parts[0]
                    v_shift = 1
                elif flag == 2:
                    # 2023/02/28
                    t_min = f'{parts[0]}-10:00'
                    v_shift = 1
                elif flag == 3:
                    # 2023-02-28 1031
                    t_min = f'{parts[0].replace("-", "/")}-{parts[1][:-2]}:{parts[1][-2:]}'
                    v_shift = 2
                elif flag == 4:
                    # 2023-02-28
                    t_min = f'{parts[0].replace("-", "/")}-10:00'
                    v_shift = 1
                else:
                    assert 1 <= flag <= 4

                if 1 <= flag <= 4:
                    print(parts, t_min)
                    t = t_min
                    shijian.append(f'{t}:10')
                    values.append(float(parts[0 + v_shift]))

                    shijian.append(f'{t}:20')
                    values.append(float(parts[1 + v_shift]))

                    shijian.append(f'{t}:30')
                    values.append(float(parts[2 + v_shift]))

                    shijian.append(f'{t}:40')
                    values.append(float(parts[3 + v_shift]))
    return shijian, values


@functools.lru_cache(maxsize=128)
def request_data(stock_code='000665', start_date='20230701', end_date='20230730', frequency=5):
    data = ef.stock.get_quote_history(stock_code, klt=frequency, beg=start_date, end=end_date)

    shijian, values = [], []
    for process_line in data.values:
        t = process_line[2].strip().replace('-', '/').replace(' ', '-')
        if not re.search(r'\d+:\d+', t):  # 日线
            t = f'{t}-10:00'
        shijian.append(f'{t}:10')
        values.append(float(process_line[3]))

        shijian.append(f'{t}:20')
        values.append(float(process_line[6]))

        shijian.append(f'{t}:30')
        values.append(float(process_line[5]))

        shijian.append(f'{t}:40')
        values.append(float(process_line[4]))
    # 股票名称    股票代码          日期       开盘       收盘       最高       最低     成交量           成交额    振幅   涨跌幅    涨跌额    换手率
    # ['湖北广电' '000665' '2023-07-20' 5.55 5.5 5.57 5.5 104447 57728957.25 1.27 -0.54 -0.03 0.92]
    return shijian, values


@functools.lru_cache(maxsize=128)
def parse_data(data_token, data_start_index, data_end_index):
    if data_token.endswith('.csv'):
        data_path = os.path.join(_data_root, data_token)
        names, data = parse_excel(data_path)

        times, data_start_index, data_end_index = parse_data_index(names, data_start_index, data_end_index)

        names = names[data_start_index:data_end_index]
        data = data[data_start_index:data_end_index]
    else:
        stock_code, frequency = data_token.split('-')
        names, data = request_data(
            stock_code=str(stock_code),
            start_date=data_start_index.replace('/', ''),
            end_date=data_end_index.replace('/', ''),
            frequency=int(frequency))
    return names, data


def run_example():
    """
    跑整个流程的样例。
    :return:
    """
    # data_path = 'static/HT34-000665-5min.xls'
    data_path = 'static/SZ#000665#1day.csv'
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

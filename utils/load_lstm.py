from datetime import datetime
import numpy as np
import os


def load_cla_data(data_path, tra_date, val_date, tes_date, seq=2,
                  date_format='%Y-%m-%d'):
    fnames = [fname for fname in os.listdir(data_path) if
              os.path.isfile(os.path.join(data_path, fname))]
    print(len(fnames), ' tickers selected')

    data_EOD = []
    for index, fname in enumerate(fnames):
        # print(fname)
        single_EOD = np.genfromtxt(
            os.path.join(data_path, fname), dtype=float, delimiter=',',
            skip_header=False
        )
        # print('data shape:', single_EOD.shape)
        data_EOD.append(single_EOD)
    fea_dim = data_EOD[0].shape[1] - 2

    trading_dates = np.genfromtxt(
        os.path.join(data_path, '..', 'trading_dates.csv'), dtype=str,
        delimiter=',', skip_header=False
    )
    print(len(trading_dates), 'trading dates:')
    print(type(trading_dates[0]))

    # transform the trading dates into a dictionary with index, at the same
    # time, transform the indices into a dictionary with weekdays
    dates_index = {}
    index_dates = {}
    # indices_weekday = {}
    data_wd = np.zeros([len(trading_dates), 5], dtype=float)
    wd_encodings = np.identity(5, dtype=float)
    for index, date in enumerate(trading_dates):
        dates_index[date] = index
        index_dates[index] = date
        # indices_weekday[index] = datetime.strptime(date, date_format).weekday()
        data_wd[index] = wd_encodings[datetime.strptime(date, date_format).weekday()]

    tra_ind = dates_index[tra_date]
    val_ind = dates_index[val_date]
    tes_ind = dates_index[tes_date]
    print(tra_ind, val_ind, tes_ind)
    # count training, validation, and testing instances
    tra_num = 0
    val_num = 0
    tes_num = 0
    # training
    for date_ind in range(tra_ind, val_ind):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if abs(data_EOD[tic_ind][date_ind][-2]) > 1e-8:
                if data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                    tra_num += 1
    print(tra_num, ' training instances')

    # validation
    for date_ind in range(val_ind, tes_ind):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if abs(data_EOD[tic_ind][date_ind][-2]) > 1e-8:
                if data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                    val_num += 1
    print(val_num, ' validation instances')

    # testing
    for date_ind in range(tes_ind, len(trading_dates)):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if abs(data_EOD[tic_ind][date_ind][-2]) > 1e-8:
                if data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                    tes_num += 1
    print(tes_num, ' testing instances')


    # 记录日期和对应股票
    tra_date_log = []
    val_date_log = []
    test_date_log = []

    tra_stock_log = []
    val_stock_log = []
    test_stock_log = []


    # generate training, validation, and testing instances
    # training
    tra_pv = np.zeros([tra_num, seq, fea_dim], dtype=float)
    tra_wd = np.zeros([tra_num, seq, 5], dtype=float)
    tra_gt = np.zeros([tra_num, 1], dtype=float)
    ins_ind = 0
    for date_ind in range(tra_ind, val_ind):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if abs(data_EOD[tic_ind][date_ind][-2]) > 1e-8 and \
                    data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                tra_date_log.append(index_dates[date_ind])
                tra_pv[ins_ind] = data_EOD[tic_ind][date_ind - seq: date_ind, : -2]
                tra_wd[ins_ind] = data_wd[date_ind - seq: date_ind, :]
                tra_gt[ins_ind, 0] = (data_EOD[tic_ind][date_ind][-2] + 1) / 2
                tra_stock_log.append(fnames[tic_ind])
                ins_ind += 1

    # validation
    val_pv = np.zeros([val_num, seq, fea_dim], dtype=float)
    val_wd = np.zeros([val_num, seq, 5], dtype=float)
    val_gt = np.zeros([val_num, 1], dtype=float)
    ins_ind = 0
    for date_ind in range(val_ind, tes_ind):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if abs(data_EOD[tic_ind][date_ind][-2]) > 1e-8 and \
                            data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                val_date_log.append(index_dates[date_ind])
                val_pv[ins_ind] = data_EOD[tic_ind][date_ind - seq: date_ind, :-2]
                val_wd[ins_ind] = data_wd[date_ind - seq: date_ind, :]
                val_gt[ins_ind, 0] = (data_EOD[tic_ind][date_ind][-2] + 1) / 2
                val_stock_log.append(fnames[tic_ind])
                ins_ind += 1

    # testing
    tes_pv = np.zeros([tes_num, seq, fea_dim], dtype=float)
    tes_wd = np.zeros([tes_num, seq, 5], dtype=float)
    tes_gt = np.zeros([tes_num, 1], dtype=float)
    ins_ind = 0
    for date_ind in range(tes_ind, len(trading_dates)):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if abs(data_EOD[tic_ind][date_ind][-2]) > 1e-8 and \
                            data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                test_date_log.append(index_dates[date_ind])
                tes_pv[ins_ind] = data_EOD[tic_ind][date_ind - seq: date_ind, :-2]
                # # for the momentum indicator
                # tes_pv[ins_ind, -1, -1] = data_EOD[tic_ind][date_ind - 1, -1] - data_EOD[tic_ind][date_ind - 11, -1]
                tes_wd[ins_ind] = data_wd[date_ind - seq: date_ind, :]
                tes_gt[ins_ind, 0] = (data_EOD[tic_ind][date_ind][-2] + 1) / 2
                test_stock_log.append(fnames[tic_ind])
                ins_ind += 1

    # import json
    #
    # txt_name = data_path[-3:]
    # with open(txt_name + "test_date_log.txt", 'w', encoding='utf-8') as f1:
    #     f1.write(json.dumps(test_date_log))
    #
    # with open(txt_name + "test_stock_log.txt", "w", encoding='utf-8') as f2:
    #     f2.write(json.dumps(test_stock_log))

    return tra_pv, tra_wd, tra_gt, val_pv, val_wd, val_gt, tes_pv, tes_wd, tes_gt, tra_date_log, val_date_log, test_date_log, tra_stock_log, val_stock_log, test_stock_log

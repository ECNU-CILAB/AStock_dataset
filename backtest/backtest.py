import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import math
import matplotlib
#指定默认字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family']='sans-serif' #解决负号'-'显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False


class backtest:
    def __init__(self, res_path, source_path, policy_type):
        self.res_path = res_path  # 结果文件位置
        self.hold_num = 20  # 持仓数
        self.adj_turn = 0.001  # 千分之一手续费
        self.st_money = 1000000  # 初始资金
        self.all_money = 1000000  # 现金流
        self.count_days = 0  # 计数当前是第几个交易日
        self.source_path = source_path  # 历史行情数据文件
        self.hold = pd.DataFrame(columns=['kdcode', 'hold'])
        self.policy_type = policy_type

    # 载入历史行情数据
    def load_history(self):
        df = pd.read_csv(self.source_path)
        df = df[['kdcode', 'dt', 'close']]
        return df

    # 查找一个元素在数组里面的位置，如果不存在，则返回-1
    def indexOf(e, a):
        for i in range(0, len(a)):
            if e == a[i]:
                return i
        return -1

    # 获取需要买入的股票代码
    def get_tobuy(self, best_stock, now_stock, max_count):
        count = 0
        toBuy = []
        for i in range(0, len(best_stock)):
            flag = 0
            for j in range(0, len(now_stock)):
                if best_stock[i] == now_stock[j]:
                    flag = 1
                    break;
            if flag == 0:
                toBuy.append(best_stock[i])
                count += 1
                if count >= max_count:
                    return toBuy
        return toBuy

    # 获取需要卖出的股票代码
    def get_tosell(self, now_price, now_stock, max_count):
        df = now_price[now_price['kdcode'].isin(now_stock)]
        #         print(df)
        df = df.sort_values(by=['close'], ascending=True)
        df = df.head(max_count)
        toSell = df['kdcode'].values
        return toSell

    # 模拟交易
    def trading(self, files, now_price):
        now_stock = self.hold['kdcode'].values
        best_stock = self.get_rank(self.res_path + '/' + files)
        best_stock = best_stock['kdcode'].values
        toBuy = best_stock
        toSell = now_stock
        # 现有策略，所有股票等金额
        if self.policy_type == 't3':
            max_count = 10
        if self.policy_type == 't5':
            max_count = 5
        if self.policy_type == 't10':
            max_count = 2
        if self.count_days == 0:
            max_count = 20
        toBuy = self.get_tobuy(best_stock, now_stock, max_count)
        max_count = len(toBuy)
        toSell = self.get_tosell(now_price, now_stock, max_count)
        self.sell(toSell, now_price)
        self.buy(toBuy, now_price)
        return self

    # 对于不在持仓列表的股票  卖出
    def sell(self, toSell, now_price):
        hold = self.hold  # 持仓份额
        for i in range(len(toSell)):
            if now_price[now_price['kdcode'] == toSell[i]].empty:  # 部分股票当日停牌
                code_price = 0.0
            else:
                code_price = 1.0 * now_price[now_price['kdcode'] == toSell[i]]['close'].values
                code_count = 1.0 * hold[hold['kdcode'] == toSell[i]]['hold'].values
                hold.drop(hold.index[hold['kdcode'] == toSell[i]], inplace=True)
            self.all_money += code_price[0] * code_count[0] * (1 - self.adj_turn)
        self.hold = hold
        return self

    # 对于需要持仓的股票  买入
    def buy(self, toBuy, now_price):
        sum_price = self.all_money / len(toBuy)  # 均分到每只股票的金额
        for i in range(len(toBuy)):
            price = 1.0 * now_price[now_price['kdcode'] == toBuy[i]]['close'].values
            hold = sum_price / (price * (1 + self.adj_turn))
            self.hold = self.hold.append({'kdcode': toBuy[i], 'hold': hold}, ignore_index=True)
        self.all_money = 0.0
        return self

    #     # 对于不在持仓列表的股票  卖出
    #     def sell_2(self, toSell, now_price):
    #         for i in range(len(toSell)):
    #             if now_price[now_price['kdcode']==toSell[i]].empty: # 部分股票当日停牌
    #                 code_price = 0.0
    #             else :
    #                 code_price = 1.0*now_price[now_price['kdcode']==toSell[i]]['close'].values
    #             self.all_money += code_price*self.counts*(1+self.adj_turn)

    #         return self

    #     # 对于需要持仓的股票  买入
    #     def buy_2(self, toBuy, now_price):
    #         sum_price = 0.00 # 需要买的股票的总价
    #         for i in range(len(toBuy)):
    #             sum_price += 1.0*now_price[now_price['kdcode']==toBuy[i]]['close'].values
    #         tmp = 0.0
    #         if sum_price:
    #             tmp = self.all_money/sum_price
    #         self.counts = int(tmp*1.0) # 股票份额 下取整
    #         self.all_money -= sum_price*self.counts*(1+self.adj_turn) # 剩余资产-现金流
    #         return self

    # 主函数 遍历所有交易日
    def my_run(self):
        pred_files = os.listdir(self.res_path)  # 获取所有的预测文件
        pred_files.sort()  # 按日期排序
        asset_list = []  # 记录总资产变化
        print("开始交易: ")
        price = self.load_history()
        for files in pred_files:
            today = files.split('.')[0]
            today_price = price[price['dt'] == today]
            self.trading(files, today_price)
            self.count_days += 1
            my_money = self.sum_money(today_price)
            print(my_money)
            asset_list.append(my_money)
        return asset_list

    # 计算当前资产总额--现金+股票
    def sum_money(self, today_price):
        now_stock = self.hold['kdcode'].values  # 获取当前持仓股票
        hold = self.hold
        s_m = self.all_money * 1.0
        for i in range(len(now_stock)):
            if today_price[today_price['kdcode'] == now_stock[i]].empty:  # 部分股票当日停牌
                code_price = 0.0
                s_m += 0
            else:
                code_price = 1.0 * today_price[today_price['kdcode'] == now_stock[i]]['close'].values
                code_count = 1.0 * hold[hold['kdcode'] == now_stock[i]]['hold'].values
                s_m += code_price[0] * code_count[0]
        return s_m

    # 对当日股票进行排序
    def get_rank(self, file_path):
        stock = pd.read_csv(file_path)
        stock = stock.sort_values(by="score", ascending=False)
        stock = stock[0:20].reset_index(drop=True)
        stock = stock[['kdcode', 'dt']]
        return stock

    # 评估函数
    def evaluate(self):
        asset_list = self.my_run()
        print(asset_list)
        print("年化利率： ")
        print(self.evaluate_AnnualizedReturn(asset_list))
        print("夏普率： ")
        print(self.evaluate_SharpeRatio(asset_list))
        print("年化波动： ")
        print(self.evaluate_Volatility(asset_list))
        print("最大回撤: ")
        print(self.evaluate_MaxDrawDown(asset_list))

    # 年化利率
    def evaluate_AnnualizedReturn(self, asset_list):
        test_return = (asset_list[-1] - asset_list[0]) / asset_list[0]
        test_days = len(asset_list) - 1
        ar = (test_return + 1) ** (365 / test_days) - 1
        return ar

    # 夏普率
    def evaluate_SharpeRatio(self, asset_list):
        ar = self.evaluate_AnnualizedReturn(asset_list)
        v = self.evaluate_Volatility(asset_list)
        return ar / v

    # 年化波动
    def evaluate_Volatility(self, asset_list):
        daily_return = [(i - j) / j for i, j in zip(asset_list[1:], asset_list[:-1])]
        daily_return = [i * 365 for i in daily_return]
        erp = sum(daily_return) / len(daily_return)
        v = sum((i - erp) * (i - erp) for i in daily_return)
        v = math.sqrt(v / len(daily_return))
        return v

    # 最大回撤
    def evaluate_MaxDrawDown(self, asset_list):
        min_asset = asset_list.copy()
        for i in range(len(min_asset) - 2, -1, -1):
            min_asset[i] = min(min_asset[i], min_asset[i + 1])
        mdd = max(-(min_asset[i] - asset_list[i]) / asset_list[i] for i in range(len(min_asset)))
        return mdd
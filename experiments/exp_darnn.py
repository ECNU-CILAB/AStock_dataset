import os
import torch
import numpy as np
import pandas as pd
from models.DARNN import DA_rnn

def previous_trading_day(dts, raw_date, days):
    for i, dt in enumerate(dts):
        if dt >= raw_date and i > days:
            return dts[i - days]
    return None


class Exp_DARNN:
    def __init__(self, data_path, res_path, model_path, val_date, test_date, group, t, gpu, batchsize, nhidden_encoder,
                 nhidden_decoder, lr, epoch, train, predict,fix):
        self.data_path = f"{data_path}/{group}_t{t}.csv"
        self.res_path = res_path
        self.model_path = model_path
        self.encoding_model_save_file = f"{model_path}/encoder_run_{group}{t}.pt"
        self.decoding_model_save_file = f"{model_path}/dencoder_run_{group}{t}.pt"
        self.val_date = val_date
        self.test_date = test_date
        self.group = group
        self.t = t
        self.batchsize = batchsize
        self.nhidden_encoder = nhidden_encoder
        self.nhidden_decoder = nhidden_decoder
        self.lr = lr
        self.epoch = epoch

        if gpu >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        if train == 1:
            self.train = True
        else:
            self.train = False

        if predict == 1:
            self.predict = True
        else:
            self.predict = False

        if fix==1:
            torch.manual_seed(4321)  # reproducible
            torch.cuda.manual_seed_all(4321)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
            torch.backends.cudnn.enabled = True

    def prepare_data(self):
        df = pd.read_csv(self.data_path)
        p = df.groupby("kdcode")['dt'].count() == (df['dt'].nunique())
        p = p.reset_index()
        stock_list = p[p['dt'] == True]['kdcode'].to_list()
        df = df[df.kdcode.isin(stock_list)]
        df.sort_values(by=["kdcode", "dt"], inplace=True)

        dts = df['dt'].unique().tolist()
        test_date_true = previous_trading_day(dts, self.test_date, self.t - 1)  # for final alignment of data
        pre_train_data = df[df['dt'] < self.val_date].reset_index(drop=True)
        pre_valid_data = df[(df['dt'] >= self.val_date) & (df['dt'] < test_date_true)].reset_index(drop=True)
        pre_test_data = df[df['dt'] >= test_date_true].reset_index(drop=True)

        features = df.columns.to_list()
        features = [x for x in features if 'return' not in x]
        features = list(set(features) - set(['kdcode', 'dt']))

        X_train = pre_train_data[features].values
        y_train = pre_train_data[f"t{self.t}_close_return_rate"].values
        X_val = pre_valid_data[features].values
        y_val = pre_valid_data[f"t{self.t}_close_return_rate"].values
        X_test = pre_test_data[features].values
        y_test = pre_test_data[f"t{self.t}_close_return_rate"].values

        return X_train, y_train, X_val, y_val, X_test, y_test, pre_test_data

    def run_model(self):
        if self.train:
            X_train, y_train, X_val, y_val, X_test, y_test, pre_test_data = self.prepare_data()
            model = DA_rnn(
                X_train,
                y_train,
                X_val,
                y_val,
                self.t,
                self.nhidden_encoder,
                self.nhidden_decoder,
                self.batchsize,
                self.lr,
                self.epoch,
                self.encoding_model_save_file,
                self.decoding_model_save_file
            )
            # Train
            print("==> Start training ...")
            model.train()
        if self.predict:
            if not self.train:
                model.Encoder.load_state_dict(torch.load(self.encoding_model_save_file))
                model.Decoder.load_state_dict(torch.load(self.decoding_model_save_file))
            pred_score = model.predict(X_test, y_test)
            score = pred_score.reshape(-1, 1)
            score_full = np.zeros((X_test.shape[0], 1))
            score_full[self.t - 1:, :] = score
            pre_test_data['score'] = score_full
            pre_test_data = pre_test_data[['kdcode', 'dt', 'score']]
            test_data_np = pre_test_data[pre_test_data['dt'] >= self.test_date]
            test_data_np = test_data_np.reset_index(drop=True)
            result_root_dir = f"{self.res_path}/{self.group}_t{self.t}"
            if not os.path.exists(result_root_dir):
                os.mkdir(result_root_dir)
            if not os.path.exists(result_root_dir + "/prediction"):
                os.mkdir(result_root_dir + "/prediction")
            test_data_np.to_csv(f"{result_root_dir}/pred_score.csv", index=False)
            for key, val in test_data_np.groupby("dt"):
                val.to_csv(f"{result_root_dir}/prediction/{key}.csv", index=False)
                print(f"{result_root_dir}/prediction/{key}.csv生成！")

    def predict(self):
        X_train, y_train, X_val, y_val, X_test, y_test, pre_test_data = self.prepare_data(True)
        model = DA_rnn(
            X_train,
            y_train,
            X_val,
            y_val,
            self.t,
            self.nhidden_encoder,
            self.nhidden_decoder,
            self.batchsize,
            self.lr,
            self.epoch,
            self.encoding_model_save_file,
            self.decoding_model_save_file
        )
        model.Encoder.load_state_dict(torch.load(self.encoding_model_save_file))
        model.Decoder.load_state_dict(torch.load(self.decoding_model_save_file))
        pred_score = model.predict(X_test, y_test)
        score = pred_score.reshape(-1, 1)
        score_full = np.zeros((X_test.shape[0], 1))
        score_full[self.t - 1:, :] = score
        pre_test_data['score'] = score_full
        pre_test_data = pre_test_data[['kdcode', 'dt', 'score']]
        test_data_np = pre_test_data[pre_test_data['dt'] >= self.test_date]
        test_data_np = test_data_np.reset_index(drop=True)
        result_root_dir = f"{self.res_path}/{self.group}_t{self.t}"
        if not os.path.exists(result_root_dir):
            os.mkdir(result_root_dir)
        if not os.path.exists(result_root_dir + "/prediction"):
            os.mkdir(result_root_dir + "/prediction")
        test_data_np.to_csv(result_root_dir, index=False)
        for key, val in test_data_np.groupby("dt"):
            val.to_csv(f"{result_root_dir}/prediction/{key}.csv", index=False)
            print(f"{result_root_dir}/prediction/{key}.csv生成！")
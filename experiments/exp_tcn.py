import os
import torch
import numpy as np
import pandas as pd
from models.TCN import TCN

def previous_trading_day(dts, raw_date, days):
    for i, dt in enumerate(dts):
        if dt >= raw_date and i > days:
            return dts[i - days]
    return None


class Exp_TCN:
    def __init__(self, data_path, res_path, model_path, val_date, test_date, group, t, gpu, batchsize,input_size,
                output_size,kernel_size,dropout,lr, epoch, train, predict,fix):
        self.data_path = f"{data_path}/{group}_t{t}.csv"
        self.res_path = res_path
        self.model_path = model_path
        self.val_date = val_date
        self.test_date = test_date
        self.group = group
        self.t = t
        self.batchsize = batchsize
        self.lr = lr
        self.epoch = epoch
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        self.kernel_size = kernel_size

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
        X_train, y_train, X_val, y_val, X_test, y_test, pre_test_data = self.prepare_data()
        # 将数据放到适合lstm输入的三维数据
        x_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        x_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))
        x_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

        y_train = np.reshape(y_train, (y_train.shape[0], 1))
        y_val = np.reshape(y_val, (y_val.shape[0], 1))
        y_test = np.reshape(y_test, (y_test.shape[0], 1))

        # 处理成torch变量
        x_train = torch.Tensor(x_train)
        y_train = torch.Tensor(y_train)

        x_val = torch.Tensor(x_val)
        y_val = torch.Tensor(y_val)

        x_test = torch.Tensor(x_test)
        y_test = torch.Tensor(y_test)

        # 定义数据加载集
        train_set = torch.utils.data.TensorDataset(x_train, y_train)
        val_set = torch.utils.data.TensorDataset(x_val, y_val)
        test_set = torch.utils.data.TensorDataset(x_test, y_test)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batchsize, shuffle=True, num_workers=8)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batchsize, shuffle=True, num_workers=8)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batchsize, shuffle=True, num_workers=8)


        if self.train:
            num_channels = [self.input_size, self.output_size]
            model = TCN(
                self.input_size,
                self.output_size,
                num_channels,
                self.kernel_size,
                self.dropout
            )
            # Train
            print("==> Start training ...")
            loss = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, betas=(0.45, 0.35), weight_decay=0.4)

            for epoch in range(0, self.epoch):
                epoch_loss = []
                for step, (x_train, y_train) in enumerate(train_loader):
                    y_pred = model(x_train)
                    loss_value = loss(y_pred, y_train)
                    model.zero_grad()
                    loss_value.backward()
                    optimizer.step()
                    epoch_loss.append(loss_value.item())
                #         print(loss_value.item())

                print("Epoch %d final had loss %.4f" % (epoch, np.mean(np.array(epoch_loss))))

            validation_loss = []
            with torch.no_grad():
                model.eval()
                for i in range(x_val.shape[0]):
                    x_input = x_val[i:i + 1, :, :]
                    y_target = y_val[i:i + 1, :]

                    x_input = x_input
                    y_target = y_target
                    y_hat = model(x_input)
                    val_loss = loss(y_hat, y_target)
                    validation_loss.append(val_loss.item())

            print('average validaiton loss :', np.mean(np.array(validation_loss)))

            torch.save(model.state_dict(), self.model_path)
        if self.predict:
            if not self.train:
                model = model.load_state_dict(torch.load(self.model_path))
            y_pred = []
            epoch_loss = []
            for step, (x_test, y_test) in enumerate(test_loader):
                y_p = model(x_test)
                loss_value = loss(y_p, y_test)
                y_pred.extend(y_p.tolist())
                epoch_loss.append(loss_value.item())

            print("final had loss %.4f" % (np.mean(np.array(epoch_loss))))
            y_pred = np.array(y_pred)
            pre_test_data['score'] = y_pred
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
        x_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        y_test = np.reshape(y_test, (y_test.shape[0], 1))
        x_test = torch.Tensor(x_test)
        y_test = torch.Tensor(y_test)
        test_set = torch.utils.data.TensorDataset(x_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=512, shuffle=True, num_workers=8)
        num_channels = [self.input_size, self.output_size]
        model = TCN(
            self.input_size,
            self.output_size,
            num_channels,
            self.kernel_size,
            self.dropout
        )
        model = model.load_state_dict(torch.load(self.model_path))

        y_pred = []
        epoch_loss = []
        loss = torch.nn.MSELoss()
        for step, (x_test, y_test) in enumerate(test_loader):
            y_p = model(x_test)
            loss_value = loss(y_p, y_test)
            y_pred.extend(y_p.tolist())
            epoch_loss.append(loss_value.item())

        print("final had loss %.4f" % (np.mean(np.array(epoch_loss))))
        y_pred = np.array(y_pred)
        pre_test_data['score'] = y_pred
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
from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
import json
import json
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.metrics import metric, R2
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        opt_name = str(getattr(self.args, 'optim', 'adam')).lower()
        weight_decay = getattr(self.args, 'weight_decay', 0.0)
        if opt_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=weight_decay)
        if opt_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9, weight_decay=weight_decay)
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=weight_decay)

    def _select_criterion(self):
        if self.args.loss == 'MAE':
            criterion = nn.L1Loss()
        elif self.args.loss == 'MSE':
            criterion = nn.MSELoss()
        elif self.args.loss == 'SmoothL1':
            criterion = nn.SmoothL1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                target_idx = getattr(vali_data, 'target_indices', None)
                if target_idx:
                    idx = torch.as_tensor(target_idx, device=outputs.device, dtype=torch.long)
                    outputs = torch.index_select(outputs, 2, idx)
                    batch_y = torch.index_select(batch_y, 2, idx)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        # 若数据集提供 target_indices，则只对目标列计算 loss
                        target_idx = getattr(train_data, 'target_indices', None)
                        if target_idx:
                            idx = torch.as_tensor(target_idx, device=outputs.device, dtype=torch.long)
                            outputs = torch.index_select(outputs, 2, idx)
                            batch_y = torch.index_select(batch_y, 2, idx)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    target_idx = getattr(train_data, 'target_indices', None)
                    if target_idx:
                        idx = torch.as_tensor(target_idx, device=outputs.device, dtype=torch.long)
                        outputs = torch.index_select(outputs, 2, idx)
                        batch_y = torch.index_select(batch_y, 2, idx)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if not torch.isfinite(loss):
                    print(f"Skip step: non-finite loss at epoch {epoch + 1}, iter {i + 1}: {loss.item()}")
                    continue

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = None
            test_loss = None

            if self.args.val_interval and (epoch + 1) % self.args.val_interval == 0:
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            if self.args.test_interval and (epoch + 1) % self.args.test_interval == 0:
                test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3} Test Loss: {4}".format(
                epoch + 1, train_steps, train_loss,
                f"{vali_loss:.7f}" if vali_loss is not None else 'NA',
                f"{test_loss:.7f}" if test_loss is not None else 'NA'))

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []

        eval_horizons = []
        for h in str(self.args.eval_horizons).split(','):
            h = h.strip()
            if not h:
                continue
            try:
                eval_horizons.append(int(h))
            except ValueError:
                continue
        eval_horizons = sorted({min(self.args.pred_len, h) for h in eval_horizons if h > 0})
        if self.args.pred_len not in eval_horizons:
            eval_horizons.append(self.args.pred_len)
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                target_idx = getattr(test_data, 'target_indices', None)
                if target_idx:
                    idx = torch.as_tensor(target_idx, device=outputs.device, dtype=torch.long)
                    outputs = torch.index_select(outputs, 2, idx)
                    batch_y = torch.index_select(batch_y, 2, idx)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0), only_target=True).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0), only_target=True).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        # Full sequence includes features + targets; inverse both parts.
                        input = test_data.inverse_transform(input.squeeze(0), only_target=False).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        horizon_results = {}
        for h in eval_horizons:
            p_slice = preds[:, :h, :]
            t_slice = trues[:, :h, :]
            mae, mse, rmse, mape, mspe = metric(p_slice, t_slice)
            r2 = R2(p_slice, t_slice)
            horizon_results[str(h)] = {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'mape': float(mape),
                'mspe': float(mspe),
                'r2': float(r2),
            }

        full_key = str(max(eval_horizons))
        full_metrics = horizon_results[full_key]
        print('Full horizon ({}): mse:{}, mae:{}, r2:{}'.format(full_key, full_metrics['mse'], full_metrics['mae'], full_metrics['r2']))

        # legacy text log
        with open("result_long_term_forecast.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write('full_horizon({}): mse:{}, mae:{}, r2:{}'.format(
                full_key, full_metrics['mse'], full_metrics['mae'], full_metrics['r2']))
            f.write('\n\n')

        # save metrics and preds
        np.save(folder_path + 'metrics.npy', np.array([full_metrics['mae'], full_metrics['mse'], full_metrics['rmse'], full_metrics['mape'], full_metrics['mspe'], full_metrics['r2']]))
        with open(folder_path + 'metrics_detail.json', 'w') as f:
            json.dump(horizon_results, f, indent=2)
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        # optional plotting: first N samples, all targets合并一张图（仅测试结束后生成）
        num_plot = max(0, int(self.args.plot_samples)) if hasattr(self.args, 'plot_samples') else 0
        if num_plot > 0:
            target_names = getattr(test_data, 'target_cols', None)
            if not target_names or len(target_names) != preds.shape[-1]:
                target_names = [f'target{i}' for i in range(preds.shape[-1])]
            num_plot = min(num_plot, preds.shape[0])
            for i in range(num_plot):
                plt.figure(figsize=(9, 4))
                for t_idx in range(preds.shape[-1]):
                    plt.plot(trues[i, :, t_idx], label=f'true_{target_names[t_idx]}')
                    plt.plot(preds[i, :, t_idx], '--', label=f'pred_{target_names[t_idx]}')
                plt.title(f'sample{i} prediction vs true')
                plt.xlabel('time step')
                plt.ylabel('value')
                plt.legend(ncol=2)
                plt.tight_layout()
                out_path = os.path.join(folder_path, f'sample{i}_targets.png')
                plt.savefig(out_path)
                plt.close()

        return


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                target_idx = getattr(pred_data, 'target_indices', None)
                if target_idx:
                    idx = torch.as_tensor(target_idx, device=outputs.device, dtype=torch.long)
                    outputs = torch.index_select(outputs, 2, idx)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
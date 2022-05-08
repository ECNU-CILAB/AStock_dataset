import argparse
import os

desc = 'the lstm model'
parser = argparse.ArgumentParser(description=desc)
# parser.add_argument('-p', '--path', help='path of pv data', type=str,
#                     default='./data/stocknet-dataset/price/ourpped')
parser.add_argument('-p', '--path', help='path of pv data', type=str,
                    default='./data/dataset')
parser.add_argument('-l', '--seq', help='length of history', type=int,
                    default=5)
parser.add_argument('-group', '--group', help='exp group', type=str,
                    default='p')
parser.add_argument('-u', '--unit', help='number of hidden units in lstm',
                    type=int, default=32)
parser.add_argument('-l2', '--alpha_l2', type=float, default=1e-2,
                    help='alpha for l2 regularizer')
parser.add_argument('-la', '--beta_adv', type=float, default=1e-2,
                    help='beta for adverarial loss')
parser.add_argument('-le', '--epsilon_adv', type=float, default=1e-2,
                    help='epsilon to control the scale of noise')
parser.add_argument('-s', '--step', help='steps to make prediction',
                    type=int, default=1)
parser.add_argument('-b', '--batch_size', help='batch size', type=int,
                    default=1025)
parser.add_argument('-e', '--epoch', help='epoch', type=int, default=20)
parser.add_argument('-r', '--learning_rate', help='learning rate',
                    type=float, default=1e-4)
parser.add_argument('-g', '--gpu', type=int, default=1, help='use gpu')
parser.add_argument('-q', '--model_path', help='path to load model',
                    type=str, default='./saved_model/dataset')
parser.add_argument('-qs', '--model_save_path', type=str, help='path to save model',
                    default='./saved_model/dataset/model_pf5')
parser.add_argument('-o', '--action', type=str, default='train',
                    help='train, test, pred')
parser.add_argument('-m', '--model', type=str, default='pure_lstm',choices=['pure_lstm','att_lstm','aw_lstm','free_model'],
                    help='pure_lstm, att_lstm, aw_lstm, free_model')
parser.add_argument('-f', '--fix_init', type=int, default=1,
                    help='use fixed initialization')
parser.add_argument('-a', '--att', type=int, default=1,
                    help='use attention model')
parser.add_argument('-w', '--week', type=int, default=0,
                    help='use week day data')
parser.add_argument('-v', '--adv', type=int, default=1,
                    help='adversarial training')
parser.add_argument('-hi', '--hinge_lose', type=int, default=1,
                    help='use hinge lose')
parser.add_argument('-rl', '--reload', type=int, default=1,
                    help='use pre-trained parameters')
parser.add_argument('-rl', '--tra_date', type=str, default="2018-01-02",
                    help='begin date of training data')
parser.add_argument('-rl', '--val_date', type=str, default="2020-06-01",
                    help='begin date of training data')
parser.add_argument('-rl', '--tes_date', type=str, default="2020-12-31",
                    help='begin date of training data')

args = parser.parse_args()
print(args)

parameters = {
    'seq': int(args.seq),
    'unit': int(args.unit),
    'alp': float(args.alpha_l2),
    'bet': float(args.beta_adv),
    'eps': float(args.epsilon_adv),
    'lr': float(args.learning_rate)
}

if __name__ == '__main__':
    """
    model_path 是模型的保存位置用于saver类使用
    model_save_path 是保存最后预测结果的地址
    """
    if args.model == "pure_lstm":
        args.att = 0
        args.adv = 0
        args.model_save_path = f'{args.model_path}/lstm/{args.group}_t{args.seq}/'
        if not os.path.exists(args.model_save_path):
            os.mkdir(args.model_save_path)
        args.model_path = f'{args.model_path}/lstm/{args.group}_t{args.seq}/exp'
    elif args.model == "att_lstm":
        args.att = 1
        args.adv = 0
        args.model_save_path = f'{args.model_path}/alstm/{args.group}_t{args.seq}/'
        if not os.path.exists(args.model_save_path):
            os.mkdir(args.model_save_path)
        args.model_path = f'{args.model_path}/alstm/{args.group}_t{args.seq}/exp'
    elif args.model == "aw_lstm":
        args.att = 1
        args.adv = 1
        args.model_save_path = f'{args.model_path}/adv_alstm/{args.group}_t{args.seq}/'
        if not os.path.exists(args.model_save_path):
            os.mkdir(args.model_save_path)
        args.model_path = f'{args.model_path}/adv_alstm/{args.group}_t{args.seq}/exp'
    else:
        pass

    # print(f'basic info: {args.group}_t{args.seq},{args.model_path},{args.model_save_path}')
    pure_LSTM = AWLSTM(
        data_path=f'{args.path}/{args.group}_t{args.seq}',
        model_path=args.model_path,
        model_save_path=args.model_save_path,
        parameters=parameters,
        steps=args.step,
        epochs=args.epoch, batch_size=args.batch_size, gpu=args.gpu,
        tra_date=tra_date, val_date=val_date, tes_date=tes_date, att=args.att,
        hinge=args.hinge_lose, fix_init=args.fix_init, adv=args.adv,
        reload=args.reload
    )
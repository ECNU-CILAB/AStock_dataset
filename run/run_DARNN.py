import argparse
from experiments.exp_darnn import Exp_DARNN

desc = 'DARNN'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-dp', '--data_path', help='path of data', type=str,
                    default='../datasets/raw')
parser.add_argument('-rp', '--res_path', help='path to save result', type=str,
                    default='../res/DARNN')
parser.add_argument('-mp', '--model_path', help='path to save model', type=str,
                    default='../model_saved/DARNN')
parser.add_argument('-gpu', '--gpu', help='gpu', type=int,
                    default=0)
parser.add_argument('-g', '--group', help='exp group', type=str,
                    default="p")
parser.add_argument('-t', '--t', help='trading intervals', type=int,
                    default=5)
parser.add_argument('-vd', '--val_date', help='val_date', type=str,
                    default="2020-06-01")
parser.add_argument('-td', '--test_date', help='val_date', type=str,
                    default="2020-12-31")

#######################################################################
parser.add_argument('-bt', '--batchsize', help='batchsize', type=int,
                    default=1024)
parser.add_argument('--nhidden_encoder', help='nhidden_encoder', type=int,
                    default=55)
parser.add_argument('--nhidden_decoder', help='nhidden_decoder', type=int,
                    default=55)
parser.add_argument('-lr', help='learning rate', type=int, default=1e-4)
parser.add_argument('--epoch', help='epoch', type=int, default=20)
parser.add_argument('--train', help='train', type=int, default=1)
parser.add_argument('--predict', help='predict', type=int, default=1)
parser.add_argument('--fix', help='fix param', type=int, default=1)



if __name__ == '__main__':
    args = parser.parse_args()

    Exp = Exp_DARNN(
        data_path=args.data_path,
        res_path=args.res_path,
        model_path=args.model_path,
        val_date=args.val_date,
        test_date=args.test_date,
        group=args.group,
        t=args.t,
        gpu=args.gpu,
        batchsize=args.batchsize,
        nhidden_encoder=args.nhidden_encoder,
        nhidden_decoder=args.nhidden_decoder,
        lr=args.lr,
        epoch=args.epoch,
        train=args.train, predict=args.predict, fix=args.fix
    )

    Exp.run_model()
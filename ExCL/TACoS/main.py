import argparse,os,time,shutil,sys
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import numpy as np
from torch.nn.utils import clip_grad_norm
from torch.utils.data import DataLoader

from config import DefaultConfig
from dataset import TACoSDataset,padding_collate_fn
from model import ExCL



best_avg_recall = 0

def train(opt,resume=False):
    """
    训练过程
    """
    global best_avg_recall

    # 准备数据
    end = time.time()
    train_loader = DataLoader(TACoSDataset(opt,mode='train'),batch_size=opt.batch_size,shuffle=True,collate_fn=padding_collate_fn)
    val_loader = DataLoader(TACoSDataset(opt,mode='val'),batch_size=opt.batch_size,shuffle=True,collate_fn=padding_collate_fn)
    print("data loading: %.2f s"%(time.time()-end))
    # 加载模型
    model = ExCL(opt)
    # model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    if resume:
        resume_path = ''
        check_point = torch.load(resume_path)
        opt.strat_epoch = check_point['epoch']
        best_avg_recall = check_point['best_avg_recall']
        model.load_state_dict(check_point['state_dict'])


    # 损失函数，ExCL-clf 负似然对数
    criterion = torch.nn.NLLLoss()#.cuda()

    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(),lr=opt.lr)

    # 训练
    for epoch in range(opt.start_epoch, opt.train_epochs):
        # AverageMeter类管理变量更新
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        # 让模型进入训练模式
        model.train()

        end = time.time()
        for i, (input_video, input_sen_emb, g_position) in enumerate(train_loader):

            # 计算data loading的时间
            data_time.update(time.time() - end)

            # 转换成Variable
            x_video = torch.autograd.Variable(input_video)
            x_text = torch.autograd.Variable(input_sen_emb)
            g_position = g_position.transpose()
            g_start = torch.autograd.Variable(torch.from_numpy(g_position[0])).long()
            g_end = torch.autograd.Variable(torch.from_numpy(g_position[1])).long()

            # 计算输出和损失、准确度并保存当前batch的结果
            p_start,p_end = model(x_video,x_text)
            print(p_start) 
            print(p_start.size())
            print(g_start)
            loss_start = criterion(p_start, g_start)
            loss_end = criterion(p_end, g_end)
            loss = loss_start + loss_end
            losses.update(loss.item(), x_text.size()[0])

            # 梯度初始化为0
            optimizer.zero_grad()

            # 反向传播
            loss.backward()

            # 更新权重
            optimizer.step()

            # 计算batch消耗时间
            batch_time.update(time.time() - end)
            end = time.time()

            # 按照一定频率打印训练信息
            if i % 1 == 0:
                output = ('Epoch: [{0}][{1}/{2}]\t'
                        'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                            epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses))
                print(output)

        # 验证
        if (epoch + 1) % 1 == 0 or epoch == opt.train_epochs - 1:
            recall = validate(val_loader, model, criterion)

            print("R@0.3:",recall['0.3'],"R@0.5:",recall['0.5'],"R@0.7:",recall['0.7'])
            
            cur_avg_recall = (recall['0.3']+recall['0.5']+recall['0.7'])/3
            is_best = cur_avg_recall > best_avg_recall
            best_avg_recall = max(cur_avg_recall, best_avg_recall)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_avg_recall,
            }, opt, is_best) 



def calculate_tIoU(gt,pred_s,pred_e):
    list = []
    for i in gt.shape[0]:
        s_min = min(gt[0][i],pred_s[i])
        s_max = max(gt[0][i],pred_s[i])
        e_min = min(gt[1][i],pred_e[i])
        e_max = max(gt[1][i],pred_e[i])
        if e_min < s_max:
            tIoU =0
        else:
            tIoU = (e_min-s_max)*1.0/(e_max-s_min)
        list.append(tIoU)
    return list


def validate(val_loader, model, criterion):
    """
    验证过程，大体类同train，但不需要更新权重
    """
    tIoUs = []

    batch_time = AverageMeter()
    losses = AverageMeter()

    model.eval()

    end = time.time()
    for i, (input_video, input_sen_emb, g_position) in enumerate(train_loader):

        x_video = torch.autograd.Variable(input_video)
        x_text = torch.autograd.Variable(input_sen_emb)
        g_position = (np.array(g_position)).transpose
        g_start = torch.autograd.Variable(g_position[0])
        g_end = torch.autograd.Variable(g_position[1])

        p_start,p_end = model(x_video,x_text) 
        loss_start = criterion(p_start, g_start)
        loss_end = criterion(p_end, g_end)
        loss = loss_start + loss_end
        losses.update(loss.data[0], x_text.size(0))

        # 计算iou
        pred_start = np.argmax(p_start.numpy(),axis=1)
        pred_end = np.argmax(p_end.numpy(),axis=1)
        tIoUs.append(calculate_tIoU(g_position,pred_s,pred_e))

        batch_time.update(time.time() - end)
        end = time.time()

        
        if i % 1 == 0:
            output = ('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses))
            print(output)
        
    tIoUs = (np.array(tIoUs)).flatten()
    recall = {}
    recall['0.3'] = 0
    recall['0.5'] = 0
    recall['0.7'] = 0
    for tIoU in tIoUs:
        if tIoU >= 0.3:
            recall['0.3'] = recall['0.3'] + 1
        if tIoU >= 0.5:
            recall['0.5'] = recall['0.5'] + 1
        if tIoU >= 0.7:
            recall['0.7'] = recall['0.7'] + 1
    recall['0.3'] = recall['0.3']/tIoUs.shape[0]
    recall['0.5'] = recall['0.5']/tIoUs.shape[0]
    recall['0.7'] = recall['0.7']/tIoUs.shape[0]

    return recall

def save_checkpoint(state, opt, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, opt.model_save_path+"ckpt_tacos_epoch"+str(state['epoch']-1)+".pth.tar")
    if is_best:
        shutil.copyfile(opt.model_save_path+"ckpt_tacos_epoch"+str(state['epoch']-1)+".pth.tar",opt.model_save_path+"ckpt_tacos_best.pth.tar")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# def adjust_learning_rate(optimizer, epoch, args):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     decay = 0.1 ** (sum(epoch >= np.array(args.lr_steps)))
#     lr = args.lr * decay
#     decay = args.weight_decay
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr * param_group['lr_mult']
#         param_group['weight_decay'] = decay * param_group['decay_mult']


def parse_args():
    """
    解析参数
    """
    parser = argparse.ArgumentParser(description='ExCL to train or test')
    parser.add_argument('--task', dest='task',
                        help='train or test',
                        default='train', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    opt = DefaultConfig()
    if args.task == 'train':
        train(opt,resume=False)
    if args.task == 'test':
        test(opt)
    if args.task == 'locate':
        pass
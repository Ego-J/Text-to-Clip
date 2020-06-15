import torch
import torch.nn as nn
from torch.autograd import Variable
from config import DefaultConfig

class ExCL(nn.Module):
    def __init__(self,opt):
        super(ExCL,self).__init__()

        self.opt = opt

        self.video_LSTM = nn.LSTM(input_size=opt.vLSTM_embedding_dim,
                                  hidden_size=opt.vLSTM_hidden_dim,
                                  dropout=opt.dropout,
                                  batch_first=True,
                                  bidirectional=True)

        self.text_LSTM = nn.LSTM(input_size=opt.tLSTM_embedding_dim,
                                 hidden_size=opt.tLSTM_hidden_dim,
                                 dropout=opt.dropout,
                                 batch_first=True,
                                 bidirectional=True)

        self.span_LSTM = nn.LSTM(input_size=opt.sLSTM_embedding_dim,
                                 hidden_size=opt.sLSTM_hidden_dim,
                                 dropout=opt.dropout,
                                 batch_first=True,
                                 bidirectional=True)     

        self.start_predictor = nn.Sequential(nn.Linear(opt.MLP_input_dim,opt.MLP_hidden_dim),
                                             nn.Tanh(),
                                             nn.Linear(opt.MLP_hidden_dim,1))

        self.end_predictor = nn.Sequential(nn.Linear(opt.MLP_input_dim,opt.MLP_hidden_dim),
                                           nn.Tanh(),
                                           nn.Linear(opt.MLP_hidden_dim,1))

    def forward(self,x_video,x_text): # x_video(b,v_len,4096) x_text(b,s_len,300)
        

        opt = self.opt
        v_len = x_video.size()[1]
        # 将video input送入LSTM中，得到每一个时间步上的输出，输出大小为hidden_dim*direction
        # 将text input送入LSTM中，得到最后一个时间步上的输出，大小分别为前向的hidden_dim+后向的hidden_dim
        x_video , __ = self.video_LSTM(x_video) # x_video(b,v_len,256*2)
        __ , (x_text , __) = self.text_LSTM(x_text) # x_text(2,b,256)
        x_text = x_text.permute(1,0,2).contiguous() # x_text(b,2,256)
        # 将文本编码LSTM输出的前向和后向进行拼接，然后扩张到v_len长度，再与视频编码LSTM输出进行拼接
        x_text = x_text.view(-1,1,opt.tLSTM_hidden_dim*2) # x_text(b,256*2)
        x_text = x_text.expand(-1,v_len,-1) # x_text(b,vlen,256*2)
        x_v_t = torch.cat((x_video,x_text),2) # x_v_t(b,vlen,256*2+256*2)
        # 将拼接输入送入LSTM中，将输出再与输入进行拼接，变形成适合MLP的输入尺寸
        x_s , __ = self.span_LSTM(x_v_t) # x_s(b,vlen,128*2)
        x = torch.cat((x_s,x_v_t),2) # x(b,vlen,256*2+256*2+128*2)
        x = x.view(-1,opt.MLP_input_dim) # x(b*vlen,256*2+256*2+128*2)
        # 先得到当前时间步的预测分数，然后变形后使用softmax得到概率
        # 起点预测
        S_start = self.start_predictor(x) # S_start(b*vlen,1)
        S_start = S_start.view(-1,v_len) # S_start(b,vlen)

        # 终点预测
        S_end = self.end_predictor(x)
        S_end = S_end.view(-1,v_len) 
        P_end = nn.functional.log_softmax(S_end,dim=1) # 同上

        return S_start,S_end

if __name__ == "__main__":
    '''
    模型测试
    '''
    x_video = torch.randn(10,30,4096)
    x_text = torch.randn(10,10,300)
    opt = DefaultConfig()
    model = ExCL(opt)
    print(model)

    P_start,P_end = model(x_video,x_text)
    print(P_start.size())
    print(P_start)
    print(P_end.size())
    print(P_end)


    
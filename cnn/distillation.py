import torch
import torch.nn.functional
from torch import nn
import torch.backends.cudnn as cudnn

def kd_loss_function(output, target_output,args):
    output=output/args.temperature
    output_log_softmax=torch.log_softmax(output,dim=1)
    return torch.mean(torch.sum(output_log_softmax*target_output,dim=1))

def feature_loss_function(feature,target_feature):
    loss=(feature-target_feature)**2*((feature>0)|(target_feature>0)).float()
    return torch.abs(loss).sum()/feature.numel()

def self_distillation(model, midden_output, target, args):

    #midden_input initialize
    cudnn.benchmark = True
    ops=nn.ModuleList()
    ops.append(nn.Sequential(nn.ReLU(),nn.Conv2d(100,400,kernel_size=6,padding=1,stride=4),nn.BatchNorm2d(400)))
    ops.append(nn.Sequential(nn.ReLU(),nn.Conv2d(200,400,kernel_size=3,padding=1,stride=2),nn.BatchNorm2d(400)))
    #ops.append(nn.Sequential(nn.ReLU(),nn.Conv2d(128,256,kernel_size=3,padding=1,stride=2),nn.BatchNorm2d(256)))
    midden_output_fea=[]
    ops.cuda()
    for i in range(2):
        midden_output_fea.append(ops[i](midden_output[i][1]))
    # midden_output_fea.append(midden_output[1][1])
    midden_output_fea.append(midden_output[2][1])
    #midden_output_fea.append(midden_output[3][1])

    criterion = nn.CrossEntropyLoss().cuda()
    midden_net=midden_block().cuda()

    #midden_cell1, midden_cell2, midden_cell3=midden_output[0][0],midden_output[1][0],midden_output[2][0]

    final_fea=model.get_final_fea()
    middle1_fea, middle2_fea, middle3_fea=midden_output_fea[0],midden_output_fea[1],midden_output_fea[2]#,midden_output_fea[3]
    
    middle_output1, middle_output2, middle_output3=midden_net(midden_output_fea[0]),midden_net(midden_output_fea[1]),midden_net(midden_output_fea[2])#,midden_net(midden_output_fea[3])


    middle1_loss = criterion(middle_output1, target)
    middle2_loss = criterion(middle_output2, target)
    middle3_loss = criterion(middle_output3, target)
    #middle4_loss = criterion(middle_output4, target)
    

    feature_loss_1 = feature_loss_function(middle1_fea, final_fea.detach()) 
    feature_loss_2 = feature_loss_function(middle2_fea, final_fea.detach()) 
    feature_loss_3 = feature_loss_function(middle3_fea, final_fea.detach())   
    #feature_loss_4 = feature_loss_function(middle4_fea, final_fea.detach())  

    final_out=model.get_final_out()
    temp4 = final_out / args.temperature
    temp4 = torch.softmax(temp4, dim=1)        

    loss1by4 = kd_loss_function(middle_output1, temp4.detach(), args) * (args.temperature**2)
    loss2by4 = kd_loss_function(middle_output2, temp4.detach(), args) * (args.temperature**2)
    loss3by4 = kd_loss_function(middle_output3, temp4.detach(), args) * (args.temperature**2)
    #loss4by4 = kd_loss_function(middle_output4, temp4.detach(), args) * (args.temperature**2)

            
    loss_fea=feature_loss_1+feature_loss_2+feature_loss_3#+feature_loss_4
    loss_l=middle1_loss+middle2_loss+middle3_loss#+middle4_loss
    loss_s_t=loss1by4+loss2by4+loss3by4#+loss4by4

    return loss_l, loss_s_t, loss_fea


class midden_block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(400,256,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,64,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,16,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,10,kernel_size=1),
            # nn.AdaptiveAvgPool2d(1)
            nn.AdaptiveAvgPool2d(1),
            nn.Softmax()
        )
    
    def forward(self,x):
        return self.net(x).reshape(x.shape[0],-1)
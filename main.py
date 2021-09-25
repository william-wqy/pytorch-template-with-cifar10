from dataloader import get_dataloader
from model import CNN
import torch.nn as nn
import torch

def train():
    data=get_dataloader('train',10)
    net=CNN()
    lossfunc=nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(net.parameters(),lr=0.01)


    for epoch in range(30):
        net.train()
        lo=[]
        for index,(dt,label) in enumerate(data):
            optimizer.zero_grad()
            output=net(dt)
            loss=lossfunc(output,label)
            loss.backward()
            optimizer.step()
            if index%100==0:
                print (loss.data)
                lo.append(float(loss.data))

        print(epoch,'finished')

    torch.save(net,'trained.pkl')
    #draw(lo)
    print('train finished')

def predict():
    data=get_dataloader('test')
    net=torch.load('trained.pkl')
    net.eval()#??
    test_acc = 0

    for dt,label in data:
        outs=net(dt) 
        prediction=torch.max(outs,1)[1] #(tensor,1) 只取每行的最大值，末尾[0]为返回最大数值，[1]为返回索引（即类别）

        for index,predict in enumerate(prediction):
            if predict==label[index]:
                test_acc+=1

    test_acc=float(test_acc)/10000
    print(test_acc)



if __name__=='__main__':
    #train()
    predict()

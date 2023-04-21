import torch
import torch.nn as nn
import torch.optim as optim
from model import DSCTransformer
from tqdm import tqdm
from data_set import MyDataset
from torch.utils.data import DataLoader

def train(epochs=100):
    #定义参数
    N = 4 #编码器个数
    input_dim = 1024
    seq_len = 16 #句子长度
    d_model = 64 #词嵌入维度
    d_ff = 256 #全连接层维度
    head = 4 #注意力头数
    dropout = 0.1
    lr = 5E-5 #学习率
    batch_size = 32

    #加载数据
    train_path = r'F:\PyCharmWorkSpace\BearingFaultDiagnosis\data\train\train.csv'
    val_path = r'F:\PyCharmWorkSpace\BearingFaultDiagnosis\data\val\val.csv'
    train_dataset = MyDataset(train_path, 'fd')
    val_dataset = MyDataset(val_path, 'fd')
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    #定义模型
    model = DSCTransformer(input_dim=input_dim, num_classes=10, dim=d_model, depth=N,
                           heads=head, mlp_dim=d_ff, dim_head=d_model, emb_dropout=dropout, dropout=dropout)
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("using {} device.".format(device))
    model.to(device)

    #训练
    for epoch in range(epochs):
        #train
        train_loss = []
        train_acc = []
        model.train()
        train_bar = tqdm(train_loader)
        for datas, labels in train_bar:
            optimizer.zero_grad()
            outputs = model(datas.float().to(device))
            loss = criterion(outputs, labels.type(torch.LongTensor).to(device))
            loss.backward()
            optimizer.step()

            # torch.argmax(dim=-1), 求每一行最大的列序号
            acc = (outputs.argmax(dim=-1) == labels.to(device)).float().mean()
            # Record the loss and accuracy
            train_loss.append(loss.item())
            train_acc.append(acc)
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss.item())

        #val
        model.eval()
        valid_loss = []
        valid_acc = []
        val_bar = tqdm(val_loader)
        for datas, labels in val_bar:
            with torch.no_grad():
                outputs = model(datas.float().to(device))
            loss = criterion(outputs, labels.type(torch.LongTensor).to(device))

            acc = (outputs.argmax(dim=-1) == labels.to(device)).float().mean()
            # Record the loss and accuracy
            valid_loss.append(loss.item())
            valid_acc.append(acc)
            val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        print(f"[{epoch + 1:02d}/{epochs:02d}] train loss = "
              f"{sum(train_loss) / len(train_loss):.5f}, train acc = {sum(train_acc) / len(train_acc):.5f}", end="  ")
        print(f"valid loss = {sum(valid_loss) / len(valid_loss):.5f}, valid acc = {sum(valid_acc) / len(valid_acc):.5f}")


if __name__ == '__main__':
    train()

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import DSCTransformer
from tqdm import tqdm
from data_set import MyDataset
from torch.utils.data import DataLoader

def train(model_save_path, train_result_path, val_result_path, hp_save_path, epochs=100):
    #定义参数
    N = 4 #编码器个数
    input_dim = 1024
    seq_len = 16 #句子长度
    d_model = 64 #词嵌入维度
    d_ff = 256 #全连接层维度
    head = 4 #注意力头数
    dropout = 0.1
    lr = 3E-5 #学习率
    batch_size = 64

    #保存超参数
    hyper_parameters = {'任务编码器堆叠数: ': '{}'.format(N),
                        '全连接层维度: ': '{}'.format(d_ff),
                        '任务注意力头数: ': '{}'.format(head),
                        'dropout: ': '{}'.format(dropout),
                        '学习率: ': '{}'.format(lr),
                        'batch_size: ': '{}'.format(batch_size)}
    fs = open(hp_save_path, 'w')
    fs.write(str(hyper_parameters))
    fs.close()

    #加载数据
    # train_path = r'F:\PyCharmWorkSpace\BearingFaultDiagnosis\data\train\train.csv'
    # val_path = r'F:\PyCharmWorkSpace\BearingFaultDiagnosis\data\val\val.csv'
    train_path = r'F:\PyCharmWorkSpace\MultiFD\data\own_data_noisy\-5db\train\train.csv'
    val_path = r'F:\PyCharmWorkSpace\MultiFD\data\own_data_noisy\-5db\val\val.csv'
    # train_path = r'F:\PyCharmWorkSpace\MultiFD\data\own_data\train\train.csv'
    # val_path = r'F:\PyCharmWorkSpace\MultiFD\data\own_data\val\val.csv'
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

    best_acc_fd = 0.0

    train_result = []
    result_train_loss = []
    result_train_acc = []
    val_result = []
    result_val_loss = []
    result_val_acc = []
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

        result_train_loss.append(sum(train_loss) / len(train_loss))
        result_train_acc.append((sum(train_acc) / len(train_acc)).item())
        result_val_loss.append(sum(valid_loss) / len(valid_loss))
        result_val_acc.append((sum(valid_acc) / len(valid_acc)).item())

        if best_acc_fd <= sum(valid_acc) / len(valid_acc):
            best_acc_fd = sum(valid_acc) / len(valid_acc)
            torch.save(model.state_dict(), model_save_path)

    train_result.append(result_train_loss)
    train_result.append(result_train_acc)
    val_result.append(result_val_loss)
    val_result.append(result_val_acc)

    np.savetxt(train_result_path, np.array(train_result), fmt='%.5f', delimiter=',')
    np.savetxt(val_result_path, np.array(val_result), fmt='%.5f', delimiter=',')

if __name__ == '__main__':
    group_index = 4
    for i in range(5):
        # model_save_path = "result/result_cu/group{}/exp0{}/model.pth".format(group_index, i+1)
        # hp_save_path = "result/result_cu/group{}/parameters.txt".format(group_index)
        # train_result_path = "result/result_cu/group{}/exp0{}/train_result.txt".format(group_index, i+1)
        # val_result_path = "result/result_cu/group{}/exp0{}/val_result.txt".format(group_index, i+1)
        # train(model_save_path, train_result_path, val_result_path, hp_save_path)

        model_save_path = "result/result_own_noisy/group{}/exp0{}/model.pth".format(group_index, i + 1)
        hp_save_path = "result/result_own_noisy/group{}/parameters.txt".format(group_index)
        train_result_path = "result/result_own_noisy/group{}/exp0{}/train_result.txt".format(group_index, i + 1)
        val_result_path = "result/result_own_noisy/group{}/exp0{}/val_result.txt".format(group_index, i + 1)
        train(model_save_path, train_result_path, val_result_path, hp_save_path)

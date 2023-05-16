import torch
import os
import numpy as np
from model import DSCTransformer
from tqdm import tqdm
from data_set import MyDataset
from torch.utils.data import DataLoader
from einops import rearrange

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def tsne_data(tsne_list, labels, save_path):
    for i in range(len(tsne_list)):
        tsne_list[i][0] = tsne_list[i][0][:, :, 0]
        tsne_list[i][0] = rearrange(tsne_list[i][0], 'b c l -> b (l c)')
        tsne_list[i][0] = tsne_list[i][0].cpu().numpy()
    global1_tsne = tsne_list[0][0]
    global2_tsne = tsne_list[1][0]
    global3_tsne = tsne_list[2][0]
    global4_tsne = tsne_list[3][0]
    fd1_tsne = tsne_list[4][0]
    fd2_tsne = tsne_list[5][0]
    fd3_tsne = tsne_list[6][0]
    fd4_tsne = tsne_list[7][0]
    loc1_tsne = tsne_list[8][0]
    loc2_tsne = tsne_list[9][0]
    loc3_tsne = tsne_list[10][0]
    loc4_tsne = tsne_list[11][0]
    dia1_tsne = tsne_list[12][0]
    dia2_tsne = tsne_list[13][0]
    dia3_tsne = tsne_list[14][0]
    dia4_tsne = tsne_list[15][0]

    np.savetxt(save_path + 'global1_tsne.txt', global1_tsne, fmt='%.5f', delimiter=',')
    np.savetxt(save_path + 'global2_tsne.txt', global2_tsne, fmt='%.5f', delimiter=',')
    np.savetxt(save_path + 'global3_tsne.txt', global3_tsne, fmt='%.5f', delimiter=',')
    np.savetxt(save_path + 'global4_tsne.txt', global4_tsne, fmt='%.5f', delimiter=',')
    np.savetxt(save_path + 'fd1_tsne.txt', fd1_tsne, fmt='%.5f', delimiter=',')
    np.savetxt(save_path + 'fd2_tsne.txt', fd2_tsne, fmt='%.5f', delimiter=',')
    np.savetxt(save_path + 'fd3_tsne.txt', fd3_tsne, fmt='%.5f', delimiter=',')
    np.savetxt(save_path + 'fd4_tsne.txt', fd4_tsne, fmt='%.5f', delimiter=',')
    np.savetxt(save_path + 'loc1_tsne.txt', loc1_tsne, fmt='%.5f', delimiter=',')
    np.savetxt(save_path + 'loc2_tsne.txt', loc2_tsne, fmt='%.5f', delimiter=',')
    np.savetxt(save_path + 'loc3_tsne.txt', loc3_tsne, fmt='%.5f', delimiter=',')
    np.savetxt(save_path + 'loc4_tsne.txt', loc4_tsne, fmt='%.5f', delimiter=',')
    np.savetxt(save_path + 'dia1_tsne.txt', dia1_tsne, fmt='%.5f', delimiter=',')
    np.savetxt(save_path + 'dia2_tsne.txt', dia2_tsne, fmt='%.5f', delimiter=',')
    np.savetxt(save_path + 'dia3_tsne.txt', dia3_tsne, fmt='%.5f', delimiter=',')
    np.savetxt(save_path + 'dia4_tsne.txt', dia4_tsne, fmt='%.5f', delimiter=',')
    np.savetxt(save_path + 'labels_tsne.txt', labels.cpu().numpy(), fmt='%.5f', delimiter=',')

def prediction(weights_path, con_matrix_path, acc_path, tsne_save_path):
    #定义参数
    N = 4 #编码器个数
    input_dim = 1024
    seq_len = 16 #句子长度
    d_model = 64 #词嵌入维度
    d_ff = 256 #全连接层维度
    head = 4 #注意力头数
    dropout = 0.1
    batch_size = 64

    test_path = r'F:\PyCharmWorkSpace\MultiFD\data\own_data_noisy\-5db\test\test.csv'
    test_dataset = MyDataset(test_path, 'fd')
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    #定义模型
    model = DSCTransformer(input_dim=input_dim, num_classes=10, dim=d_model, depth=N,
                           heads=head, mlp_dim=d_ff, dim_head=d_model, emb_dropout=dropout, dropout=dropout)
    model.to(device)

    # load model weights
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    model.eval()
    acc_fd = []
    con_matrix = torch.zeros(7, 7).type(torch.LongTensor).to(device) # 自己的数据是7分类
    test_bar = tqdm(test_loader)
    for datas, labels in test_bar:
        datas, labels = datas.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(datas.float().to(device))

            #tsne_data(tsne_list, labels, tsne_save_path)

        acc_fd.append((outputs.argmax(dim=-1) == labels).float().mean())
        #生成混淆矩阵
        for i in range(labels.size(0)):
            vect = torch.zeros(1, 7).type(torch.LongTensor).to(device) # 自己的数据是7分类
            temp = torch.max(outputs, dim=1)[1]
            vect[0, temp[i]] = 1
            index = labels[i].item()
            con_matrix[int(index)] = con_matrix[int(index)] + vect

    print(f'Test acc_fd = {(sum(acc_fd) / len(acc_fd)).item():.5f}')
    np.savetxt(con_matrix_path, con_matrix.cpu(), fmt='%d', delimiter=',')
    fs = open(acc_path, 'w')
    fs.write(f'{(sum(acc_fd) / len(acc_fd) * 100.0).item():.5f}')
    fs.close()


if __name__ == '__main__':
    group_index = 4
    for i in range(5):
        weights_path = "result/result_own_noisy/group{}/exp0{}/model.pth".format(group_index, i + 1)
        con_matrix_path = "result/result_own_noisy/group{}/exp0{}/confusion_matrix.txt".format(group_index, i + 1)
        acc_path = "result/result_own_noisy/group{}/exp0{}/test_result.txt".format(group_index, i + 1)
        tsne_save_path = "result/result_own_noisy/group{}/exp0{}/".format(group_index, i + 1)
        prediction(weights_path, con_matrix_path, acc_path, tsne_save_path)

    # weights_path = "result/result_own/group{}/exp0{}/model.pth".format(26, 1)
    # con_matrix_path = "result/result_own/group{}/exp0{}/confusion_matrix.txt".format(26, 1)
    # acc_path = "result/result_own/group{}/exp0{}/test_result.txt".format(26, 1)
    # tsne_save_path = "result/result_own/group{}/exp0{}/".format(26, 1)
    # prediction(weights_path, con_matrix_path, acc_path, tsne_save_path)
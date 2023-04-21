import numpy as np
import scipy.io as scio
import pandas as pd

def normalization(original_path,class_num):
    #读取mat文件数据
    matdata = scio.loadmat(original_path)
    #取出对应传感器数据
    for key in matdata.keys():
        if key.endswith('_DE_time'):
            c1 = matdata[key]
    # 计算均值与标准差并进行归一化
    c1_mean = np.mean(c1)
    c1_std = np.std(c1)
    c1 = (c1 - c1_mean) / c1_std
    #进行转置,变成行向量
    c1 = np.transpose(c1)
    #初始化一个data矩阵,存储200个样本,每个样本1024个数据点
    data = np.zeros([200,1024])
    #对data矩阵进行赋值
    j = 0
    for i in range(200):
        data[i] = c1[:,j:j+1024]
        j += 512
    #将样本数据进行存储
    csv_pd = pd.DataFrame(data)
    #csv_pd.to_csv(r'F:\PyCharmWorkSpace\BearingFaultDiagnosis\data\diameter\Processed\{}\{}.csv'.format(class_num,class_num), sep=',', header=False, index=False)
    return data

def makeSample(paths):
    datas = []
    for i in range(10):
        label = np.zeros([200,3])
        if i == 1:
            label += 1
        elif i == 2:
            label[:,0] += 2
            label[:,1] += 1
            label[:,2] += 2
        elif i == 3:
            label[:, 0] += 3
            label[:, 1] += 1
            label[:, 2] += 3
        elif i == 4:
            label[:, 0] += 4
            label[:, 1] += 2
            label[:, 2] += 1
        elif i == 5:
            label[:, 0] += 5
            label[:, 1] += 2
            label[:, 2] += 2
        elif i == 6:
            label[:, 0] += 6
            label[:, 1] += 2
            label[:, 2] += 3
        elif i == 7:
            label[:, 0] += 7
            label[:, 1] += 3
            label[:, 2] += 1
        elif i == 8:
            label[:, 0] += 8
            label[:, 1] += 3
            label[:, 2] += 2
        elif i == 9:
            label[:, 0] += 9
            label[:, 1] += 3
            label[:, 2] += 3
        temp = normalization(path[i],i)
        datas.append(np.hstack((temp,label)))

    train_arr, val_arr, test_arr = [], [], []
    for data in datas:
        np.random.shuffle(data)
        train_arr.append(data[0:120, :])
        val_arr.append(data[120:160, :])
        test_arr.append(data[160:, :])

    train_datas = train_arr[0]
    val_datas = val_arr[0]
    test_datas = test_arr[0]
    for i in range(1, 10):
        train_datas = np.vstack((train_datas, train_arr[i]))
        val_datas = np.vstack((val_datas, val_arr[i]))
        test_datas = np.vstack((test_datas, test_arr[i]))

    csv_pd = pd.DataFrame(train_datas)
    csv_pd.to_csv(r'F:\PyCharmWorkSpace\BearingFaultDiagnosis\data\train\train.csv', sep=',',
                  header=False, index=False)
    csv_pd = pd.DataFrame(val_datas)
    csv_pd.to_csv(r'F:\PyCharmWorkSpace\BearingFaultDiagnosis\data\val\val.csv', sep=',',
                  header=False, index=False)
    csv_pd = pd.DataFrame(test_datas)
    csv_pd.to_csv(r'F:\PyCharmWorkSpace\BearingFaultDiagnosis\data\test\test.csv', sep=',',
                  header=False, index=False)

if __name__ == '__main__':
    path = []
    path.append(r'F:\PyCharmWorkSpace\BearingFaultDiagnosis\data\Original\0HP\normal_0_97.mat')
    path.append(r'F:\PyCharmWorkSpace\BearingFaultDiagnosis\data\Original\0HP\12k_Drive_End_B007_0_118.mat')
    path.append(r'F:\PyCharmWorkSpace\BearingFaultDiagnosis\data\Original\0HP\12k_Drive_End_B014_0_185.mat')
    path.append(r'F:\PyCharmWorkSpace\BearingFaultDiagnosis\data\Original\0HP\12k_Drive_End_B021_0_222.mat')
    path.append(r'F:\PyCharmWorkSpace\BearingFaultDiagnosis\data\Original\0HP\12k_Drive_End_IR007_0_105.mat')
    path.append(r'F:\PyCharmWorkSpace\BearingFaultDiagnosis\data\Original\0HP\12k_Drive_End_IR014_0_169.mat')
    path.append(r'F:\PyCharmWorkSpace\BearingFaultDiagnosis\data\Original\0HP\12k_Drive_End_IR021_0_209.mat')
    path.append(r'F:\PyCharmWorkSpace\BearingFaultDiagnosis\data\Original\0HP\12k_Drive_End_OR007@6_0_130.mat')
    path.append(r'F:\PyCharmWorkSpace\BearingFaultDiagnosis\data\Original\0HP\12k_Drive_End_OR014@6_0_197.mat')
    path.append(r'F:\PyCharmWorkSpace\BearingFaultDiagnosis\data\Original\0HP\12k_Drive_End_OR021@6_0_234.mat')

    makeSample(path)





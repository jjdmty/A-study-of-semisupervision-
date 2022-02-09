"""
we got a set of number size 330 
10 labeled   320 unlabelled 
find out the most insure 5 numbers in the dataset and label it by human 
now 15 labelled  315 unlabeled 
after several iterations, the labelled numbers has increased.
"""
# Using Label Propagation Algorithm

# a study on sklearn dataset: hand written numbers 

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import datasets
# there are several datasets in sklearn
from sklearn.semi_supervised import _label_propagation
from sklearn.metrics import classification_report,confusion_matrix
from scipy.sparse.csgraph import *


digits = datasets.load_digits()
# 1797 samples, each contains 64 elements, related to a 8*8matrix of grey scale. target:0-9
# 共有1797个样本，每个样本有64的元素，对应到一个8x8像素点组成的矩阵，每一个值是其灰度值， target值是0-9，适用于分类任务。
# https://zhuanlan.zhihu.com/p/108393576 
rng = np.random.RandomState(0) # Random Number Generation
# print("rng",rng)


# random produce a set of 0-1796 numbers and mix up 
index = np.arange(len(digits.data))

rng.shuffle(index)
print(index)
# indices是随机产生的0-1796个数字，且打乱
# indices:[1081 1707  927 ... 1653  559  684]
print(len(index)) #1797

# 取前130个数字来玩
# take the first 130 
x = digits.data[index[:130]]
y = digits.target[index[:130]]
images = digits.images[index[:130]]

print(len(x))

n_total_samples = len(y) # 130
n_labeled_points = 10 # 标注好的数据共10条
max_iterations = 3 # 迭代3次

# unlabeled point 120
# index [10,11,12 ... 129]
unlabeled_indices = np.arange(n_total_samples)[n_labeled_points:]  # step 1
# np arange  https://blog.csdn.net/qq_41800366/article/details/86589680 

f = plt.figure() # 画图用的

for i in range(max_iterations):
    # if every element was labeled, break
    if len(unlabeled_indices)==0:
        print("no unlabeled items left to label") # 没有未标记的标签了，全部标注好了
        break 
    y_train = np.copy(y) # deep copy 
    y_train[unlabeled_indices] = -1 #把未标注的数据全部标记为-1，也就是后320条数据
    # set the rest unlabeled 320 numbers as -1, and regard as non labeled 

# Label Propagation Algorithm
    lp_model = _label_propagation.LabelSpreading(gamma=0.25,max_iter=3) # 训练模型
    lp_model.fit(x,y_train)

    predicted_labels = lp_model.transduction_[unlabeled_indices] # 预测的标签
    true_labels = y[unlabeled_indices] # 真实的标签

    # print('**************************')
    # print(predicted_labels)
    # print(true_labels)
    # print('**************************')
    cm = confusion_matrix(true_labels,predicted_labels,
                         labels = lp_model.classes_)
    print("iteration %i %s" % (i,70 * "_")) # 打印迭代次数
    print("Label Spreading model: %d labeled & %d unlabeled (%d total)" % (n_labeled_points,n_total_samples-n_labeled_points,n_total_samples))
    print(classification_report(true_labels,predicted_labels))
    print("Confusion matrix")
    print(cm)

    # 计算转换标签分布的熵
    # lp_model.label_distributions_作用是Categorical distribution for each item
    pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)
    # 选择分类器最不确定的前5位数字的索引
    # 首先计算出所有的熵，也就是不确定性，然后从320个中选择出前5个熵最大的
    # numpy.argsort(A)提取排序后各元素在原来数组中的索引。具体情况可看下面
    #  np.in1d 用于测试一个数组中的值在另一个数组中的成员资格，返回一个布尔型数组。具体情况可看下面
    uncertainty_index = np.argsort(pred_entropies)[::1]
    uncertainty_index = uncertainty_index[
        np.in1d(uncertainty_index,unlabeled_indices)][:5] # 这边可以确定每次选前几个作为不确定的数，最终都会加回到训练集
    
    # 跟踪我们获得标签的索引
    delete_indices = np.array([])
    
    # 可视化前5次的结果
    if i < 5:
        f.text(.05,(1 - (i + 1) * .183),
              'model %d\n\nfit with\n%d labels' %
              ((i + 1),i*5+10),size=10)
    for index,image_index in enumerate(uncertainty_index):
        # image_index是前5个不确定标签
        # index就是0-4
        image = images[image_index]
 
        # 可视化前5次的结果
        if i < 5:
            sub = f.add_subplot(5,5,index + 1 + (5*i))
            sub.imshow(image,cmap=plt.cm.gray_r)
            sub.set_title("predict:%i\ntrue: %i" % (
                lp_model.transduction_[image_index],y[image_index]),size=10)
            sub.axis('off')
        
        # 从320条里删除要那5个不确定的点
        # np.where里面的参数是条件，返回的是满足条件的索引
        delete_index, = np.where(unlabeled_indices == image_index)
        delete_indices = np.concatenate((delete_indices,delete_index))
        
    unlabeled_indices = np.delete(unlabeled_indices,delete_indices)
    # n_labeled_points是前面不确定的点有多少个被标注了
    n_labeled_points += len(uncertainty_index)
    
f.suptitle("Active learning with label propagation.\nRows show 5 most"
          "uncertain labels to learn with the next model")
plt.subplots_adjust(0.12,0.03,0.9,0.8,0.2,0.45)
plt.show()
    
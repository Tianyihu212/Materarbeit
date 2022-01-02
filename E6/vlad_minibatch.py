import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from data.dataset import GLDDataset 
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from data.dataset_contra import GLDDataset 
# from efficientnet_pytorch.model_e3 import EfficientNet
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
# from fast_pytorch_kmeans import KMeans
import os
import cv2
import itertools
import numpy as np
import pickle
# from sklearn.cluster import KMeans
from collections import Counter
from six.moves import cPickle
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax
from kmeans_pytorch import kmeans, kmeans_predict

os.environ['CUDA_VISIBLE_DEVICES']='0'
model = EfficientNet.from_pretrained('efficientnet-b0',num_classes=10)
model_path = 'model_e4_mAP_0.321865.pth'
print('Loading model from ', model_path)
checkpoint = torch.load(model_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.cuda()
model.eval()
index_feat_list = []
test_feat_list = []
num_workers = 6


import csv
from itertools import islice
from collections import defaultdict
from collections import Counter
import numpy as np
import torch.nn.functional as F
import cv2 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np




def count_distance(vector_1, vector_2, distance_type):
    '''
    :param vector_1:
    :param vector_2:
    :param distance_type: 'L1' Manhattan distance or 'L2' Euclidean distance

    Counting a distance between two vectors
    '''
    # Manhattan distance
    if distance_type == 'L1':
        return np.sum(np.absolute(vector_1 - vector_2))

    # Euclidean distance
    elif distance_type == 'L2':
        return np.sum((vector_1 - vector_2)**2)

def compute_vlad_descriptor(descriptors, cluster_centers):
    '''
    :param descriptor: SIFT descriptor of image
    :param kmeans_clusters: Object of Kmeans (sklearn)

    First we need to predict clusters fot key-points of image (row in
    input descriptor). Then for each cluster we get descriptors, which belong to it,
    and calculate sum of residuals between descriptor and centroid (cluster center)
    '''
    # Get SIFT dimension (default: 128)
    sift_dim = descriptors.shape[1]
    descriptors = torch.FloatTensor(descriptors).cuda()
    # Predict clusters for each key-point of image
#         labels_pred = kmeans_clusters.predict(descriptors)
#     labels_pred = kmeans_predict(descriptors)
    labels_pred = kmeans_predict(
    descriptors, cluster_centers, 'euclidean', device=device
)
#         kmeans_labels = kmeans.fit_predict(sift_descriptors)
    labels_pred = labels_pred.detach().cpu().numpy()
#     kmeans_clusters = kmeans.centroids
    kmeans_clusters = cluster_centers.detach().cpu().numpy()
    # Get centers fot each cluster and number of clusters
#         centers_cluster =  kmeans_clusters.cluster_centers_
    centers_cluster = kmeans_clusters
#         numb_cluster = kmeans_clusters.n_clusters
    numb_cluster = kmeans_clusters.shape[0]
    vlad_descriptors = np.zeros([numb_cluster, sift_dim])
    descriptors = descriptors.detach().cpu().numpy()
    # Compute the sum of residuals (for belonging x for cluster) for each cluster
    for i in range(numb_cluster):
        if np.sum(labels_pred == i) > 0:

            # Get descritors which belongs to cluster and compute residuals between x and centroids
            x_belongs_cluster = descriptors[labels_pred == i, :]
            vlad_descriptors[i] = np.sum(x_belongs_cluster - centers_cluster[i], axis=0)

    # Create vector from matrix
    vlad_descriptors = vlad_descriptors.flatten()

    # Power and L2 normalization
    vlad_descriptors = np.sign(vlad_descriptors) * (np.abs(vlad_descriptors)**(0.5))
    vlad_descriptors = vlad_descriptors / np.sqrt(vlad_descriptors @ vlad_descriptors)
    return vlad_descriptors
def map_at_k(y_true, y_denominator, y_pred):
    """
    y_true : ndarray of shape  (n_test_image, n_true_label)
        The true label of query images.  正确的标签
    y_denominator : ndarray of shape   (n_test_image)
    这个类别在index数据集中有几张图像
    y_pred : ndarray of shape  (n_test_image, top_k)
        The predicted label of query images. 预测结果
    """
    # Check format of input
    assert isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)
    assert y_true.ndim == 2 and y_pred.ndim == 2

    k = y_pred.shape[1]
    is_correct_list = []

    # In case of multiple true labels, check correctness of each label
    # Then use np.logical_or to conbine
    for i in range(y_true.shape[1]):
        is_correct = y_true[:, i][:, np.newaxis] == y_pred
        is_correct_list.append(is_correct)
    is_correct_mat = np.logical_or.reduce(np.array(is_correct_list))

    # Compute map
    cumsum_mat = np.apply_along_axis(np.cumsum, axis=1, arr=is_correct_mat)
    arange_mat = np.expand_dims(np.arange(1, k + 1), axis=0)
    ap_100_list = np.sum((cumsum_mat / arange_mat) * is_correct_mat, axis=1) / y_denominator

    return np.mean(ap_100_list), ap_100_list
index_list_csv = 'index_final.csv'
test_list_csv = 'test_final.csv'
index_list = []
index_class_list = []
test_list = []
test_class_list = []
with open(index_list_csv)as f:
    f_csv = csv.reader(f)
    for row in islice(f_csv, 1, None):
        label = int(row[0])
        img_id = row[1].split('\\')[-1].split('.')[0]
        # img_path = os.path.join(, img_name[0], img_name[1], img_name[2], img_name)
        index_list.append((img_id, label))
        index_class_list.append(label)
index_class_counter = Counter(index_class_list)

max_len_test = 0
with open(test_list_csv)as f:
    f_csv = csv.reader(f)
    for row in islice(f_csv, 1, None):
        label = row[0]
        label = [int(x) for x in label.split()]
        img_id = row[1].split('\\')[-1].split('.')[0]
        # img_path = os.path.join(, img_name[0], img_name[1], img_name[2], img_name)
        max_len_test = len(label) if len(label)>max_len_test else max_len_test
        test_list.append((img_id, label))
        test_class_list.append(label)
print(max_len_test)

index_feats_all = torch.load('index_feats_e4.pt').cuda()
test_feats_all = torch.load('test_feats_e4.pt').cuda()


with torch.no_grad():
    # for i in range(test_feats_all.size(0)):
    index_feats_all = F.normalize(index_feats_all, dim=1)
    test_feats_all=  F.normalize(test_feats_all, dim=1)
    # torch.cosine_similarity(p1.reshape(-1), p2.reshape(-1), dim=0)
    similarity_matrix = torch.mm(test_feats_all, index_feats_all.transpose(0, 1))


with open('sift_des_1.pkl', 'rb') as file:
    descriptors = pickle.load(file)
with open('sift_des_1.pkl', 'rb') as file:
    descriptors = descriptors + pickle.load(file)
print(len(descriptors))
# descriptors = np.array(list(itertools.chain.from_iterable(descriptors)))

index_root = '../../data/train'
test_root = '../../test_1k_final'
score, idx = similarity_matrix.topk(100, dim=1, largest=True, sorted=True)
# print(values)
# print(idx)

sift = cv2.SIFT_create(contrastThreshold=0.0)
bf = cv2.BFMatcher(cv2.NORM_L2)
# ori_score = score.detach().cpu()
save_ori_score_path = 'ori_score.pt'
# torch.save(ori_score, save_ori_score_path)
ori_score = torch.load(save_ori_score_path)

save_idx_path = 'idx.pt'
# torch.save(idx_list, save_idx_path)
idx_list = torch.load(save_idx_path)
idx_list = idx_list.numpy() #1000, 100
new_idx_list = np.zeros_like(idx_list)
prtv_score_list = []


des_bs = 500
num_batch = len(descriptors)//des_bs 
device = torch.device('cuda:0')
# kmeans = KMeans(n_clusters=16, mode='euclidean', verbose=1)
# k-means
# cluster_ids_x, cluster_centers = kmeans(
#     X=batch_1, num_clusters=num_clusters, distance='euclidean', device=device
# )
num_clusters = 16
cluster_centers = []
for batch_id in range(num_batch):
    print(batch_id, num_batch)
    sift_descriptors = descriptors[batch_id*des_bs : (batch_id+1)*des_bs]
    sift_descriptors = np.array(list(itertools.chain.from_iterable(sift_descriptors)))
#         kmeans = KMeans(n_clusters=16, mode='euclidean', verbose=1)
    # kmeans_clusters = KMeans(n_clusters=k).fit(sift_descriptors)
    sift_descriptors = torch.FloatTensor(sift_descriptors).cuda()
#     kmeans.fit(sift_descriptors, centroids = kmeans.centroids)
    cluster_ids_x, cluster_centers = kmeans(
    X=sift_descriptors, num_clusters=num_clusters,
    cluster_centers = cluster_centers,
    distance='euclidean', device=device
)


#     prtv_score = torch.load('prtv_score.pt')


for test_id in range(1000): 
    print(test_id)
    index_ids = idx_list[test_id]
    test_img_id = test_list[test_id][0]
    test_img_name = test_img_id+'.jpg'
    test_img_path = os.path.join(test_root, test_img_name)
    test_img = cv2.imread(test_img_path)
    test_img = cv2.resize(test_img, (224,224))
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    # test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
    # sift = cv2.SIFT_create(contrastThreshold=0.0)
    kps, test_feat = sift.detectAndCompute(test_img, None)
#     test_feat = torch.FloatTensor(test_feat).cuda().unsqueeze(0).mean(1)
    index_feat_t100 = []
    t100_good  = defaultdict()


    test_vlad_descriptor = compute_vlad_descriptor(test_feat, cluster_centers)
    for k, idx in enumerate(index_ids):
        index_sift_descriptor = descriptors[idx]
        index_vlad_descriptor = compute_vlad_descriptor(index_sift_descriptor, cluster_centers)
#         vlad_score = count_distance(test_vlad_descriptor, index_vlad_descriptor, 'L2')
        vlad_score = cosine_similarity(test_vlad_descriptor.reshape(1, -1), index_vlad_descriptor.reshape(1, -1)).mean()
        t100_good[idx] = vlad_score + ori_score[test_id][k]

    idx_ori_list = dict(sorted(t100_good.items(), key=lambda x:x[1], reverse=True)[:100]).keys()
    new_idx_list[test_id] = list(idx_ori_list)

pri_idx_list = [350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999]
pub_idx_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349]

new_idx_list_pri = new_idx_list[np.array(pri_idx_list)]
new_idx_list_pub = new_idx_list[np.array(pub_idx_list)]
test_class_list_pri =[ test_class_list[x] for x in pri_idx_list]
test_class_list_pub = [ test_class_list[x] for x in pub_idx_list]
# score_list = score.detach().cpu().numpy()

index_class_list = np.array(index_class_list)
y_pred = index_class_list[new_idx_list_pri] #1000, 100
print(y_pred[0])
# print(index_class_counter)
# print(y_pred)
# y_pred_list = 
# y_pred = np.array(idx2class)
# print(idx_list)
y_denominator_list = []
index_class_counter = dict(index_class_counter)
# print(index_class_counter)
for item in  test_class_list_pri:
    cnt = 0
    for label in item:
        cnt += index_class_counter[label]
#     print(item, cnt)
    y_denominator_list.append(cnt)
y_denominator = np.array(y_denominator_list)

y_true_list = []
for item in test_class_list_pri:
    new_item = [x for x in item]
    k =  max_len_test - len(item)
    for y in range(k):
        new_item += [item[0]]
    y_true_list.append(new_item)
y_true = np.array(y_true_list)
mAP, ap_list = map_at_k(y_true, y_denominator, y_pred)

# gather the stats from all processes
# metric_logger.synchronize_between_processes()
# print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
#       .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
print('pri * *mAP@100:', mAP)

index_class_list = np.array(index_class_list)
y_pred = index_class_list[new_idx_list_pub] #1000, 100
print(y_pred[0])
# print(index_class_counter)
# print(y_pred)
# y_pred_list = 
# y_pred = np.array(idx2class)
# print(idx_list)
y_denominator_list = []
index_class_counter = dict(index_class_counter)
# print(index_class_counter)
for item in  test_class_list_pub:
    cnt = 0
    for label in item:
        cnt += index_class_counter[label]
#     print(item, cnt)
    y_denominator_list.append(cnt)
y_denominator = np.array(y_denominator_list)

y_true_list = []
for item in test_class_list_pub:
    new_item = [x for x in item]
    k =  max_len_test - len(item)
    for y in range(k):
        new_item += [item[0]]
    y_true_list.append(new_item)
y_true = np.array(y_true_list)
mAP, ap_list = map_at_k(y_true, y_denominator, y_pred)

# gather the stats from all processes
# metric_logger.synchronize_between_processes()
# print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
#       .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
print('pub * *mAP@100:', mAP)
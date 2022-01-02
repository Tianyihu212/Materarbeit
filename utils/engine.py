
import torch
import torch.nn.functional as F
import utils
from utils import accuracy
import csv
import os
from itertools import islice
from collections import Counter
import numpy as np

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

def train_one_epoch(model, criterion, data_loader, optimizer, epoch, max_epoch):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]/[{}]'.format(epoch, max_epoch - 1)
    print_freq = 20

    for batch in metric_logger.log_every(data_loader, print_freq, header):

        images = batch['img'].cuda()
        labels = batch['label'].cuda()
        logits = model(images)
        loss = criterion(logits, labels)
#         print(loss.detach().cpu().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
        metric_logger.update(loss=loss.item())

    print("stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch_contra(model, criterion, criterion_contra, data_loader, optimizer, epoch, max_epoch):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]/[{}]'.format(epoch, max_epoch - 1)
    print_freq = 20

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        images1 = batch['img'].cuda()
        labels1 = batch['label'].cuda()
        images2 = batch['img2'].cuda()
        labels2 = batch['label2'].cuda()

        images = torch.cat([images1, images2], 0)
        labels = torch.cat([labels1, labels2], 0)
        flag = batch['flag'].cuda()

        bs = images1.size(0)
        logits, feats = model(images)
        feats1 =  feats[:bs]
        feats2 =  feats[bs:]
        loss_cls = criterion(logits, labels)
        loss_contra = criterion_contra(feats1, feats2, flag)
        loss = loss_cls *0.5  + loss_contra *0.5
#         print(loss.detach().cpu().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_contra=loss_contra.item())
        metric_logger.update(loss_cls=loss_cls.item())
    print("stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch_contra_only(model, criterion_contra, data_loader, optimizer, epoch, max_epoch):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]/[{}]'.format(epoch, max_epoch - 1)
    print_freq = 20

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        images1 = batch['img'].cuda()
        labels1 = batch['label'].cuda()
        images2 = batch['img2'].cuda()
        labels2 = batch['label2'].cuda()

        images = torch.cat([images1, images2], 0)
        labels = torch.cat([labels1, labels2], 0)
        flag = batch['flag'].cuda()

        bs = images1.size(0)
        logits, feats = model(images)
        feats1 =  feats[:bs]
        feats2 =  feats[bs:]
        # loss_cls = criterion(logits, labels)
        loss = criterion_contra(feats1, feats2, flag)
        # loss = loss_cls   + loss_contra 
#         print(loss.detach().cpu().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item())
        # metric_logger.update(loss_contra=loss_contra.item())
        # metric_logger.update(loss_cls=loss_cls.item())
    print("stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate(data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, 100, header):
            images = batch['img'].cuda()
            labels = batch['label'].cuda()
            # compute output
            output = model(images)
            if type(output) == tuple:
                logits = output[0]
            else:
                logits = output
            loss = criterion(logits, labels)
            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))

            batch_size = images.size(0)
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def test_retrieval(index_dataloader, test_dataloader, model):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    index_feat_list = []
    test_feat_list = []
    with torch.no_grad():
        # for epoch in range(args.start_epoch, args.epochs):
        for i_batch, data in enumerate(index_dataloader):
            print('{}/{}'.format(i_batch, len(index_dataloader)))
            img = data['img'].cuda()
            # label = data['label']
            feat = model.extract_features(img).view(img.size(0), 1280,-1).mean(2)
            index_feat_list.append(feat.detach().cpu())
        for i_batch, data in enumerate(test_dataloader):
            print('{}/{}'.format(i_batch, len(test_dataloader)))
            img = data['img'].cuda()
            feat = model.extract_features(img).view(img.size(0), 1280,-1).mean(2)
            test_feat_list.append(feat.detach().cpu())
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
    # index_feats_all = torch.load('index_feats_pretrain_mean_e3.pt').cuda()
    # test_feats_all = torch.load('test_feats_pretrain_mean_e3.pt').cuda()
    index_feats_all = torch.cat(index_feat_list, 0).cuda()
    test_feats_all = torch.cat(test_feat_list, 0).cuda()
    # print(index_feats_all.size(), test_feats_all.size())

    with torch.no_grad():
        # for i in range(test_feats_all.size(0)):
        index_feats_all = F.normalize(index_feats_all, dim=1)
        test_feats_all=  F.normalize(test_feats_all, dim=1)
        # torch.cosine_similarity(p1.reshape(-1), p2.reshape(-1), dim=0)
        similarity_matrix = torch.mm(test_feats_all, index_feats_all.transpose(0, 1))

    # print(similarity_matrix)

    score, idx = similarity_matrix.topk(100, dim=1, largest=True, sorted=True)

    idx_list = idx.detach().cpu().numpy()
    score_list = score.detach().cpu().numpy()
    index_class_list = np.array(index_class_list)
    y_pred = index_class_list[idx_list] #1000, 100
    # print(y_pred[0])
    y_denominator_list = []
    index_class_counter = dict(index_class_counter)
    # print(index_class_counter)
    for item in  test_class_list:
        cnt = 0
        for label in item:
            cnt += index_class_counter[label]
    #     print(item, cnt)
        y_denominator_list.append(cnt)
    y_denominator = np.array(y_denominator_list)

    y_true_list = []
    for item in test_class_list:
        new_item = [x for x in item]
        k =  max_len_test - len(item)
        for y in range(k):
            new_item += [item[0]]
        y_true_list.append(new_item)
    y_true = np.array(y_true_list)
    mAP, ap_list = map_at_k(y_true, y_denominator, y_pred)

    print('* *mAP@100:', mAP)
    return {'mAP': mAP}
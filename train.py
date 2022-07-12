import torch
from torch import distributed
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from tqdm import tqdm
import utils
from utils.loss import KnowledgeDistillationLoss, BCEWithLogitsLossWithIgnoreIndex, \
    UnbiasedKnowledgeDistillationLoss, UnbiasedCrossEntropy, IcarlLoss, FeaturesClusteringSeparationLoss, SNNL, \
    DistillationEncoderLoss, DistillationEncoderPrototypesLoss, FeaturesSparsificationLoss, \
    KnowledgeDistillationCELossWithGradientScaling, CAB_loss, CosineSimilarityloss
from utils import get_regularizer
import time
from PIL import Image
from utils.run_utils import *
import matplotlib.pyplot as plt

import numpy as np
# fuben

class Trainer:
    def __init__(self, model, model_old, device, opts, trainer_state=None, classes=None, logdir=None):

        self.model_old = model_old
        self.model = model
        self.device = device
        self.where_to_sim = opts.where_to_sim
        self.step = opts.step
        self.no_mask = opts.no_mask  # if True sequential dataset from https://arxiv.org/abs/1907.13372
        self.overlap = opts.overlap
        self.loss_de_prototypes_sumafter = opts.loss_de_prototypes_sumafter
        self.num_classes = sum(classes) if classes is not None else 0
        self.cab_loss = opts.cab_loss
        self.cab_loss_flag = opts.cab_loss and model_old is not None
        self.featupdate = opts.featupdate_loss
        self.featupdate_loss_flag = opts.featupdate_loss and model_old is not None
        self.featupdate_count_flag = opts.featupdate_count
        self.featupdate_count_last = opts.featupdate_last
        self.threshold_sum_flag = opts.threshold_sum
        self.cosine_similarity_loss_flag = opts.cosine_similarity_loss
        self.selfcosinepmap_prototypes_flag = opts.selfcosinepmap_prototypes
        self.bkgprototypefeature_flag = opts.bkgprototypefeature
        self.consinemap_flag = opts.consinemap

        if classes is not None:
            new_classes = classes[-1]
            tot_classes = reduce(lambda a, b: a + b, classes)
            self.current_classes = new_classes
            self.old_classes = tot_classes - new_classes
        else:
            self.old_classes = 0
        if self.cab_loss_flag:
            self.cab_loss_ = CAB_loss(2048, 2048, self.old_classes).cuda()

        reduction = 'none'
        self.cosine_similarity = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)
        self.bce = opts.bce or opts.icarl
        if self.bce:
            self.criterion = BCEWithLogitsLossWithIgnoreIndex(reduction=reduction)
        elif opts.unce and self.old_classes != 0:
            self.criterion = UnbiasedCrossEntropy(old_cl=self.old_classes, ignore_index=255, reduction=reduction)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)

        # features clustering loss
        self.lfc = opts.loss_fc
        self.lfc_flag = self.lfc > 0.
        # Separation between clustering loss
        self.lfc_sep_clust = opts.lfc_sep_clust
        #Contrastive Learning loss
        self.lfc_loss = FeaturesClusteringSeparationLoss(num_classes=sum(classes) if classes is not None else 0,
                                                         logdir=logdir if logdir is not None else '', feat_dim=2048,
                                                         device=self.device, lfc_L2normalized=opts.lfc_L2normalized,
                                                         lfc_nobgr=opts.lfc_nobgr, lfc_sep_clust=self.lfc_sep_clust,
                                                         lfc_sep_clust_ison_proto=opts.lfc_sep_clust_ison_proto,
                                                         orth_sep=opts.lfc_orth_sep, lfc_orth_maxonly=opts.lfc_orth_maxonly)

        # SNNL loss at features space
        self.lSNNL = opts.loss_SNNL
        self.lSNNL_flag = self.lSNNL > 0.
        if classes is not None and logdir is not None:
            self.lSNNL_loss = SNNL(num_classes=sum(classes), logdir=logdir, feat_dim=2048, device=self.device)

        # ILTSS paper loss: http://openaccess.thecvf.com/content_ICCVW_2019/papers/TASK-CV/Michieli_Incremental_Learning_Techniques_for_Semantic_Segmentation_ICCVW_2019_paper.pdf
        # https://arxiv.org/abs/1911.03462
        self.lde = opts.loss_de
        self.lde_flag = self.lde > 0. and model_old is not None
        self.lde_loss = DistillationEncoderLoss(mask=opts.loss_de_maskedold, loss_de_cosine=opts.loss_de_cosine)
        #Prototypes Matching loss
        self.ldeprototype = opts.loss_de_prototypes
        self.ldeprototype_flag = self.ldeprototype > 0.
        self.ldeprototype_loss = DistillationEncoderPrototypesLoss(num_classes=sum(classes) if classes is not None else 0,
                                                                   device=self.device)

        # CIL paper loss: https://arxiv.org/abs/2005.06050
        self.lCIL = opts.loss_CIL
        self.lCIL_flag = self.lCIL > 0. and model_old is not None
        self.lCIL_loss = KnowledgeDistillationCELossWithGradientScaling(temp=1, gs=self.lCIL, device=self.device, norm=False)

        # Features Sparsification Loss
        self.lfs = opts.loss_featspars
        self.lfs_flag = self.lfs > 0.
        self.lfs_loss = FeaturesSparsificationLoss(lfs_normalization=opts.lfs_normalization,
                                                   lfs_shrinkingfn=opts.lfs_shrinkingfn,
                                                   lfs_loss_fn_touse=opts.lfs_loss_fn_touse)

        self.lkd = opts.loss_kd
        self.lkd_flag = self.lkd > 0. and model_old is not None
        if opts.unkd:
            self.lkd_loss = UnbiasedKnowledgeDistillationLoss(alpha=opts.alpha)
        else:
            self.lkd_loss = KnowledgeDistillationLoss(alpha=opts.alpha)

        # ICARL
        self.icarl_combined = False
        self.icarl_only_dist = False
        if opts.icarl:
            self.icarl_combined = not opts.icarl_disjoint and model_old is not None
            self.icarl_only_dist = opts.icarl_disjoint and model_old is not None
            if self.icarl_combined:
                self.licarl = nn.BCEWithLogitsLoss(reduction='mean')
                self.icarl = opts.icarl_importance
            elif self.icarl_only_dist:
                self.licarl = IcarlLoss(reduction='mean', bkg=opts.icarl_bkg)
        self.icarl_dist_flag = self.icarl_only_dist or self.icarl_combined

        # Regularization
        regularizer_state = trainer_state['regularizer'] if trainer_state is not None else None
        self.regularizer = get_regularizer(model, model_old, device, opts, regularizer_state) # EWC PI RW
        self.regularizer_flag = self.regularizer is not None
        self.reg_importance = opts.reg_importance
        self.loss_bkgpaint = opts.bkgpaint
        self.update_background = opts.update_background
        self.ours_bkgpaint_flag = self.loss_bkgpaint > 0 and model_old is not None
        self.update_background_flag = self.update_background>0 and model_old is not None
        self.ret_intermediate = self.lde or self.lfc or self.lfc_sep_clust or self.lSNNL or self.ldeprototype or \
                                self.lfs or self.lCIL or self.cab_loss_flag or self.ours_bkgpaint_flag or self.featupdate_loss_flag




    def train(self, cur_epoch, optim, train_loader, world_size, scheduler=None, print_int=10, logger=None,
              prototypes=None, count_features=None, label2color=None, denorm=None, old_epoch_prototype=None, old_count_features=None):
        """Train and return epoch loss"""
        logger.info("Epoch %d, lr = %f" % (cur_epoch + 1, optim.param_groups[0]['lr']))

        device = self.device
        model = self.model
        criterion = self.criterion

        epoch_loss = 0.0
        reg_loss = 0.0
        interval_loss = 0.0
        lkd = torch.tensor(0.)
        lde = torch.tensor(0.)
        l_icarl = torch.tensor(0.)
        l_reg = torch.tensor(0.)
        lfc = torch.tensor(0.)
        lsep_clusters = torch.tensor(0.)
        lSNNL = torch.tensor(0.)
        ldeprototype = torch.tensor(0.)
        lfs = torch.tensor(0.)
        lCIL = torch.tensor(0.)
        cab_loss = torch.tensor(0.)
        cosine_loss = torch.tensor(0.)

        train_loader.sampler.set_epoch(cur_epoch)

        model.train()
        start_time = time.time()
        start_epoch_time = time.time()
        save_image = 0

        for cur_step, (images, labels) in enumerate(train_loader):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            label_tosave = label2color(labels.detach().cpu().numpy())[0].astype(np.uint8)
            if (self.lde_flag or self.lkd_flag or self.icarl_dist_flag or self.lfc_flag or self.lfc_sep_clust
                or self.lSNNL_flag or self.ldeprototype_flag or self.lCIL_flag or self.ours_bkgpaint_flag or self.cab_loss_flag or self.featupdate_loss_flag) \
                    and self.model_old is not None:
                with torch.no_grad():
                    # cosine_map_feat_ = None
                    outputs_old, features_old = self.model_old(images, ret_intermediate=self.ret_intermediate)

            # batch, feat, H, W = features_old['body'].size()
            # cosine_map = []
            # for bs in range(batch):
            #     Query = features_old['body'][bs, ...]
            #     prototypes_center = prototypes[...,None,None].repeat(1, 1, Query.size(1), Query.size(2))
            #     cosine_map = F.cosine_similarity(prototypes_center, Query.unsqueeze(0), dim=0, eps=1e-7)
            #     cosine_map = torch.max(cosine_map)(0)[1]
            #     sp_guide_feat = prototypes[:,cosine_map]
            # if self.selfcosinepmap_prototypes_flag==True or self.bkgprototypefeature_flag==True or self.consinemap_flag==True:
            #     cosine_map_feat_=1

            # if self.selfcosinepmap_prototypes_flag:
            #     batch, feat, H, W = features_old['body'].size()
            #     cosine_map_feat_ = []
            #     for bs in range(batch):
            #         Query = features_old['body'][bs, ...]  # 2048 * 32 * 32
            #         Query = Query.expand(self.old_classes, feat, H, W)#11*2048*32*32
            #         prototypes_tmp = prototypes[:self.old_classes,...].clone() # 11*32*32
            #         key = prototypes_tmp.expand(H, W, self.old_classes, feat).permute(2,3,0,1)#11*2048*32*32
            #         cosine_map_class = F.cosine_similarity(key, Query, dim=1, eps=1e-7) # old_classes * H * W 11*32*32
            #         guide_map = cosine_map_class.max(0)[1]  # 32*32
            #         cosine_map_feat = prototypes_tmp[guide_map, :]
            #         cosine_map_feat_.append(cosine_map_feat.unsqueeze(0).permute(0, 3, 1, 2))
            #     cosine_map_feat_ = torch.cat(cosine_map_feat_, dim=0)

            if self.bkgprototypefeature_flag:
                batch, feat, H, W = features_old['body'].size()
                guide_map= torch.argmax(torch.softmax(outputs_old, dim=1),dim=1)
                # guide_map = np.argmax(torch.softmax(outputs_old, dim=1).data.cpu().numpy(), axis=1)
                guide_map_down = (F.interpolate(input=guide_map.double().unsqueeze(0), size=(H,W),
                                  mode='nearest')).long()
                prototypes_tmp = prototypes[:self.old_classes, ...].clone()
                cosine_map_feat_ = prototypes_tmp[guide_map_down.squeeze(0), :].permute(0, 3, 1, 2).contiguous()
                # cosine_map_feat_.append(cosine_map_feat.unsqueeze(0).permute(0, 3, 1, 2))

            if self.consinemap_flag:
                batch, feat, H, W = features_old['body'].size()
                guide_map_feat_ = []
                for bs in range(batch):
                    Query = features_old['body'][bs, ...]  # 2048 * 32 * 32
                    Query = Query.expand(self.old_classes, feat, H, W)  # 11*2048*32*32
                    prototypes_tmp = prototypes[:self.old_classes, ...].clone()  # 11*32*32
                    key = prototypes_tmp.expand(H, W, self.old_classes, feat).permute(2, 3, 0, 1)  # 11*2048*32*32
                    cosine_map_class = F.cosine_similarity(key, Query, dim=1, eps=1e-7)  # old_classes * H * W 11*32*32
                    # guide_map = cosine_map_class.max(0)[1]  # 32*32
                    # cosine_map_feat = prototypes_tmp[guide_map, :]
                    guide_map_feat_.append(cosine_map_class.unsqueeze(0)) #1*32*32
                cosine_map_feat_ = torch.cat(guide_map_feat_, dim=0) #8*32*32
                # cosine_map_feat_ = guide_map_feat_.expand(2048,8,32,32).permute(1,0,2,3)



            optim.zero_grad()
            outputs, features = model(images,prototypes, ret_intermediate=self.ret_intermediate, cosinepmap_prototypes_flag=self.selfcosinepmap_prototypes_flag,old_classes=self.old_classes)


            # if save_image < 500:
            #     outputs_old_mask = torch.softmax(outputs, dim=1)
            #     outputs_old_mask = np.argmax(outputs_old_mask.data.cpu().numpy(), axis=1)
            #     save_image = save_image + 4
            #     image_name = train_loader.dataset.dataset.dataset.images[cur_step][0].split('/')[-1].split('.')[0]
            #     # name = name[0].split('/')[-1].split('.')[0]
            #     image_tosave = (denorm(images[0].detach().cpu().numpy()) * 255).astype(np.uint8).transpose(1, 2, 0)
            #     prediction_tosave = label2color(outputs_old_mask)[0].astype(np.uint8)
            #     label_tosave_update = label2color(labels.detach().cpu().numpy())[0].astype(np.uint8)
            #
            #     scale_pred = F.upsample(input=features['pre_logits'], size=(512, 512), mode='bilinear', align_corners=True)[0,...]
            #     scale_pred = torch.squeeze(scale_pred, 0)
            #     scale_pred = torch.max(scale_pred, dim=0)
            #     visual = scale_pred.cpu().detach().numpy()
            #     fig = plt.gcf()
            #     fig.set_size_inches(2, 2)
            #     plt.gca().xaxis.set_major_locator(plt.NullLocator())
            #     plt.gca().yaxis.set_major_locator(plt.NullLocator())
            #     plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            #     plt.margins(0, 0)
            #     plt.imshow(visual, cmap='jet')
            #     plt.axis('off')
            #     # plt.colorbar()
            #     plt.savefig("/workspace/SDR/logs/CAM_max/{}_{}_feature.png".format(
            #         image_name,cur_step), dpi=256.5)
            #     vis_dir = "/workspace/SDR/logs/CAM_max"
            #     # Image.fromarray(torch.squeeze(torch.from_numpy(prediction[0]),2)).save(f'{test_dir}/{image_name}.png')
            #     # Image.fromarray(image_tosave).save(f'{vis_dir}/{image_name}_{i}_RGB.jpg')
            #     # Image.fromarray(prediction_tosave).save(f'{vis_dir}/visual/{image_name}_{i}_pred.png')
            #     Image.fromarray(prediction_tosave).save(f'{vis_dir}/{image_name}_{cur_step}.png')
            #     Image.fromarray(image_tosave).save(f'{vis_dir}/{image_name}_{cur_step}_RGB.jpg')
            #     Image.fromarray(label_tosave).save(f'{vis_dir}/{image_name}_{cur_step}_label.png')
            #     Image.fromarray(label_tosave_update).save(f'{vis_dir}/{image_name}_{cur_step}_label_update.png')



            if self.lfc_flag or self.ldeprototype_flag or self.lfc_sep_clust or self.featupdate_loss_flag:
                # F.interpolate 上采样
                #--
                # a = labels.unsqueeze(dim=1).double()
                b = F.interpolate(input=labels.unsqueeze(dim=1).double(),
                                  size=(features['body'].shape[2], features['body'].shape[3]),
                                  mode='nearest')
                tmp_ = torch.tensor([0.]).to(device)
                # print(len(torch.unique(b)))
                # tmp = torch.unique(b)
                # tmp1 = torch.tensor([0.],dtype=torch.float64)
                # if tmp1 == [0]:
                #     pass
                if len(torch.unique(b)) == 1 :
                    if torch.unique(b) != tmp_:
                        # 这一步的目的是为了更新prototype，背景类的prototype没有被更新,只更新有真实标签类别的prototype
                        prototypes, count_features = self._update_running_stats((F.interpolate(
                        input=labels.unsqueeze(dim=1).double(), size=(features['body'].shape[2], features['body'].shape[3]),
                        mode='nearest')).long(), features['body'], self.no_mask, self.overlap, self.step, prototypes,
                                                                        count_features)
                    else:
                        print('torch.unique(b) == [0]')
                else:
                    prototypes, count_features = self._update_running_stats((F.interpolate(
                        input=labels.unsqueeze(dim=1).double(),
                        size=(features['body'].shape[2], features['body'].shape[3]),
                        mode='nearest')).long(), features['body'], self.no_mask, self.overlap, self.step, prototypes,
                                                                            count_features)



            if self.ours_bkgpaint_flag:
                outputs_old_mask = torch.argmax(torch.softmax(outputs_old, dim=1),dim=1)
                labels[labels==0] = outputs_old_mask[labels==0]

            if self.update_background_flag:
                prototypes, count_features = self.Update_background_prototype(
                    features=features['body'],
                    features_old=features_old[
                        'body'] if self.step != 0 else None,
                    labels=labels,
                    classes_old=self.old_classes,
                    incremental_step=self.step,
                    sequential=self.no_mask,
                    overlapped=self.overlap,
                    outputs_old=outputs_old if self.step != 0 else None,
                    outputs=outputs,
                    loss_de_prototypes_sumafter=self.loss_de_prototypes_sumafter,
                    prototypes=prototypes,
                    count_features=count_features,
                    old_epoch_prototype=old_epoch_prototype,
                    bkgpaint_flag=self.ours_bkgpaint_flag,
                    old_count_features=old_count_features,
                    thred=self.threshold_sum_flag)


            if self.cosine_similarity_loss_flag:

                # batch, feat, H, W = features['body'].size()
                # guide_map_feat_ = []
                # # cosine_map = cosine_map.cuda()
                # for bs in range(batch):
                #     Query = features['body'][bs, ...].clone()  # 2048 * 32 * 32
                #     Query = Query.expand(self.old_classes, feat, H, W)  # 11*2048*32*32
                #     prototypes_tmp = prototypes[:self.old_classes, ...].clone()  # 11*32*32
                #     key = prototypes_tmp.expand(H, W, self.old_classes, feat).permute(2, 3, 0, 1)  # 11*2048*32*32
                #     cosine_map_class = F.cosine_similarity(key, Query, dim=1, eps=1e-7)  # old_classes * H * W 11*32*32
                #     guide_map_feat_.append(cosine_map_class.unsqueeze(0)) #1*11*32*32
                # cosine_map_feat_ = torch.cat(guide_map_feat_, dim=0) #8*11*32*32
                # cosine_map = F.interpolate(cosine_map_feat_.double(), size=(512, 512), mode='nearest').squeeze(0)
                # # cosine_map = torch.sigmoid(cosine_map)
                # cosine_map = torch.softmax(cosine_map,dim=1)
                # cosine_loss = criterion(cosine_map, labels).mean()
                # ---1220 modify
                if self.model_old is not None:
                    # task 2+
                    batch, feat, H, W = features['body'].size()
                    cosine_map = []
                    for bs in range(batch):
                        cosine_map_list = []
                        Query = features['body'][bs, ...]  # 2048 * 32 * 32
                        for cls in range(self.current_classes + self.old_classes):
                            key = prototypes[cls]
                            key = key.expand(H, W, feat)  # 32 * 32 * 2048
                            key = key.permute(2, 0, 1)
                            cosine_map_class = F.cosine_similarity(key, Query, dim=0, eps=1e-7)
                            cosine_map_list.append(cosine_map_class.unsqueeze(0))
                        cosine_map.append(torch.cat(cosine_map_list, dim=0).unsqueeze(0))
                    cosine_map = torch.cat(cosine_map, dim=0)
                    cosine_map = F.interpolate(cosine_map.double(), size=(512, 512), mode='nearest').squeeze(0)
                    # cosine_map = torch.sigmoid(cosine_map)
                    cosine_map = torch.softmax(cosine_map, dim=1)
                    cosine_loss = criterion(cosine_map, labels).mean()
                    pass
                else:
                    # task 1
                    batch, feat, H, W = features['body'].size()
                    cosine_map = []
                    for bs in range(batch):
                        cosine_map_list = []
                        Query = features['body'][bs, ...]  # 2048 * 32 * 32
                        for cls in range(self.current_classes):
                            key = prototypes[cls]
                            key = key.expand(H, W, feat)  # 32 * 32 * 2048
                            key = key.permute(2, 0, 1)
                            cosine_map_class = F.cosine_similarity(key, Query, dim=0,eps=1e-7)
                            cosine_map_list.append(cosine_map_class.unsqueeze(0))
                        cosine_map.append(torch.cat(cosine_map_list, dim=0).unsqueeze(0))
                    cosine_map = torch.cat(cosine_map, dim=0)
                    cosine_map = F.interpolate(cosine_map.double(), size=(512, 512), mode='nearest').squeeze(0)
                    # cosine_map = torch.sigmoid(cosine_map)
                    cosine_map = torch.softmax(cosine_map, dim=1)
                    cosine_loss = criterion(cosine_map, labels).mean()
                # -- 1220 modify
                # print(cosine_loss)
                # if cur_epoch>=0:
                #     # cosine_map[cosine_map>0.9] = 1
                #     mask = (cosine_map < 0.9).sum(dim=1)
                #     cosine_map = torch.argmax(cosine_map, dim=1)
                #     cosine_map[mask == 0] = 0
                #     image_name = train_loader.dataset.dataset.dataset.images[8 * cur_step][0].split('/')[-1].split('.')[0]
                #     cosine_mask_save = label2color(cosine_map.cpu().int().numpy())[0].astype(np.uint8)
                #     image_tosave = (denorm(images[0].detach().cpu().numpy()) * 255).astype(np.uint8).transpose(1, 2, 0)
                #     outputs_old_mask = torch.softmax(outputs, dim=1)
                #     outputs_old_mask = np.argmax(outputs_old_mask.data.cpu().numpy(), axis=1)
                #     prediction_tosave = label2color(outputs_old_mask)[0].astype(np.uint8)
                #     # Image.fromarray(cosine_mask_save).save(f'/workspace/cosine_map/{image_name}_{cur_step}_cosine_mask.png')
                #     Image.fromarray(cosine_mask_save).save(f'/workspace/cosine_map/{image_name}_{cur_step}_cosine_mask.png')
                #     Image.fromarray(image_tosave).save(f'/workspace/cosine_map/{image_name}_{cur_step}_image_RGB.jpg')
                #     Image.fromarray(label_tosave).save(f'/workspace/cosine_map/{image_name}_{cur_step}_label.png')
                #     Image.fromarray(prediction_tosave).save(f'/workspace/cosine_map/{image_name}_{cur_step}_predection.png')


            # if self.ours_bkgpaint_flag:
            #     outputs_old_mask = torch.argmax(torch.softmax(outputs_old, dim=1),dim=1)
            #     labels[labels==0] = outputs_old_mask[labels==0]

                # outputs_old_mask = torch.softmax(outputs_old, dim=1)
                # outputs_old_mask = np.argmax(outputs_old_mask.data.cpu().numpy(), axis=1)
                # labels = labels.data.cpu().numpy()
                # labels[labels == 0] = outputs_old_mask[labels == 0]
                # labels = torch.from_numpy(labels).cuda()


                # loss_bkgpaint = criterion(outputs, labels_bkg)

                # 可视化代码
                # if save_image < 500:
                #     save_image = save_image + 4
                #     image_name = train_loader.dataset.dataset.dataset.images[cur_step][0].split('/')[-1].split('.')[0]
                #     # name = name[0].split('/')[-1].split('.')[0]
                #     image_tosave = (denorm(images[0].detach().cpu().numpy()) * 255).astype(np.uint8).transpose(1, 2, 0)
                #     prediction_tosave = label2color(outputs_old_mask)[0].astype(np.uint8)
                #     label_tosave_update = label2color(labels.detach().cpu().numpy())[0].astype(np.uint8)
                #
                #     # print(prediction.shape())
                #     vis_dir = "/workspace/SDR/outputs/cab"
                #     # Image.fromarray(torch.squeeze(torch.from_numpy(prediction[0]),2)).save(f'{test_dir}/{image_name}.png')
                #     # Image.fromarray(image_tosave).save(f'{vis_dir}/{image_name}_{i}_RGB.jpg')
                #     # Image.fromarray(prediction_tosave).save(f'{vis_dir}/visual/{image_name}_{i}_pred.png')
                #     Image.fromarray(prediction_tosave).save(f'{vis_dir}/cab/{image_name}_{cur_step}.png')
                #     Image.fromarray(image_tosave).save(f'{vis_dir}/cab/{image_name}_{cur_step}_RGB.jpg')
                #     Image.fromarray(label_tosave).save(f'{vis_dir}/cab/{image_name}_{cur_step}_label.png')
                #     Image.fromarray(label_tosave_update).save(f'{vis_dir}/cab/{image_name}_{cur_step}_label_update.png')

            # if self.cosine_map_flag:
            #     batch, feat, H, W = features_old['body'].size()
            #     cosine_map = torch.ones(batch,self.old_classes,H,W)
            #     cosine_map = cosine_map.cuda()
            #     for bs in range(batch):
            #         cosine_map_list = []
            #         Query = features_old['body'][bs,...] # 2048 * 32 * 32
            #         for cls in range(self.old_classes):
            #             key = prototypes[cls]
            #             key = key.expand(H,W,feat) #32 * 32 * 2048
            #             key = key.permute(2,0,1)
            #             cosine_map_class = F.cosine_similarity(key, Query, dim=0,eps=1e-7)
            #             cosine_map_list.append(cosine_map_class.unsqueeze(0))
            #         cosine_map[bs,...] = torch.cat(cosine_map_list,dim=0)
            #     # cosine_map batch * class_num * H * W
            #     cosine_map = torch.softmax(cosine_map, dim=1) # batch * old_class_num * H * W
            #     # difference = torch.softmax(outputs_old,dim=1) - F.interpolate(cosine_map.double(), size=(512, 512), mode='nearest')
            #     # diff_mask = 1 - torch.abs(difference)
            #     # mask = (diff_mask > 0.95).sum(dim=1)
            #     # cosine_map = torch.argmax(diff_mask, dim=1)
            #     # cosine_map[mask==0]=0
            #     # cosine_map[cosine_map>0]=0
            #     # cosine_map[labels > 10] = 0
            #     mask = (cosine_map>0.65).sum(dim=1)
            #     cosine_map = torch.argmax(cosine_map, dim=1)
            #     cosine_map[mask==0]=0
            #     cosine_map = F.interpolate(cosine_map.unsqueeze(0).double(), size=(512, 512), mode='nearest').squeeze(0)
            #     cosine_map[labels > 10] = 0



            #     #-----
            # if self.pairwise_distance_flag:
            #     batch, feat, H, W = features_old['body'].size()
            #     cosine_map = torch.ones(batch,self.old_classes,H,W)
            #     cosine_map = cosine_map.cuda()
            #     for bs in range(batch):
            #         cosine_map_list = []
            #         Query = features_old['body'][bs,...].view(-1, feat) # 2048 * 32 * 32
            #         for cls in range(self.old_classes):
            #             key = prototypes[cls]
            #             cosine_map_class = F.pairwise_distance(key, Query,p=2)
            #             cosine_map_list.append(cosine_map_class.view(32,32).unsqueeze(0))
            #         cosine_map[bs,...] = torch.cat(cosine_map_list,dim=0)
            #     # cosine_map batch * class_num * H * W
            #     cosine_map = torch.softmax(cosine_map, dim=1)
            #     cosine_map = torch.argmax(cosine_map, dim=1)
            #     # cosine_map[mask==0]=0
            #     cosine_map = F.interpolate(cosine_map.unsqueeze(0).double(), size=(512, 512), mode='nearest').squeeze(0)
            #     # cosine_map[labels > 10] = 0
            # ----------



                # image_name = train_loader.dataset.dataset.dataset.images[8*cur_step][0].split('/')[-1].split('.')[0]
                # cosine_mask_save = label2color(cosine_map.cpu().int().numpy())[0].astype(np.uint8)
                # image_tosave = (denorm(images[0].detach().cpu().numpy()) * 255).astype(np.uint8).transpose(1, 2, 0)
                # prediction_tosave = label2color(outputs_old_mask)[0].astype(np.uint8)
                # # Image.fromarray(cosine_mask_save).save(f'/workspace/cosine_map/{image_name}_{cur_step}_cosine_mask.png')
                # Image.fromarray(cosine_mask_save).save(f'/workspace/cosine_map/{image_name}_{cur_step}_cosine_mask.png')
                # Image.fromarray(image_tosave).save(f'/workspace/cosine_map/{image_name}_{cur_step}_image_RGB.jpg')
                # Image.fromarray(label_tosave).save(f'/workspace/cosine_map/{image_name}_{cur_step}_label.png')
                # Image.fromarray(prediction_tosave).save(f'/workspace/cosine_map/{image_name}_{cur_step}_predection.png')




            if self.cab_loss_flag:
                out = self.cab_loss_(prototypes.narrow(0,0,self.old_classes), outputs_old, features['body'])
                cab_loss = self.lkd_loss(out, outputs_old)
                outputs_old_mask = torch.softmax(outputs_old, dim=1)
                outputs_old_mask = np.argmax(outputs_old_mask.data.cpu().numpy(), axis=1)
                labels = labels.data.cpu().numpy()
                labels[labels == 0] = outputs_old_mask[labels == 0]
                labels = torch.from_numpy(labels).cuda()
                labels_bkgpaint = labels

                outputs_mask = torch.softmax(outputs, dim=1)
                self.argmax = np.argmax(outputs_mask.data.cpu().numpy(), axis=1)
                outputs_mask = self.argmax
                if cur_epoch>20:
                    outputs_old_mask_out = torch.softmax(out, dim=1)
                    outputs_old_mask_out = np.argmax(outputs_old_mask_out.data.cpu().numpy(), axis=1)
                    labels = labels.data.cpu().numpy()
                    labels[labels == 0] = outputs_old_mask_out[labels == 0]
                    labels = torch.from_numpy(labels).cuda()





                # --可视化代码----------------------------------------------

                if save_image<100:
                    save_image = save_image + 5
                    image_name = train_loader.dataset.dataset.dataset.images[cur_step][0].split('/')[-1].split('.')[0]
                    # name = name[0].split('/')[-1].split('.')[0]
                    image_tosave = (denorm(images[0].detach().cpu().numpy()) * 255).astype(np.uint8).transpose(1, 2, 0)
                    prediction_tosave = label2color(outputs_old_mask)[0].astype(np.uint8)
                    label_tosave_bkgpaint = label2color(labels_bkgpaint.detach().cpu().numpy())[0].astype(np.uint8)
                    label_tosave_bkgpaint_cabpaint = label2color(labels.detach().cpu().numpy())[0].astype(np.uint8)
                    # print(prediction.shape())
                    vis_dir = "/workspace/SDR/outputs/cab"
                    # Image.fromarray(torch.squeeze(torch.from_numpy(prediction[0]),2)).save(f'{test_dir}/{image_name}.png')
                    # Image.fromarray(image_tosave).save(f'{vis_dir}/{image_name}_{i}_RGB.jpg')
                    # Image.fromarray(prediction_tosave).save(f'{vis_dir}/visual/{image_name}_{i}_pred.png')
                    Image.fromarray(prediction_tosave).save(f'{vis_dir}/cab_new/{image_name}_{cur_epoch}_{cur_step}_predction_old.png')
                    Image.fromarray(image_tosave).save(f'{vis_dir}/cab_new/{image_name}_{cur_epoch}_{cur_step}_RGB.jpg')
                    Image.fromarray(label_tosave).save(f'{vis_dir}/cab_new/{image_name}_{cur_epoch}_{cur_step}_label.png')
                    Image.fromarray(label_tosave_bkgpaint).save(f'{vis_dir}/cab_new/{image_name}_{cur_epoch}_{cur_step}_label_bkgpaint.png')
                    Image.fromarray(prediction_tosave).save(
                        f'{vis_dir}/cab_new/{image_name}_{cur_epoch}_{cur_step}_predction_new.png')
                    if cur_epoch>20:

                        Image.fromarray(label_tosave_bkgpaint_cabpaint).save(
                            f'{vis_dir}/cab_new/{image_name}_{cur_epoch}_{cur_step}_label_bkgpaint_cabpaint_afterepoch20.png')
                    else:

                        Image.fromarray(label_tosave_bkgpaint_cabpaint).save(
                            f'{vis_dir}/cab_new/{image_name}_{cur_epoch}_{cur_step}_label_bkgpaint_cabpaint_beforeepoch20.png')
                # -------------------------------------------------




                # xxx BCE / Cross Entropy Loss
            if not self.icarl_only_dist:
                loss = criterion(outputs, labels)  # B x H x W
            else:
                loss = self.licarl(outputs, labels, torch.sigmoid(outputs_old))

            loss = loss.mean()  # scalar

            if self.icarl_combined:
                # tensor.narrow( dim, start, end) -> slice tensor from start to end in the specified dim
                n_cl_old = outputs_old.shape[1]
                # use n_cl_old to sum the contribution of each class, and not to average them (as done in our BCE).
                l_icarl = self.icarl * n_cl_old * self.licarl(outputs.narrow(1, 0, n_cl_old),
                                                              torch.sigmoid(outputs_old))

            # features clustering loss
            if self.lfc_flag or self.lfc_sep_clust:
                lfc, lsep_clusters = self.lfc_loss(labels=labels, outputs=outputs,
                                                   features=features['body'], train_step=cur_step, step=self.step,
                                                   epoch=cur_epoch, incremental_step=self.step, prototypes=prototypes)
            lfc *= self.lfc
            if torch.isnan(lfc):  lfc = torch.tensor(0.)
            lsep_clusters *= self.lfc_sep_clust

            # SNNL loss at features space
            if self.lSNNL_flag:
                lSNNL = self.lSNNL * self.lSNNL_loss(labels=labels, outputs=outputs,
                                                     features=features['body'], train_step=cur_step,
                                                     epoch=cur_epoch)

            # xxx ILTSS (distillation on features or logits)
            if self.lde_flag:
                lde = self.lde * self.lde_loss(features=features['body'], features_old=features_old['body'],
                                               labels=labels, classes_old=self.old_classes)

            if self.lCIL_flag:
                outputs_old_temp = torch.zeros_like(outputs)
                outputs_old_temp[:,:outputs_old.shape[1],:,:] = outputs_old

                lCIL = self.lCIL_loss(outputs=outputs, targets=outputs_old_temp, targets_new=labels)

            if self.ldeprototype_flag:
                ldeprototype = self.ldeprototype * self.ldeprototype_loss(features=features['body'],
                                                                          features_old=features_old[
                                                                              'body'] if self.step != 0 else None,
                                                                          labels=labels,
                                                                          classes_old=self.old_classes,
                                                                          incremental_step=self.step,
                                                                          sequential=self.no_mask,
                                                                          overlapped=self.overlap,
                                                                          outputs_old=outputs_old if self.step != 0 else None,
                                                                          outputs=outputs,
                                                                          loss_de_prototypes_sumafter=self.loss_de_prototypes_sumafter,
                                                                          prototypes=prototypes,
                                                                          count_features=count_features,
                                                                          bkgpaint_flag=self.ours_bkgpaint_flag)
            if self.featupdate_loss_flag:
                if not self.featupdate_count_flag:
                #------success-------
                    old_epoch_prototype = self.Update_old_prototype(features=features['body'],
                                                                              features_old=features_old[
                                                                                  'body'] if self.step != 0 else None,
                                                                              labels=labels,
                                                                              classes_old=self.old_classes,
                                                                              incremental_step=self.step,
                                                                              sequential=self.no_mask,
                                                                              overlapped=self.overlap,
                                                                              outputs_old=outputs_old if self.step != 0 else None,
                                                                              outputs=outputs,
                                                                              loss_de_prototypes_sumafter=self.loss_de_prototypes_sumafter,
                                                                              prototypes=prototypes,
                                                                              count_features=count_features,
                                                                    old_epoch_prototype=old_epoch_prototype,
                                                                    bkgpaint_flag=self.ours_bkgpaint_flag)
                #--------hope useful--------
                else:
                    if self.featupdate_count_last:
                        if cur_epoch ==29:
                            print(self.threshold_sum_flag)
                            old_epoch_prototype, old_count_features = self.Update_old_prototype_teaandstu(features=features['body'],
                                                                                      features_old=features_old[
                                                                                          'body'] if self.step != 0 else None,
                                                                                      labels=labels,
                                                                                      classes_old=self.old_classes,
                                                                                      incremental_step=self.step,
                                                                                      sequential=self.no_mask,
                                                                                      overlapped=self.overlap,
                                                                                      outputs_old=outputs_old if self.step != 0 else None,
                                                                                      outputs=outputs,
                                                                                      loss_de_prototypes_sumafter=self.loss_de_prototypes_sumafter,
                                                                                      prototypes=prototypes,
                                                                                      count_features=count_features,
                                                                            old_epoch_prototype=old_epoch_prototype,
                                                                            bkgpaint_flag=self.ours_bkgpaint_flag,
                                                                                old_count_features=old_count_features,
                                                                            thred=self.threshold_sum_flag)
                        else:
                            pass
                    else:
                        # print('Update_old_prototype_tea!!!!!!!!!!!')
                        old_epoch_prototype, old_count_features = self.Update_old_prototype_teaandstu(
                            features=features['body'],
                            features_old=features_old[
                                'body'] if self.step != 0 else None,
                            labels=labels,
                            classes_old=self.old_classes,
                            incremental_step=self.step,
                            sequential=self.no_mask,
                            overlapped=self.overlap,
                            outputs_old=outputs_old if self.step != 0 else None,
                            outputs=outputs,
                            loss_de_prototypes_sumafter=self.loss_de_prototypes_sumafter,
                            prototypes=prototypes,
                            count_features=count_features,
                            old_epoch_prototype=old_epoch_prototype,
                            bkgpaint_flag=self.ours_bkgpaint_flag,
                            old_count_features=old_count_features,
                        thred=self.threshold_sum_flag)
            # Features Sparsification Loss
            if self.lfs_flag:
                lfs = self.lfs * self.lfs_loss(features=features['body'], labels=labels)

            if self.lkd_flag:
                # resize new output to remove new logits and keep only the old ones
                lkd = self.lkd * self.lkd_loss(outputs, outputs_old)

            # xxx first backprop of previous loss (compute the gradients for regularization methods)
            # loss_tot = cosine_loss.cpu()
            loss_tot = loss + lkd + lde + l_icarl + lfc + lSNNL + lsep_clusters + ldeprototype + lfs + lCIL + cab_loss + 0.01*cosine_loss
            # loss_tot = cosine_loss + lkd + lde + l_icarl + lfc + lSNNL + lsep_clusters + ldeprototype + lfs + lCIL + cab_loss

            if self.where_to_sim == 'GPU_server':
                from apex import amp
                with amp.scale_loss(loss_tot, optim) as scaled_loss:
                    scaled_loss.backward()
            else:
                pass
                loss_tot.backward()

            # xxx Regularizer (EWC, RW, PI)
            if self.regularizer_flag:
                if self.where_to_sim == 'GPU_windows' or self.where_to_sim == 'CPU_windows' or distributed.get_rank() == 0:
                    self.regularizer.update()
                l_reg = self.reg_importance * self.regularizer.penalty()
                if l_reg != 0.:
                    if self.where_to_sim == 'GPU_server':
                        with amp.scale_loss(l_reg, optim) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        l_reg.backward()

            optim.step()
            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item()
            reg_loss += l_reg.item() if l_reg != 0. else 0.
            reg_loss += lkd.item() + lde.item() + l_icarl.item() + lfc.item() + lSNNL.item() + lsep_clusters.item()
            interval_loss += loss.item() + lkd.item() + lde.item() + l_icarl.item() + lfc.item() + \
                             lSNNL.item() + lsep_clusters.item() + ldeprototype.item() + lfs.item() + lCIL.item() + cosine_loss.item()
            interval_loss += l_reg.item() if l_reg != 0. else 0.

            if (cur_step + 1) % print_int == 0:
                interval_loss = interval_loss / print_int
                logger.info(f"Epoch {cur_epoch + 1}, Batch {cur_step + 1}/{len(train_loader)},"
                            f" Loss={interval_loss}, Time taken={time.time() - start_time}")
                logger.debug(f"Loss made of: CE {loss}, LKD {lkd}, LDE {lde}, LReg {l_reg}, Lfc {lfc}, "
                             f"LSNNL {lSNNL}, Lsepclus {lsep_clusters}, LDEProto {ldeprototype}, Lfeatspars {lfs}, "
                             f"LCIL {lCIL}, cabloss {cab_loss}, cosine_loss {cosine_loss}")
                print(f"Epoch {cur_epoch + 1}, Batch {cur_step + 1}/{len(train_loader)},"
                            f" Loss={interval_loss}, Time taken={time.time() - start_time}")
                print(f"Loss made of: CE {loss}, LKD {lkd}, LDE {lde}, LReg {l_reg}, Lfc {lfc}, "
                             f"LSNNL {lSNNL}, Lsepclus {lsep_clusters}, LDEProto {ldeprototype}, Lfeatspars {lfs}, "
                             f"LCIL {lCIL}, cabloss {cab_loss}", f"cosine_loss {cosine_loss}")
                # visualization
                if logger is not None:
                    x = cur_epoch * len(train_loader) + cur_step + 1
                    logger.add_scalar('Losses/interval_loss', interval_loss, x)
                    if self.lfc_flag:
                        logger.add_scalar('Losses/lfc', lfc.item(), x)
                    if self.lSNNL_flag:
                        logger.add_scalar('Losses/lSNNL', lSNNL.item(), x)
                    if self.lfc_sep_clust:
                        logger.add_scalar('Losses/lsep_clusters', lsep_clusters.item(), x)
                    if self.ldeprototype_flag:
                        logger.add_scalar('Losses/lde_prototypes', ldeprototype.item(), x)
                    if self.lfs_flag:
                        logger.add_scalar('Losses/lfs', lfs.item(), x)
                    if self.lCIL_flag:
                        logger.add_scalar('Losses/lCIL', lCIL.item(), x)


                interval_loss = 0.0
                start_time = time.time()

        logger.info(f"END OF EPOCH {cur_epoch + 1}, TOTAL TIME={time.time() - start_epoch_time}")
        print(f"END OF EPOCH {cur_epoch + 1}, TOTAL TIME={time.time() - start_epoch_time}")

        # collect statistics from multiple processes
        epoch_loss = torch.tensor(epoch_loss).to(self.device)
        reg_loss = torch.tensor(reg_loss).to(self.device)

        if not self.where_to_sim == 'GPU_windows':
            torch.distributed.reduce(epoch_loss, dst=0)
            torch.distributed.reduce(reg_loss, dst=0)

        if self.where_to_sim == 'GPU_windows' or self.where_to_sim == 'CPU_windows':
            epoch_loss = epoch_loss / world_size / len(train_loader)
            reg_loss = reg_loss / world_size / len(train_loader)
        else:
            if distributed.get_rank() == 0:
                epoch_loss = epoch_loss / distributed.get_world_size() / len(train_loader)
                reg_loss = reg_loss / distributed.get_world_size() / len(train_loader)

        logger.info(f"Epoch {cur_epoch + 1}, Class Loss={epoch_loss}, Reg Loss={reg_loss}")
        if old_epoch_prototype is not None:
            print(old_epoch_prototype)
            logger.info(f"old_epoch_prototype:{old_epoch_prototype}")
            print(old_count_features)
            logger.info(f"old_count_features:{old_count_features}")

        if old_epoch_prototype is not None:
            if self.featupdate_count_flag:
                return (epoch_loss, reg_loss), prototypes, count_features, old_epoch_prototype, old_count_features
            else:
                return (epoch_loss, reg_loss), prototypes, count_features, old_epoch_prototype
        else:
            return (epoch_loss, reg_loss), prototypes, count_features



    def validate(self, loader, metrics, world_size, prototypes=None, ret_samples_ids=None, logger=None, vis_dir=None, test_dir=None, label2color=None, denorm=None):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        device = self.device
        criterion = self.criterion
        model.eval()

        class_loss = 0.0
        reg_loss = 0.0
        lkd = torch.tensor(0.)
        lde = torch.tensor(0.)
        l_icarl = torch.tensor(0.)
        l_reg = torch.tensor(0.)
        lfc = torch.tensor(0.)
        lsep_clusters = torch.tensor(0.)
        lSNNL = torch.tensor(0.)
        ldeprototype = torch.tensor(0.)
        lfs = torch.tensor(0.)
        lCIL = torch.tensor(0.)

        ret_samples = []
        features_total = []
        with torch.no_grad():
            pbar = tqdm(loader)
            for i, (images, labels) in enumerate(pbar):

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                if (self.lde_flag or self.lkd_flag or self.icarl_dist_flag or self.ldeprototype_flag or
                    self.lfc_flag or self.lfc_sep_clust or self.lSNNL_flag or self.lCIL_flag) and self.model_old is not None:
                    with torch.no_grad():
                        outputs_old, features_old = self.model_old(images, ret_intermediate=True)

                outputs, features = model(images, ret_intermediate=True)
                # scale_pred = F.upsample(input=features['body'][0,...].unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=True)
                # scale_pred = F.upsample(input=outputs[0,...].unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=True)
                # scale_pred = torch.squeeze(scale_pred, 0)
                # # scale_pred = torch.mean(scale_pred, dim=0)
                # scale_pred = torch.max(scale_pred, dim=0)[0]
                # visual = scale_pred.cpu().numpy()
                # fig = plt.gcf()
                # fig.set_size_inches(2, 2)
                # plt.gca().xaxis.set_major_locator(plt.NullLocator())
                # plt.gca().yaxis.set_major_locator(plt.NullLocator())
                # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
                # plt.margins(0, 0)
                # plt.imshow(visual, cmap='jet')
                # plt.axis('off')
                # # plt.colorbar()
                # plt.savefig('/workspace/visual/sdr_{}.png'.format(i), dpi=1000)
                # image_tosave = (denorm(images[0].detach().cpu().numpy()) * 255).astype(np.uint8).transpose(1, 2, 0)
                #
                # Image.fromarray(image_tosave).save(f'/workspace/visual/sdr_{i}_RGB.jpg')


                # xxx BCE / Cross Entropy Loss
                if not self.icarl_only_dist:
                    loss = criterion(outputs, labels)  # B x H x W
                else:
                    loss = self.licarl(outputs, labels, torch.sigmoid(outputs_old))

                loss = loss.mean()  # scalar

                if self.icarl_combined:
                    # tensor.narrow( dim, start, end) -> slice tensor from start to end in the specified dim
                    n_cl_old = outputs_old.shape[1]
                    # use n_cl_old to sum the contribution of each class, and not to average them (as done in our BCE).
                    l_icarl = self.icarl * n_cl_old * self.licarl(outputs.narrow(1, 0, n_cl_old),
                                                                  torch.sigmoid(outputs_old))

                # features clustering loss
                if self.lfc_flag or self.lfc_sep_clust:
                    lfc, lsep_clusters = self.lfc_loss(labels=labels, outputs=outputs,
                                                       features=features['body'], val=True)

                # SNNL loss at features space
                if self.lSNNL_flag:
                    lSNNL = self.lSNNL * self.lSNNL_loss(labels=labels, outputs=outputs,
                                                         features=features['body'], val=True)

                # xxx ILTSS (distillation on features or logits)
                if self.lde_flag:
                    lde = self.lde * self.lde_loss(features=features['body'], features_old=features_old['body'],
                                                   labels=labels, classes_old=self.old_classes)

                # Features Sparsification Loss
                if self.lfs_flag:
                    lfs = self.lfs * self.lfs_loss(features=features['body'], labels=labels, val=True)

                if self.lkd_flag:
                    lkd = self.lkd_loss(outputs, outputs_old)

                if self.lCIL_flag:
                    outputs_old_temp = torch.zeros_like(outputs)
                    outputs_old_temp[:, :outputs_old.shape[1], :, :] = outputs_old
                    lCIL = self.lCIL_loss(outputs=outputs, targets=outputs_old_temp, targets_new=labels)

                # xxx Regularizer (EWC, RW, PI)
                if self.regularizer_flag:
                    l_reg = self.regularizer.penalty()

                # labels_ori = labels.clone()

                # if self.ours_bkgpaint_flag:
                #     outputs_old_mask = torch.argmax(torch.softmax(outputs_old, dim=1), dim=1)
                #     labels[labels<11] =0
                #     labels[labels == 0] = outputs_old_mask[labels == 0]

                class_loss += loss.item()
                reg_loss += l_reg.item() if l_reg != 0. else 0.
                reg_loss += lkd.item() + lde.item() + l_icarl.item() + lfc.item() + lSNNL.item() + \
                            lsep_clusters.item() + ldeprototype.item() + lfs.item() + lCIL.item()

                _, prediction = outputs.max(dim=1)
                # _, prediction_old = outputs_old.max(dim=1)

                # labels = labels.cpu().numpy()
                # labels_ori = labels_ori.cpu().numpy()
                labels = labels.cpu().numpy()
                prediction = prediction.cpu().numpy()
                # prediction_old = prediction_old.cpu().numpy()
                metrics.update(labels, prediction)
                # labels = label2color(labels)[0].astype(np.uint8)
                # Image.fromarray(labels).save(f'/workspace/visual/sdr_{i}_label.png')
                if vis_dir is not None:
                    image_name = loader.dataset.dataset.dataset.images[i][0].split('/')[-1].split('.')[0]
                    # name = name[0].split('/')[-1].split('.')[0]
                    image_tosave = (denorm(images[0].detach().cpu().numpy()) * 255).astype(np.uint8).transpose(1,2,0)
                    prediction_tosave = label2color(prediction)[0].astype(np.uint8)
                    # prediction_old_tosave = label2color(prediction_old)[0].astype(np.uint8)

                    # label_tosave = label2color(labels_ori)[0].astype(np.uint8)
                    labels = label2color(labels)[0].astype(np.uint8)
                    # print(prediction.shape())


                    Image.fromarray(image_tosave).save(f'{vis_dir}/{image_name}_{i}_RGB.jpg')
                    Image.fromarray(prediction_tosave).save(f'{vis_dir}/{image_name}_{i}_pred_updateprototype.png')
                    # Image.fromarray(prediction_old_tosave).save(f'{vis_dir}/{image_name}_{i}_pred_old.png')
                    # Image.fromarray(label_tosave).save(f'{vis_dir}/{image_name}_{i}_label.png')
                    Image.fromarray(labels).save(f'{vis_dir}/{image_name}_{i}_label.png')
                    features_total.append(features['body'])
                    # save also features here
                    if i%10 == 0:
                        try:
                            os.mkdir(f'{vis_dir}/features/')
                        except:
                            pass
                        torch.cat(features_total,dim=0)
                        np.save(f'{vis_dir}/features/{i}.npy', features['body'].cpu().numpy())


                if ret_samples_ids is not None and i in ret_samples_ids:  # get samples
                    ret_samples.append((images[0].detach().cpu().numpy(),
                                        labels[0],
                                        prediction[0]))

            # collect statistics from multiple processes
            metrics.synch(device)
            score = metrics.get_results()

            class_loss = torch.tensor(class_loss).to(self.device)
            reg_loss = torch.tensor(reg_loss).to(self.device)

            if not self.where_to_sim == 'GPU_windows':
                torch.distributed.reduce(class_loss, dst=0)
                torch.distributed.reduce(reg_loss, dst=0)

            if self.where_to_sim == 'GPU_windows' or self.where_to_sim == 'CPU_windows':
                class_loss = class_loss / world_size / len(loader)
                reg_loss = reg_loss / world_size / len(loader)
            else:
                if distributed.get_rank() == 0:
                    class_loss = class_loss / distributed.get_world_size() / len(loader)
                    reg_loss = reg_loss / distributed.get_world_size() / len(loader)

            if logger is not None:
                logger.info(f"Validation, Class Loss={class_loss}, Reg Loss={reg_loss} (without scaling)")

        return (class_loss, reg_loss), score, ret_samples

    def state_dict(self):
        state = {"regularizer": self.regularizer.state_dict() if self.regularizer_flag else None}

        return state

    def load_state_dict(self, state):
        if state["regularizer"] is not None and self.regularizer is not None:
            self.regularizer.load_state_dict(state["regularizer"])

    def _update_running_stats(self, labels_down, features, sequential, overlapped, incremental_step, prototypes, count_features):
        cl_present = torch.unique(input=labels_down)

        # if overlapped: exclude background as we could not have a reliable statistics 排除背景，因为我们没法获得可靠的统计数据
        # if disjoint (not overlapped) and step is > 0: exclude bgr as could contain old classes
        if overlapped or ((not sequential) and incremental_step > 0):
            cl_present = cl_present[1:]

        if cl_present[-1] == 255:
            cl_present = cl_present[:-1]

        features_local_mean = torch.zeros([self.num_classes, 2048], device=self.device)
        # features_local_mean_batch = torch.zeros([features.shape[0], self.num_classes, 2048], device=self.device)
        # prototypes_batch = torch.zeros([features.shape[0],self.num_classes, 2048])
        for cl in cl_present:
            if features[(labels_down == cl).expand(-1, features.shape[1], -1, -1)].shape[0] % 2048 != 0:
                pass

            features_cl = features[(labels_down == cl).expand(-1, features.shape[1], -1, -1)].view(features.shape[1],-1).detach()
            # features_cl_batch = features[(labels_down == cl).expand(-1, features.shape[1], -1, -1)].view(features.shape[0], features.shape[1],-1).detach()
            # features_local_mean_batch = torch.mean(features_cl_batch, dim=-1)
            features_local_mean[cl] = torch.mean(features_cl.detach(), dim=-1)
            features_cl_sum = torch.sum(features_cl.detach(), dim=-1)
            # cumulative moving average for each feature vector
            # S_{n+f} = ( sum(x_{n+1} + ... + x_{n+f}) + n * S_n) / (n + f)
            features_running_mean_tot_cl = (features_cl_sum + count_features.detach()[cl] *
                                            prototypes.detach()[cl]) \
                                           / (count_features.detach()[cl] + features_cl.shape[-1])
            # count_features 表示到现在为止一共有多少个2048维的数组参与了当前protopype的计算。
            count_features[cl] += features_cl.shape[-1]
            prototypes[cl] = features_running_mean_tot_cl


        return prototypes, count_features



    def Update_old_prototype(self, outputs, outputs_old, features, features_old, labels, classes_old, incremental_step,
                sequential=False, overlapped=False, loss_de_prototypes_sumafter=False, val=False, prototypes=None,
                count_features=None, old_epoch_prototype=None, bkgpaint_flag=False):



        labels = labels.unsqueeze(dim=1)  # 8*1*512*512
        labels_down = (F.interpolate(input=labels.double(), size=(features.shape[2], features.shape[3]),
                                     mode='nearest')).long()  # 8*1*32*32
        if bkgpaint_flag:
            labels_down_bgr_mask = ((labels_down < classes_old) & (labels_down > 0)).long()
        else:
            labels_down_bgr_mask = (labels_down == 0).long()  # 8*1*32*32

        if self.num_classes is not None and not val:

            if incremental_step != 0:
                if sequential:  # we can directly use the current groundtruth masked to consider only previous classes.
                    pseudolabel_old_down = labels_down * (labels_down < classes_old).long()
                else:  # if disjoint or overlapped: old classes are bgr in new images, hence we rely on previous model output
                    outputs_old = torch.argmax(outputs_old, dim=1,
                                               keepdim=True)  # TODO: investigate with other functions (entropy,...)
                    outputs_old_down = (
                        F.interpolate(input=outputs_old.double(), size=(features.shape[2], features.shape[3]),
                                      mode='nearest')).long()
                    pseudolabel_old_down = outputs_old_down * labels_down_bgr_mask.long()
                # torch.unique() 挑出tensor中的独立不重复元素
                # eg. task1 0,1,2,2  cl_present is [0,1,2]
                cl_present = torch.unique(input=pseudolabel_old_down).long()
                if cl_present[0] == 0:
                    # eg cl_present is [0,1,2], after cl_present = cl_present[1:], cl_present is [1,2]
                    cl_present = cl_present[1:]

                for cl in cl_present:
                    # prototype 6*2048 classes * feat_dim
                    prototype = prototypes.detach()[cl]
                    # torch.expand_as()把一个tensor变成和括号内一样形状的tensor
                    # features 8*2048*32**32
                    # pseudolabel_old_down == cl 返回的是和pseudolabel_old_down大小相同的True、False 8*1*32*32
                    # (pseudolabel_old_down == cl).expand_as(features) 返回和features大小相同的tensor 8*2048*512*512, 里面的元素都是Ture、False
                    # features[(pseudolabel_old_down == cl).expand_as(features)] 把Ture位置的数值都取出来，返回的是一个一位数组
                    # features_cl = features[(labels_down == cl).expand(-1, features.shape[1], -1, -1)].view(
                    #     features.shape[1], -1).detach()
                    current_features = features[(pseudolabel_old_down == cl).expand_as(features)].view(-1,
                                                                                                       features.shape[
                                                                                                           1]).detach()


                    if old_epoch_prototype != None:
                        current_features_ = torch.cat((current_features.detach(),
                                                      torch.unsqueeze(old_epoch_prototype[cl, :], dim=0)), dim=0)
                        current_proto = torch.mean(current_features_.detach(), dim=0)

                        a = torch.sum(old_epoch_prototype[cl, :] - prototype)
                        if abs(a.item()) < 1000:
                            old_epoch_prototype[cl, :] = current_proto.detach()
                        else:
                            old_epoch_prototype[cl, :] = prototype

                    # print(a)

        return old_epoch_prototype

    def Update_old_prototype_new(self, outputs, outputs_old, features, features_old, labels, classes_old, incremental_step,
                sequential=False, overlapped=False, loss_de_prototypes_sumafter=False, val=False, prototypes=None,
                count_features=None, old_epoch_prototype=None, bkgpaint_flag=False, old_count_features=None, thred=10000):



        labels = labels.unsqueeze(dim=1)  # 8*1*512*512
        labels_down = (F.interpolate(input=labels.double(), size=(features.shape[2], features.shape[3]),
                                     mode='nearest')).long()  # 8*1*32*32
        if bkgpaint_flag:
            labels_down_bgr_mask = ((labels_down < classes_old) & (labels_down > 0)).long()
        else:
            labels_down_bgr_mask = (labels_down == 0).long()  # 8*1*32*32

        if self.num_classes is not None and not val:

            if incremental_step != 0:
                if sequential:  # we can directly use the current groundtruth masked to consider only previous classes.
                    pseudolabel_old_down = labels_down * (labels_down < classes_old).long()
                else:  # if disjoint or overlapped: old classes are bgr in new images, hence we rely on previous model output
                    outputs_old = torch.argmax(outputs_old, dim=1,
                                               keepdim=True)  # TODO: investigate with other functions (entropy,...)
                    outputs_old_down = (
                        F.interpolate(input=outputs_old.double(), size=(features.shape[2], features.shape[3]),
                                      mode='nearest')).long()
                    pseudolabel_old_down = outputs_old_down * labels_down_bgr_mask.long()
                    # print(torch.sum(mask))
                    # print(torch.sum(pseudolabel_old_down))
                # torch.unique() 挑出tensor中的独立不重复元素
                # eg. task1 0,1,2,2  cl_present is [0,1,2]
                cl_present = torch.unique(input=pseudolabel_old_down).long()
                if cl_present[0] == 0:
                    # eg cl_present is [0,1,2], after cl_present = cl_present[1:], cl_present is [1,2]
                    cl_present = cl_present[1:]
                features_local_mean = torch.zeros([classes_old, 2048], device=self.device)
                for cl in cl_present:
                    # prototype 6*2048 classes * feat_dim
                    prototype_cl = prototypes.detach()[cl]
                    # torch.expand_as()把一个tensor变成和括号内一样形状的tensor
                    # features 8*2048*32**32
                    # pseudolabel_old_down == cl 返回的是和pseudolabel_old_down大小相同的True、False 8*1*32*32
                    # (pseudolabel_old_down == cl).expand_as(features) 返回和features大小相同的tensor 8*2048*512*512, 里面的元素都是Ture、False
                    # features[(pseudolabel_old_down == cl).expand_as(features)] 把Ture位置的数值都取出来，返回的是一个一位数组
                    # features_cl = features[(labels_down == cl).expand(-1, features.shape[1], -1, -1)].view(
                    #     features.shape[1], -1).detach()
                    #------update prototype according to the representation of student model-----
                    # current_features = features[(mask == cl).expand_as(features)].view(features.shape[1], -1).detach()
                    # ------update prototype according to the representation of teacher model-----
                    current_features = features[(pseudolabel_old_down == cl).expand_as(features)].view(features_old.shape[1], -1).detach()
                    # current_features = torch.cat((current_features_old, current_features_new),dim=-1)
                    if current_features.shape[-1] > 4:
                        N = int(current_features.shape[-1] / 4)
                        index = torch.squeeze(torch.randint(0,current_features.shape[-1],(1,N)))
                        current_features = current_features[:,index]
                    features_local_mean[cl] = torch.mean(current_features.detach(), dim=-1)
                    features_cl_sum = torch.sum(current_features.detach(), dim=-1)
                    # cumulative moving average for each feature vector
                    # S_{n+f} = ( sum(x_{n+1} + ... + x_{n+f}) + n * S_n) / (n + f)
                    features_running_mean_tot_cl = (features_cl_sum + old_count_features[cl] *
                                                    old_epoch_prototype[cl, :]) / (old_count_features[cl] + current_features.shape[-1])
                    # count_features 表示到现在为止一共有多少个2048维的数组参与了当前protopype的计算。
                    prototype_sum = torch.sum(features_local_mean[cl] - prototype_cl)
                    # print('thred:', thred)
                    # print(torch.sum(old_epoch_prototype[cl, :] - features_running_mean_tot_cl))
                    if torch.abs(prototype_sum) < thred:
                        old_epoch_prototype[cl, :] = features_running_mean_tot_cl
                        old_count_features[cl] = old_count_features[cl] + current_features.shape[-1]
                    else:
                        print('!>thr')

                    # print(a)

        return old_epoch_prototype, old_count_features


    def Update_old_prototype_nodown(self, outputs, outputs_old, features, features_old, labels, classes_old, incremental_step,
                sequential=False, overlapped=False, loss_de_prototypes_sumafter=False, val=False, prototypes=None,
                count_features=None, old_epoch_prototype=None, bkgpaint_flag=False, old_count_features=None, thred=10000):



        labels = labels.unsqueeze(dim=1)  # 8*1*512*512
        labels_down = (F.interpolate(input=labels.double(), size=(128, 128),
                                     mode='nearest')).long()  # 8*1*32*32
        features_old = (F.interpolate(input=features_old.double(), size=(128, 128),
                                     mode='nearest')).long()
        features = (F.interpolate(input=features.double(), size=(128, 128),
                                      mode='nearest')).long()
        if bkgpaint_flag:
            labels_down_bgr_mask = ((labels_down < classes_old) & (labels_down > 0)).long()
        else:
            labels_down_bgr_mask = (labels_down == 0).long()  # 8*1*32*32

        if self.num_classes is not None and not val:

            if incremental_step != 0:
                if sequential:  # we can directly use the current groundtruth masked to consider only previous classes.
                    pseudolabel_old_down = labels_down * (labels_down < classes_old).long()
                else:  # if disjoint or overlapped: old classes are bgr in new images, hence we rely on previous model output
                    outputs_old = torch.argmax(outputs_old, dim=1,
                                               keepdim=True)  # TODO: investigate with other functions (entropy,...)
                    outputs_old_down = (
                        F.interpolate(input=outputs_old.double(), size=(128, 128),
                                      mode='nearest')).long()
                    pseudolabel_old_down = outputs_old_down * labels_down_bgr_mask.long()
                    # print(torch.sum(mask))
                    # print(torch.sum(pseudolabel_old_down))
                # torch.unique() 挑出tensor中的独立不重复元素
                # eg. task1 0,1,2,2  cl_present is [0,1,2]
                cl_present = torch.unique(input=pseudolabel_old_down).long()
                if cl_present[0] == 0:
                    # eg cl_present is [0,1,2], after cl_present = cl_present[1:], cl_present is [1,2]
                    cl_present = cl_present[1:]
                features_local_mean = torch.zeros([classes_old, 2048], device=self.device)
                for cl in cl_present:
                    # prototype 6*2048 classes * feat_dim
                    prototype_cl = prototypes.detach()[cl]

                    current_features_old = features_old[(pseudolabel_old_down == cl).expand_as(features_old)].view(features_old.shape[1], -1).detach()
                    current_features_new = features[(pseudolabel_old_down == cl).expand_as(features)].view(features_old.shape[1], -1).detach()
                    current_features = torch.cat((current_features_old, current_features_new),dim=-1)
                    features_local_mean[cl] = torch.mean(current_features.detach(), dim=-1)
                    features_cl_sum = torch.sum(current_features.detach(), dim=-1)
                    # cumulative moving average for each feature vector
                    # S_{n+f} = ( sum(x_{n+1} + ... + x_{n+f}) + n * S_n) / (n + f)
                    features_running_mean_tot_cl = (features_cl_sum + old_count_features[cl] *
                                                    old_epoch_prototype[cl, :]) / (old_count_features[cl] + current_features.shape[-1])
                    # count_features 表示到现在为止一共有多少个2048维的数组参与了当前protopype的计算。
                    prototype_sum = torch.sum(features_local_mean[cl] - prototype_cl)
                    # print('thred:', thred)
                    # print(torch.sum(old_epoch_prototype[cl, :] - features_running_mean_tot_cl))
                    if torch.abs(prototype_sum) < thred:
                        old_epoch_prototype[cl, :] = features_running_mean_tot_cl
                        old_count_features[cl] = old_count_features[cl] + current_features.shape[-1]
                    else:
                        print('!>thr')

                    # print(a)

        return old_epoch_prototype, old_count_features


    def Update_old_prototype_teaandstu(self, outputs, outputs_old, features, features_old, labels, classes_old, incremental_step,
                sequential=False, overlapped=False, loss_de_prototypes_sumafter=False, val=False, prototypes=None,
                count_features=None, old_epoch_prototype=None, bkgpaint_flag=False, old_count_features=None, thred=10000):



        labels = labels.unsqueeze(dim=1)  # 8*1*512*512
        labels_down = (F.interpolate(input=labels.double(), size=(features.shape[2], features.shape[3]),
                                     mode='nearest')).long()  # 8*1*32*32
        if bkgpaint_flag:
            labels_down_bgr_mask = ((labels_down < classes_old) & (labels_down > 0)).long()
        else:
            labels_down_bgr_mask = (labels_down == 0).long()  # 8*1*32*32

        if self.num_classes is not None and not val:

            if incremental_step != 0:
                if sequential:  # we can directly use the current groundtruth masked to consider only previous classes.
                    pseudolabel_old_down = labels_down * (labels_down < classes_old).long()
                else:  # if disjoint or overlapped: old classes are bgr in new images, hence we rely on previous model output
                    outputs_old = torch.argmax(outputs_old, dim=1,
                                               keepdim=True)  # TODO: investigate with other functions (entropy,...)
                    outputs_old_down = (
                        F.interpolate(input=outputs_old.double(), size=(features.shape[2], features.shape[3]),
                                      mode='nearest')).long()
                    pseudolabel_old_down = outputs_old_down * labels_down_bgr_mask.long()
                    # print(torch.sum(mask))
                    # print(torch.sum(pseudolabel_old_down))
                # torch.unique() 挑出tensor中的独立不重复元素
                # eg. task1 0,1,2,2  cl_present is [0,1,2]
                cl_present = torch.unique(input=pseudolabel_old_down).long()
                if cl_present[0] == 0:
                    # eg cl_present is [0,1,2], after cl_present = cl_present[1:], cl_present is [1,2]
                    cl_present = cl_present[1:]
                features_local_mean = torch.zeros([classes_old, 2048], device=self.device)
                for cl in cl_present:
                    # prototype 6*2048 classes * feat_dim
                    prototype_cl = prototypes.detach()[cl]
                    # torch.expand_as()把一个tensor变成和括号内一样形状的tensor
                    # features 8*2048*32**32
                    # pseudolabel_old_down == cl 返回的是和pseudolabel_old_down大小相同的True、False 8*1*32*32
                    # (pseudolabel_old_down == cl).expand_as(features) 返回和features大小相同的tensor 8*2048*512*512, 里面的元素都是Ture、False
                    # features[(pseudolabel_old_down == cl).expand_as(features)] 把Ture位置的数值都取出来，返回的是一个一位数组
                    # features_cl = features[(labels_down == cl).expand(-1, features.shape[1], -1, -1)].view(
                    #     features.shape[1], -1).detach()
                    #------update prototype according to the representation of student model-----
                    # current_features = features[(mask == cl).expand_as(features)].view(features.shape[1], -1).detach()
                    # ------update prototype according to the representation of teacher model-----
                    current_features_old = features_old[(pseudolabel_old_down == cl).expand_as(features_old)].view(features_old.shape[1], -1).detach()
                    current_features_new = features[(pseudolabel_old_down == cl).expand_as(features)].view(features_old.shape[1], -1).detach()
                    current_features = torch.cat((current_features_old, current_features_new),dim=-1)
                    features_local_mean[cl] = torch.mean(current_features.detach(), dim=-1)
                    features_cl_sum = torch.sum(current_features.detach(), dim=-1)
                    # cumulative moving average for each feature vector
                    # S_{n+f} = ( sum(x_{n+1} + ... + x_{n+f}) + n * S_n) / (n + f)
                    features_running_mean_tot_cl = (features_cl_sum + old_count_features[cl] *
                                                    old_epoch_prototype[cl, :]) / (old_count_features[cl] + current_features.shape[-1])
                    # count_features 表示到现在为止一共有多少个2048维的数组参与了当前protopype的计算。
                    prototype_sum = torch.sum(features_local_mean[cl] - prototype_cl)
                    # print('thred:', thred)
                    # print(torch.sum(old_epoch_prototype[cl, :] - features_running_mean_tot_cl))
                    if torch.abs(prototype_sum) < thred:
                        old_epoch_prototype[cl, :] = features_running_mean_tot_cl
                        old_count_features[cl] = old_count_features[cl] + current_features.shape[-1]
                    else:
                        print('!>thr')

                    # print(a)

        return old_epoch_prototype, old_count_features

    def Update_old_prototype_tea(self, outputs, outputs_old, features, features_old, labels, classes_old, incremental_step,
                sequential=False, overlapped=False, loss_de_prototypes_sumafter=False, val=False, prototypes=None,
                count_features=None, old_epoch_prototype=None, bkgpaint_flag=False, old_count_features=None, thred=10000):



        labels = labels.unsqueeze(dim=1)  # 8*1*512*512
        labels_down = (F.interpolate(input=labels.double(), size=(features.shape[2], features.shape[3]),
                                     mode='nearest')).long()  # 8*1*32*32
        if bkgpaint_flag:
            labels_down_bgr_mask = ((labels_down < classes_old) & (labels_down > 0)).long()
        else:
            labels_down_bgr_mask = (labels_down == 0).long()  # 8*1*32*32

        if self.num_classes is not None and not val:

            if incremental_step != 0:
                if sequential:  # we can directly use the current groundtruth masked to consider only previous classes.
                    pseudolabel_old_down = labels_down * (labels_down < classes_old).long()
                else:  # if disjoint or overlapped: old classes are bgr in new images, hence we rely on previous model output
                    outputs_old = torch.argmax(outputs_old, dim=1,
                                               keepdim=True)  # TODO: investigate with other functions (entropy,...)
                    outputs_old_down = (
                        F.interpolate(input=outputs_old.double(), size=(features.shape[2], features.shape[3]),
                                      mode='nearest')).long()
                    pseudolabel_old_down = outputs_old_down * labels_down_bgr_mask.long()
                    # print(torch.sum(mask))
                    # print(torch.sum(pseudolabel_old_down))
                # torch.unique() 挑出tensor中的独立不重复元素
                # eg. task1 0,1,2,2  cl_present is [0,1,2]
                cl_present = torch.unique(input=pseudolabel_old_down).long()
                if cl_present[0] == 0:
                    # eg cl_present is [0,1,2], after cl_present = cl_present[1:], cl_present is [1,2]
                    cl_present = cl_present[1:]
                features_local_mean = torch.zeros([classes_old, 2048], device=self.device)
                for cl in cl_present:
                    # prototype 6*2048 classes * feat_dim
                    prototype_cl = prototypes.detach()[cl]
                    # torch.expand_as()把一个tensor变成和括号内一样形状的tensor
                    # features 8*2048*32**32
                    # pseudolabel_old_down == cl 返回的是和pseudolabel_old_down大小相同的True、False 8*1*32*32
                    # (pseudolabel_old_down == cl).expand_as(features) 返回和features大小相同的tensor 8*2048*512*512, 里面的元素都是Ture、False
                    # features[(pseudolabel_old_down == cl).expand_as(features)] 把Ture位置的数值都取出来，返回的是一个一位数组
                    # features_cl = features[(labels_down == cl).expand(-1, features.shape[1], -1, -1)].view(
                    #     features.shape[1], -1).detach()
                    #------update prototype according to the representation of student model-----
                    # current_features = features[(mask == cl).expand_as(features)].view(features.shape[1], -1).detach()
                    # ------update prototype according to the representation of teacher model-----
                    current_features = features_old[(pseudolabel_old_down == cl).expand_as(features_old)].view(features_old.shape[1], -1).detach()
                    features_local_mean[cl] = torch.mean(current_features.detach(), dim=-1)
                    features_cl_sum = torch.sum(current_features.detach(), dim=-1)
                    # cumulative moving average for each feature vector
                    # S_{n+f} = ( sum(x_{n+1} + ... + x_{n+f}) + n * S_n) / (n + f)
                    features_running_mean_tot_cl = (features_cl_sum + old_count_features[cl] *
                                                    old_epoch_prototype[cl, :]) / (old_count_features[cl] + current_features.shape[-1])
                    # count_features 表示到现在为止一共有多少个2048维的数组参与了当前protopype的计算。
                    prototype_sum = torch.sum(features_local_mean[cl] - prototype_cl)
                    # print('thred:', thred)
                    # print(torch.sum(old_epoch_prototype[cl, :] - features_running_mean_tot_cl))
                    if torch.abs(prototype_sum) < thred:
                        old_epoch_prototype[cl, :] = features_running_mean_tot_cl
                        old_count_features[cl] = old_count_features[cl] + current_features.shape[-1]
                    else:
                        print('!>thr')

                    # print(a)

        return old_epoch_prototype, old_count_features

    def Update_old_prototype_stu(self, outputs, outputs_old, features, features_old, labels, classes_old, incremental_step,
                sequential=False, overlapped=False, loss_de_prototypes_sumafter=False, val=False, prototypes=None,
                count_features=None, old_epoch_prototype=None, bkgpaint_flag=False, old_count_features=None, thred=10000):



        labels = labels.unsqueeze(dim=1)  # 8*1*512*512
        labels_down = (F.interpolate(input=labels.double(), size=(features.shape[2], features.shape[3]),
                                     mode='nearest')).long()  # 8*1*32*32
        if bkgpaint_flag:
            labels_down_bgr_mask = ((labels_down < classes_old) & (labels_down > 0)).long()
        else:
            labels_down_bgr_mask = (labels_down == 0).long()  # 8*1*32*32

        if self.num_classes is not None and not val:

            if incremental_step != 0:
                if sequential:  # we can directly use the current groundtruth masked to consider only previous classes.
                    pseudolabel_old_down = labels_down * (labels_down < classes_old).long()
                else:  # if disjoint or overlapped: old classes are bgr in new images, hence we rely on previous model output
                    outputs_old = torch.argmax(outputs_old, dim=1,
                                               keepdim=True)  # TODO: investigate with other functions (entropy,...)
                    outputs_old_down = (
                        F.interpolate(input=outputs_old.double(), size=(features.shape[2], features.shape[3]),
                                      mode='nearest')).long()
                    pseudolabel_old_down = outputs_old_down * labels_down_bgr_mask.long()
                    # print(torch.sum(mask))
                    # print(torch.sum(pseudolabel_old_down))
                # torch.unique() 挑出tensor中的独立不重复元素
                # eg. task1 0,1,2,2  cl_present is [0,1,2]
                cl_present = torch.unique(input=pseudolabel_old_down).long()
                if cl_present[0] == 0:
                    # eg cl_present is [0,1,2], after cl_present = cl_present[1:], cl_present is [1,2]
                    cl_present = cl_present[1:]
                features_local_mean = torch.zeros([classes_old, 2048], device=self.device)
                for cl in cl_present:
                    # prototype 6*2048 classes * feat_dim
                    prototype_cl = prototypes.detach()[cl]
                    # torch.expand_as()把一个tensor变成和括号内一样形状的tensor
                    # features 8*2048*32**32
                    # pseudolabel_old_down == cl 返回的是和pseudolabel_old_down大小相同的True、False 8*1*32*32
                    # (pseudolabel_old_down == cl).expand_as(features) 返回和features大小相同的tensor 8*2048*512*512, 里面的元素都是Ture、False
                    # features[(pseudolabel_old_down == cl).expand_as(features)] 把Ture位置的数值都取出来，返回的是一个一位数组
                    # features_cl = features[(labels_down == cl).expand(-1, features.shape[1], -1, -1)].view(
                    #     features.shape[1], -1).detach()
                    #------update prototype according to the representation of student model-----
                    # current_features = features[(mask == cl).expand_as(features)].view(features.shape[1], -1).detach()
                    # ------update prototype according to the representation of teacher model-----
                    current_features = features[(pseudolabel_old_down == cl).expand_as(features)].view(features.shape[1], -1).detach()
                    features_local_mean[cl] = torch.mean(current_features.detach(), dim=-1)
                    features_cl_sum = torch.sum(current_features.detach(), dim=-1)
                    # cumulative moving average for each feature vector
                    # S_{n+f} = ( sum(x_{n+1} + ... + x_{n+f}) + n * S_n) / (n + f)
                    features_running_mean_tot_cl = (features_cl_sum + old_count_features[cl] *
                                                    old_epoch_prototype[cl, :]) / (old_count_features[cl] + current_features.shape[-1])
                    # count_features 表示到现在为止一共有多少个2048维的数组参与了当前protopype的计算。
                    prototype_sum = torch.sum(features_local_mean[cl] - prototype_cl)
                    # print('thred:', thred)
                    # print(torch.sum(old_epoch_prototype[cl, :] - features_running_mean_tot_cl))
                    if torch.abs(prototype_sum) < thred:
                        old_epoch_prototype[cl, :] = features_running_mean_tot_cl
                        old_count_features[cl] = old_count_features[cl] + current_features.shape[-1]
                    else:
                        print('!>thr')

                    # print(a)

        return old_epoch_prototype, old_count_features


    def Update_old_prototype_new_thred9(self, outputs, outputs_old, features, features_old, labels, classes_old, incremental_step,
                sequential=False, overlapped=False, loss_de_prototypes_sumafter=False, val=False, prototypes=None,
                count_features=None, old_epoch_prototype=None, bkgpaint_flag=False, old_count_features=None, thred=10000):



        labels = labels.unsqueeze(dim=1)  # 8*1*512*512
        labels_down = (F.interpolate(input=labels.double(), size=(features.shape[2], features.shape[3]),
                                     mode='nearest')).long()  # 8*1*32*32
        if bkgpaint_flag:
            labels_down_bgr_mask = ((labels_down < classes_old) & (labels_down > 0)).long()
        else:
            labels_down_bgr_mask = (labels_down == 0).long()  # 8*1*32*32

        if self.num_classes is not None and not val:

            if incremental_step != 0:
                if sequential:  # we can directly use the current groundtruth masked to consider only previous classes.
                    mask = labels_down * (labels_down < classes_old).long()
                else:  # if disjoint or overlapped: old classes are bgr in new images, hence we rely on previous model output
                    thred_mask = (torch.max(torch.softmax(outputs_old,dim=1), dim=1, keepdim=True)[0] > 0.9).long()
                    outputs_old = torch.argmax(outputs_old, dim=1,
                                               keepdim=True)  # TODO: investigate with other functions (entropy,...)
                    outputs_old_down = (
                        F.interpolate(input=outputs_old.double(), size=(features.shape[2], features.shape[3]),
                                      mode='nearest')).long()
                    thred_mask_down = (
                        F.interpolate(input=thred_mask.double(), size=(features.shape[2], features.shape[3]),
                                      mode='nearest')).long()
                    pseudolabel_old_down = outputs_old_down * labels_down_bgr_mask.long()
                    mask = pseudolabel_old_down * thred_mask_down.long()
                    # print(torch.sum(mask))
                    # print(torch.sum(pseudolabel_old_down))
                # torch.unique() 挑出tensor中的独立不重复元素
                # eg. task1 0,1,2,2  cl_present is [0,1,2]
                cl_present = torch.unique(input=mask).long()
                if cl_present[0] == 0:
                    # eg cl_present is [0,1,2], after cl_present = cl_present[1:], cl_present is [1,2]
                    cl_present = cl_present[1:]
                features_local_mean = torch.zeros([classes_old, 2048], device=self.device)
                for cl in cl_present:
                    # prototype 6*2048 classes * feat_dim
                    prototype_cl = prototypes.detach()[cl]
                    # torch.expand_as()把一个tensor变成和括号内一样形状的tensor
                    # features 8*2048*32**32
                    # pseudolabel_old_down == cl 返回的是和pseudolabel_old_down大小相同的True、False 8*1*32*32
                    # (pseudolabel_old_down == cl).expand_as(features) 返回和features大小相同的tensor 8*2048*512*512, 里面的元素都是Ture、False
                    # features[(pseudolabel_old_down == cl).expand_as(features)] 把Ture位置的数值都取出来，返回的是一个一位数组
                    # features_cl = features[(labels_down == cl).expand(-1, features.shape[1], -1, -1)].view(
                    #     features.shape[1], -1).detach()
                    #------update prototype according to the representation of student model-----
                    # current_features = features[(mask == cl).expand_as(features)].view(features.shape[1], -1).detach()
                    # ------update prototype according to the representation of teacher model-----
                    current_features = features[(mask == cl).expand_as(features)].view(features.shape[1], -1).detach()
                    features_local_mean[cl] = torch.mean(current_features.detach(), dim=-1)
                    features_cl_sum = torch.sum(current_features.detach(), dim=-1)
                    # cumulative moving average for each feature vector
                    # S_{n+f} = ( sum(x_{n+1} + ... + x_{n+f}) + n * S_n) / (n + f)
                    features_running_mean_tot_cl = (features_cl_sum + old_count_features[cl] *
                                                    old_epoch_prototype[cl, :]) / (old_count_features[cl] + current_features.shape[-1])
                    # count_features 表示到现在为止一共有多少个2048维的数组参与了当前protopype的计算。
                    prototype_sum = torch.sum(features_local_mean[cl] - prototype_cl)
                    # print('thred:', thred)
                    # print(torch.sum(old_epoch_prototype[cl, :] - features_running_mean_tot_cl))
                    if torch.abs(prototype_sum) < thred:
                        old_epoch_prototype[cl, :] = features_running_mean_tot_cl
                        old_count_features[cl] = old_count_features[cl] + current_features.shape[-1]
                    else:
                        print('!>thr')

                    # print(a)

        return old_epoch_prototype, old_count_features

    def Update_old_prototype_threshold(self, outputs, outputs_old, features, features_old, labels, classes_old, incremental_step,
                sequential=False, overlapped=False, loss_de_prototypes_sumafter=False, val=False, prototypes=None,
                count_features=None, old_epoch_prototype=None, bkgpaint_flag=False, old_count_features=None):



        labels = labels.unsqueeze(dim=1)  # 8*1*512*512
        labels_down = (F.interpolate(input=labels.double(), size=(features.shape[2], features.shape[3]),
                                     mode='nearest')).long()  # 8*1*32*32
        if bkgpaint_flag:
            labels_down_bgr_mask = ((labels_down < classes_old) & (labels_down > 0)).long()
        else:
            labels_down_bgr_mask = (labels_down == 0).long()  # 8*1*32*32

        # threshold_mask = (outputs[:, 0, ...] > 0.9)
        # threshold_mask = torch.unsqueeze(threshold_mask,dim=1)
        threshold_mask = torch.unsqueeze((outputs[:, 0, ...] > 0.9),dim=1)
        threshold_mask_down = (F.interpolate(input=threshold_mask.double(), size=(features.shape[2], features.shape[3]),
                                             mode='nearest')).long()

        if self.num_classes is not None and not val:

            if incremental_step != 0:
                if sequential:  # we can directly use the current groundtruth masked to consider only previous classes.
                    pseudolabel_old_down = labels_down * (labels_down < classes_old).long()
                else:  # if disjoint or overlapped: old classes are bgr in new images, hence we rely on previous model output
                    outputs_old = torch.argmax(outputs_old, dim=1,
                                               keepdim=True)  # TODO: investigate with other functions (entropy,...)
                    outputs_old_down = (
                        F.interpolate(input=outputs_old.double(), size=(features.shape[2], features.shape[3]),
                                      mode='nearest')).long()
                    # pseudolabel_old_down = outputs_old_down * labels_down_bgr_mask.long()
                    pseudolabel_old_down = outputs_old_down * threshold_mask_down.long()

                # torch.unique() 挑出tensor中的独立不重复元素
                # eg. task1 0,1,2,2  cl_present is [0,1,2]
                cl_present = torch.unique(input=pseudolabel_old_down).long()
                if cl_present[0] == 0:
                    # eg cl_present is [0,1,2], after cl_present = cl_present[1:], cl_present is [1,2]
                    cl_present = cl_present[1:]
                features_local_mean = torch.zeros([classes_old, 2048], device=self.device)
                for cl in cl_present:
                    # prototype 6*2048 classes * feat_dim
                    prototype_cl = prototypes.detach()[cl]
                    # torch.expand_as()把一个tensor变成和括号内一样形状的tensor
                    # features 8*2048*32**32
                    # pseudolabel_old_down == cl 返回的是和pseudolabel_old_down大小相同的True、False 8*1*32*32
                    # (pseudolabel_old_down == cl).expand_as(features) 返回和features大小相同的tensor 8*2048*512*512, 里面的元素都是Ture、False
                    # features[(pseudolabel_old_down == cl).expand_as(features)] 把Ture位置的数值都取出来，返回的是一个一位数组
                    # features_cl = features[(labels_down == cl).expand(-1, features.shape[1], -1, -1)].view(
                    #     features.shape[1], -1).detach()
                    current_features = features[(pseudolabel_old_down == cl).expand_as(features)].view(features.shape[
                                                                                                           1], -1).detach()

                    features_local_mean[cl] = torch.mean(current_features.detach(), dim=-1)
                    features_cl_sum = torch.sum(current_features.detach(), dim=-1)
                    # cumulative moving average for each feature vector
                    # S_{n+f} = ( sum(x_{n+1} + ... + x_{n+f}) + n * S_n) / (n + f)
                    features_running_mean_tot_cl = (features_cl_sum + count_features.detach()[cl] *
                                                    prototype_cl) / (count_features.detach()[cl] + current_features.shape[-1])
                    # count_features 表示到现在为止一共有多少个2048维的数组参与了当前protopype的计算。



                    old_epoch_prototype[cl, :] = features_running_mean_tot_cl
                    old_count_features[cl] = count_features[cl] + current_features.shape[-1]

                    # print(a)

        return old_epoch_prototype, old_count_features
    def Update_old_prototype_useful(self, outputs, outputs_old, features, features_old, labels, classes_old, incremental_step,
                sequential=False, overlapped=False, loss_de_prototypes_sumafter=False, val=False, prototypes=None,
                count_features=None, old_epoch_prototype=None, bkgpaint_flag=False, old_count_features=None, thred=10000):



        labels = labels.unsqueeze(dim=1)  # 8*1*512*512
        labels_down = (F.interpolate(input=labels.double(), size=(features.shape[2], features.shape[3]),
                                     mode='nearest')).long()  # 8*1*32*32
        if bkgpaint_flag:
            labels_down_bgr_mask = ((labels_down < classes_old) & (labels_down > 0)).long()
        else:
            labels_down_bgr_mask = (labels_down == 0).long()  # 8*1*32*32

        if self.num_classes is not None and not val:

            if incremental_step != 0:
                if sequential:  # we can directly use the current groundtruth masked to consider only previous classes.
                    pseudolabel_old_down = labels_down * (labels_down < classes_old).long()
                else:  # if disjoint or overlapped: old classes are bgr in new images, hence we rely on previous model output
                    outputs_old = torch.argmax(outputs_old, dim=1,
                                               keepdim=True)  # TODO: investigate with other functions (entropy,...)
                    outputs_old_down = (
                        F.interpolate(input=outputs_old.double(), size=(features.shape[2], features.shape[3]),
                                      mode='nearest')).long()
                    pseudolabel_old_down = outputs_old_down * labels_down_bgr_mask.long()
                cl_present = torch.unique(input=pseudolabel_old_down).long()
                if cl_present[0] == 0:
                    cl_present = cl_present[1:]
                features_local_mean = torch.zeros([classes_old, 2048], device=self.device)
                for cl in cl_present:
                    prototype_cl = prototypes.detach()[cl]
                    current_features = features[(features_local_mean == cl).expand_as(features)].view(features.shape[1], -1).detach()
                    features_local_mean[cl] = torch.mean(current_features.detach(), dim=-1)
                    features_cl_sum = torch.sum(current_features.detach(), dim=-1)
                    features_running_mean_tot_cl = (features_cl_sum + old_count_features[cl] *
                                                    old_epoch_prototype[cl, :]) / (old_count_features[cl] + current_features.shape[-1])
                    prototype_sum = torch.sum(features_local_mean[cl] - prototype_cl)
                    if torch.abs(prototype_sum) < thred:
                        old_epoch_prototype[cl, :] = features_running_mean_tot_cl
                        old_count_features[cl] = old_count_features[cl] + current_features.shape[-1]
                    else:
                        print('!>thr')

                    # print(a)

        return old_epoch_prototype, old_count_features

    def Update_background_prototype(self, outputs, outputs_old, features, features_old, labels, classes_old,
                                       incremental_step,
                                       sequential=False, overlapped=False, loss_de_prototypes_sumafter=False, val=False,
                                       prototypes=None,
                                       count_features=None, old_epoch_prototype=None, bkgpaint_flag=False,
                                       old_count_features=None, thred=10000):

        labels = labels.unsqueeze(dim=1)  # 8*1*512*512
        labels_down = (F.interpolate(input=labels.double(), size=(features.shape[2], features.shape[3]),
                                     mode='nearest')).long()  # 8*1*32*32


        features_local_mean = torch.zeros([1, 2048], device=self.device)
        if features[(labels_down == 0).expand(-1, features.shape[1], -1, -1)].shape[0] % 2048 != 0:
            pass
        else:

            features_cl = features[(labels_down == 0).expand(-1, features.shape[1], -1, -1)].view(features.shape[1],
                                                                                               -1).detach()
            features_local_mean[0] = torch.mean(features_cl.detach(), dim=-1)
            features_cl_sum = torch.sum(features_cl.detach(), dim=-1)
            features_running_mean_tot_cl = (features_cl_sum + count_features.detach()[0] *
                                        prototypes.detach()[0]) \
                                       / (count_features.detach()[0] + features_cl.shape[-1])
            count_features[0] += features_cl.shape[-1]
            prototypes[0] = features_running_mean_tot_cl

        return prototypes, count_features


import os
import numpy as np
import matplotlib.pyplot as plt

from torch.autograd import Variable as V
from datasets.data_utils import CDDataAugmentation
import torchvision.transforms.functional as TF
from models.networks import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
from utils import de_norm
import utils
import cv2
from tqdm import tqdm
from osgeo import gdal,ogr,osr


# Decide which device we want to run on
# torch.cuda.current_device()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CDEvaluator():

    def __init__(self, args, dataloader):

        self.dataloader = dataloader

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        print(self.device)

        # define some other vars to record the training states
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)


        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0

        self.steps_per_epoch = len(dataloader)

        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)


    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device)

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)


    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis


    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_pred.detach()
        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _collect_running_batch_states(self):

        running_acc = self._update_metric()

        m = len(self.dataloader)

        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' %\
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)

        # if np.mod(self.batch_id, 100) == 1:
        vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
        vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))

        vis_pred = utils.make_numpy_grid(self._visualize_pred())

        vis_gt = utils.make_numpy_grid(self.batch['L'])
        vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
        vis = np.clip(vis, a_min=0.0, a_max=1.0)
        file_name = os.path.join(
            self.vis_dir, 'eval_' + str(self.batch_id)+'.jpg')
        plt.imsave(file_name, vis)


    def _collect_epoch_states(self):

        scores_dict = self.running_metric.get_scores()

        np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)

        self.epoch_acc = scores_dict['mf1']

        with open(os.path.join(self.checkpoint_dir, '%s.txt' % (self.epoch_acc)),
                  mode='a') as file:
            pass

        message = ''
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write('%s\n' % message)  # save the message

        self.logger.write('\n')

    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        self.G_pred = self.net_G(img_in1, img_in2)[-1]

    def eval_models(self,checkpoint_name='best_ckpt.pt'):

        self._load_checkpoint(checkpoint_name)

        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()

        # Iterate over data.
        for self.batch_id, batch in enumerate(self.dataloader, 0):
            with torch.no_grad():
                self._forward_pass(batch)
            self._collect_running_batch_states()
        self._collect_epoch_states()

    def block_gdal_input(self, img, img_size, crop=512, pad=0): # gdal分块读取
        [img_width, img_height] = img_size
        x_height = x_width = crop
        crop_width = x_width - 2 * pad
        crop_height = x_height - 2 * pad

        numBand = 3
        # numBand = img.RasterCount
        num_Xblock = img_width // crop_width
        x_start, x_end = [], []
        x_start.append(0)
        for i in range(num_Xblock):
            xs = crop_width * (i + 1) - pad
            xe = crop_width * i + x_width - pad
            if (i == num_Xblock - 1):
                xs = img_width - crop_width - pad
                xe = min(xe, img_width)
            x_start.append(xs)
            x_end.append(xe)
        x_end.append(img_width)

        num_Yblock = img_height // crop_height
        y_start, y_end = [], []
        y_start.append(0)
        for i in range(num_Yblock):
            ys = crop_height * (i + 1) - pad
            ye = crop_height * i + x_height - pad
            if (i == num_Yblock - 1):
                ys = img_height - crop_height - pad
                ye = min(ye, img_height)
            y_start.append(ys)
            y_end.append(ye)
        y_end.append(img_height)

        if img_width % crop_width > 0:
            num_Xblock = num_Xblock + 1
        if img_height % crop_height > 0:
            num_Yblock = num_Yblock + 1
        for i in range(num_Yblock):
            for j in range(num_Xblock):
                [x0, x1, y0, y1] = [x_start[j], x_end[j], y_start[i], y_end[i]]

                feature = np.zeros(np.append([y1 - y0, x1 - x0], numBand), np.float32)
                for ii in range(numBand):
                    floatData = np.array(img.GetRasterBand(ii + 1).ReadAsArray(x0, y0, x1 - x0, y1 - y0),dtype=np.float32)
                    # floatData = np.array(img.GetRasterBand(4-ii).ReadAsArray(x0,y0,x1-x0,y1-y0))

                    feature[..., ii] = (floatData/255-0.5)/0.5
                    # feature[..., ii] = floatData

                if (i == 0):
                    feature_pad = cv2.copyMakeBorder(feature,
                                                     pad, x_height - pad - feature.shape[0],
                                                     0, 0, cv2.BORDER_REFLECT_101)
                else:
                    feature_pad = cv2.copyMakeBorder(feature,
                                                     0, x_height - feature.shape[0],
                                                     0, 0, cv2.BORDER_REFLECT_101)
                if (j == 0):
                    feature_pad = cv2.copyMakeBorder(feature_pad,
                                                     0, 0, pad, x_width - pad - feature_pad.shape[1],
                                                     cv2.BORDER_REFLECT_101)
                else:
                    feature_pad = cv2.copyMakeBorder(feature_pad,
                                                     0, 0, 0, x_width - feature_pad.shape[1],
                                                     cv2.BORDER_REFLECT_101)

                yield feature_pad, [x0, x1, y0, y1]

    def pred_gdal_blocks_write(self, img_pathA, img_pathB,out_path=''):
        self._load_checkpoint()

        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()

        #logger.info('predicting %s' % img_pathA)

        batch_size = 1
        pad = 16
        x_width = 256
        x_height = 256
        crop_width = x_width - 2 * pad
        crop_height = x_height - 2 * pad
        datasetname = gdal.Open(img_pathA, gdal.GA_ReadOnly)
        # datasetname = reproject_dataset(img_path,5500,5500)
        if datasetname is None:
            print('Could not open %s' % img_pathA)
        img_width = datasetname.RasterXSize
        img_height = datasetname.RasterYSize
        imageSize = [img_width, img_height]
        nBand = datasetname.RasterCount

        datasetname2 = gdal.Open(img_pathB, gdal.GA_ReadOnly)
        if datasetname2 is None:
            print('Could not open %s' % img_pathB)
        img_width2 = datasetname2.RasterXSize
        img_height2 = datasetname2.RasterYSize

        if img_width != img_width2 or img_height != img_height2:
            print("范围不一致")
            return

        driver = gdal.GetDriverByName('GTiff')
        if out_path == '':
            out_path = img_pathA.rsplit('.', 1)[0] + '_res.tif'
        outRaster = driver.Create(out_path, img_width, img_height, 1, gdal.GDT_Byte)
        outband = outRaster.GetRasterBand(1)
        outRaster.SetGeoTransform(datasetname.GetGeoTransform())
        outRaster.SetProjection(datasetname.GetProjection())

        num_Xblock = img_width // crop_width
        if img_width % crop_width > 0:
            num_Xblock += 1
        num_Yblock = img_height // crop_height
        if img_height % crop_height > 0:
            num_Yblock += 1
        i = 0
        blocks = num_Xblock * num_Yblock
        # mask = np.zeros([batch_size, img_height, img_width],dtype=np.float32)
        input_gen = self.block_gdal_input(datasetname, imageSize, x_width, pad)
        input_gen2 = self.block_gdal_input(datasetname2, imageSize, x_width, pad)
        for i in tqdm(range(blocks)):
            imgA, xy = next(input_gen)
            imgB, xyB = next(input_gen2)
            if (xy[0] > 0):
                xs = xy[0] + pad
            else:
                xs = xy[0]

            if (xy[2] > 0):
                ys = xy[2] + pad
            else:
                ys = xy[2]
            #if np.max(imgA[pad: pad + crop_height, pad: pad + crop_width]) < 5:
                #predictions = np.zeros([batch_size, x_height, x_width])
            #else:
            imgs = []
            imgs.append(imgA)
            imgs = np.array(imgs)
            # imgs = imgs[:,np.newaxis]
            # np.squeeze(imgs)
            # np.expand_dims(imgs,axis=1)
            imgs = imgs.transpose(0, 3, 1, 2)
            imgs = V(torch.Tensor(np.array(imgs, np.float32)).to(self.device))

            # imgs = TF.normalize(imgs, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            # imgs = imgs.resize(1,3,256,256)

            imgs2 = []
            imgs2.append(imgB)
            imgs2 = np.array(imgs2)
            # imgs = imgs[:,np.newaxis]
            # np.squeeze(imgs)
            # np.expand_dims(imgs,axis=1)
            imgs2 = imgs2.transpose(0, 3, 1, 2)
            imgs2 = V(torch.Tensor(np.array(imgs2, np.float32)).to(self.device))
            # imgs2 = TF.to_tensor(np.array(imgs2[0], np.float32)).to(self.device)
            # imgs2 = TF.normalize(imgs2, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            # imgs2 = imgs2.resize(1, 3, 256, 256)

            predictions = self.net_G(imgs, imgs2)[-1]
            # predictions = predictions.numpy()
            predictions = torch.argmax(predictions, dim=1, keepdim=True)
            # print(predictions)
            predictions = np.array(predictions)[0][0]

            prediction = predictions[pad: pad + crop_height,
                         pad: pad + crop_width]

            outband.WriteArray((prediction * 255).astype(np.int), xs, ys)
            # mask[0,ys: ys+crop_height,\
            #    xs : xs+crop_width] = prediction.astype(np.float32)

            # if(i%num_Xblock==0):
            #    y=i//num_Xblock
            #    logger.info('predicting data: [{}{}] {}%'.\
            #                 format('=' * (y+1),
            #                        ' ' * (num_Yblock - y-1),
            #                        100 * (y+1)/num_Yblock))
            outband.FlushCache()
            # sys.stdout.flush()
            # i=i+1
        datasetname = None
        datasetname2 = None
        outRaster = None
        return  # np.squeeze(mask*255).astype(np.int)

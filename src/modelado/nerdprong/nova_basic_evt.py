#######################################################################
#
# Contains a config and dataset for running, testing, and training
# on NOvA data contained in hdf5 files.
#
# Also included are examples for viewing data, testing a model, and
# training.
#
#######################################################################

import os
import numpy as np
import h5py
import cv2
import itertools

import tensorflow as tf
from tensorflow.python.framework import graph_util
from keras import backend as K
from keras.utils import to_categorical

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Create a Config class
class novaConfig(Config):
    NAME = "NOvA"

#    NUM_CLASSES = 1 + 6
    NUM_CLASSES = 1 + 1

    GPU_COUNT = 1
    IMAGES_PER_GPU = 10

    IMAGE_RESIZE_MODE = 'pad64'
    IMAGE_MIN_DIM = 384
    IMAGE_MAX_DIM = 384

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    TRAIN_ROIS_PER_IMAGE = 100
    MAX_GT_INSTANCES = 50

    STEPS_PER_EPOCH = 4000
    VALIDATION_STEPS = 100

    IMAGE_CHANNEL_COUNT = 3
    MEAN_PIXEL = np.array([0,0,0])

    MASK_THRESHOLD = 0.1

    DETECTION_MIN_CONFIDENCE = 0
###################################################

# A class for loading nova images from numpy arrays in an hdf5 file
class novaDataset(utils.Dataset):
    # Load the nova classes, and set the the image ids
    def load_nova(self,data_path,cat):
        if type(cat) is not list: cat = [cat]

        print('Loading data from',data_path)

        n = 0
        for c in cat:
            print('Loading',c,'data')
            path = os.path.join(data_path,c)
            files = os.listdir(path)
            for f in files:
                self.add_image('NOvA',n,os.path.join(path,f),name=f[:-3])
                n+=1

        print('Loaded',n,'images from',cat,'data.')

        self.add_class("NOvA", 1, "Neutrino")
        # self.add_class("NOvA", 1, "Electron")
        # self.add_class("NOvA", 2, "Muon")
        # self.add_class("NOvA", 3, "Proton")
        # self.add_class("NOvA", 4, "Photon")
        # self.add_class("NOvA", 5, "Pion")
        # self.add_class("NOvA", 6, "Other")

    # Useful for visualizing results
    def load_image_BonW(self,image_id):
        hf = h5py.File(self.image_info[image_id]['path'],'r')
        image_array = hf['image'][:]
        hf.close()

        image = np.zeros(image_array.shape+(3,))

        image[image_array > 0] = [255,255,255]

        return self.pm_resize(image)

    # image made from the dep*10^7
    def load_image(self,image_id):
        hf = h5py.File(self.image_info[image_id]['path'],'r')
        image_array = hf['image'][:]
        time_array  = hf['time'][:]
        hf.close()

        image_array = image_array/image_array.max()

        # I think events can start up to 50 usec early
        t0 = time_array.min()
        tf = 550.0

        if t0 < 0:
            time_array = time_array + (t0 * -1)

        time_array = time_array/tf

        blue = np.zeros_like(image_array)
        blue[image_array > 0] = 0.5

        image = np.zeros(image_array.shape+(3,))
        image[:,:,2] = image_array
        image[:,:,1] = blue
        image[:,:,0] = time_array

        return self.pm_resize(image)

    def pm_resize(self,image,n=3):
        return image.repeat(n,axis=0).repeat(n,axis=1)

    # Color image corresponding to the label of each pixel
    def load_label(self,image_id):
        hf = h5py.File(self.image_info[image_id]['path'],'r')
        label_array = hf['label'][:]
        hf.close()

        arrLabCol = np.zeros((len(label_array),len(label_array[0]),3),dtype=np.uint8)
        for x in range(len(label_array)):
            for y in range(len(label_array[0])):
                label = label_array[x][y]
                if label == 1:
                    arrLabCol[x,y] = [248,19,35]    # electron - kRed-4      CVN PID = 1
                if label == 2:
                    arrLabCol[x,y] =  [48,129,233]  # muon     - kAzure+1    CVN PID = 2
                if label == 3:
                    arrLabCol[x,y] =  [187,0,250]   # Proton   - kViolet     CVN PID = 3
                if label == 4:
                    arrLabCol[x,y] =  [37,255,97]   # Neutron  - kTeal-3     CVN PID = 4
                if label == 5:
                    arrLabCol[x,y] =  [252,72,210]  # Pi+-     - kPink+6     CVN PID = 5
                if label == 6:
                    arrLabCol[x,y] =  [253,135,11]  # Pi0      - kOrange-3   CVN PID = 6
                if label == 7:
                    arrLabCol[x,y] = [253,254,147]  # Gamma    - kYellow+9   CVN PID = 7
                if label in [8,9,10,11]:
                    arrLabCol[x,y] =  [190,190,190] # Other    - kGray       CVN PID = 8,9,10

        return arrLabCol

    # Color image corresponding to the object instance of each pixel
    def load_object(self,image_id):
        hf = h5py.File(self.image_info[image_id]['path'],'r')
        object_array = hf['object'][:]
        hf.close()

        # Convert the object count to a rainbow scale
        # I'm very sorry Fer...
        arrObjCol = np.zeros((len(object_array),len(object_array[0]),3),dtype=np.uint8)

        max_object = np.max(object_array)

        for x in range(len(object_array)):
            for y in range(len(object_array[0])):
                scale = float(object_array[x][y])/max_object
                a = (1-scale)/0.25
                intp  = int(a)
                fracp = int(255*(a-intp))

                if intp == 0:
                    arrObjCol[x][y] = [255,fracp,0]
                if intp == 1:
                    arrObjCol[x][y] = [255-fracp,255,0]
                if intp == 2:
                    arrObjCol[x][y] = [0,255,fracp]
                if intp == 3:
                    arrObjCol[x][y] = [0,255-fracp,255]
                if intp == 4:
                    arrObjCol[x][y] = [0,0,0]

        return arrObjCol

    # Create the binary mask from the label and object instance of each pixel
    def load_mask(self,image_id):
        hf = h5py.File(self.image_info[image_id]['path'],'r')
        object_array = hf['object'][:]
        label_array = hf['label'][:]
        hf.close()

        max_object = np.max(object_array)
        # Makes binary mask with shape (image width, image height, # objects)
        # with elements = 1 in pixels with the corresponding object present
        mask = to_categorical(object_array, num_classes=max_object+1).astype(dtype=np.bool)
        # Throw away the mask for object 0
        mask = mask[:,:,1:]
        # Array of object labels
        label = np.ones(max_object,dtype=np.int32)

        # for i in range(max_object):
        #     uniq = np.unique(label_array[object_array == i+1])
        #     id = np.min(uniq)
        #     if id==7:
        #         id=4
        #     elif id>7 or id==4:
        #         id=6
        #     label[i] = id

        return self.pm_resize(mask), label

###################################################

def subsample(mask, bbox=None, image=None):
    rows, cols, _ = mask.shape
    mask = mask.reshape(rows//3,3,cols//3,3,-1)
    mask = mask.any(axis=(1,3))

    if image is not None:
        image = image.reshape(rows//3,3,cols//3,3,-1)
        image = image.sum(axis=(1,3))/9

    if bbox is not None:
        bbox = bbox/3
        bbox = np.asarray(list(map(lambda x: [int(x[0]), int(x[1]), int(x[2]+0.5), int(x[3]+0.5)], bbox)))

    return mask, bbox, image

# Prediction wrapper with post-processing
def nova_detect(model, image):
    def dist(r,c,mask):
        row,col = np.where(mask>0)
        return min( map(lambda pt: np.sqrt((pt[0]-r)**2 + (pt[1]-c)**2), zip(row,col)) )

    # Network Prediction
    results = model.detect([image], verbose=0)
    r = results[0]

    # Suppress Backgroun
    r['masks'][image[:,:,2] == 0] = np.zeros(r['class_ids'].shape)

    # Downsample to hit level
    r['masks'], r['rois'], image = subsample(r['masks'], r['rois'], image)

    # Remove false positives
    bad = []
    for i,j in itertools.combinations(np.arange(r['class_ids'].shape[0]),2):
        # Grab the pair of masks and compute area
        m1,m2  = r['masks'][:,:,i], r['masks'][:,:,j]
        a1,a2 = np.sum(m1), np.sum(m2)
        ac = np.sum(m1 & m2)

        # If cluster is direct subset of another and of sufficient size...
        if (ac>=a1-1 and ac>=0.1*a2) or (ac>=a2-1 and ac>=0.1*a1):
            if a1>=a2-1 and a2>=a1-1:
                # Grab the largest score of identical clusters
                if r['scores'][i] > r['scores'][j]:
                    bad.append(j)
                else:
                    bad.append(i)
            else:
                # or the larger of two clusters
                if a1 > a2:
                    bad.append(j)
                else:
                    bad.append(i)

    # and delete them
    r['masks']     = np.delete(r['masks'],bad,axis=2)
    r['rois']      = np.delete(r['rois'],bad,axis=0)
    r['class_ids'] = np.delete(r['class_ids'],bad)
    r['scores']    = np.delete(r['scores'],bad)

    # Identify unclustered hits
    row,col = np.where((image[:,:,2] > 0) & ~r['masks'].any(axis=2))
    for row,col in zip(row,col):
        cont=[]
        for n, [y1, x1, y2, x2] in enumerate(r['rois']):
            if row>=y1 and row<=y2 and col>=x1 and col<=x2:
                cont.append(n)
        if len(cont)==1:
            if (dist(row,col,r['masks'][:,:,cont[0]]) < 10 and r['class_ids'][cont[0]] in (1,7)) or \
                dist(row,col,r['masks'][:,:,cont[0]]) < 3.5:
                r['masks'][row,col,cont[0]] = 1
        elif len(cont)>1:
            m=256
            for id in cont:
                d = dist(row,col,r['masks'][:,:,id])
                if d < m:
                    m = d
                    i = id
            r['masks'][row,col,i] = 1

    return r


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='For Nova Data!')

    parser.add_argument('command',metavar='<command>',help='train or test on Nova data')
    parser.add_argument('--dataset', required=False,
                        default='data/sample',
                        help='Directory of the nova h5 files.')
    parser.add_argument('--model', required=False,
                        default='',
                        help='Weights to load')
    parser.add_argument('--logdir',required=False,
                        default=os.path.join(os.getcwd(), 'logs'),
                        help='Output log files directory')
    parser.add_argument('--limit',required=False,
                        default=0, type=int,
                        help='Number of images to evaluate or predict')

    args=parser.parse_args()

    # Predict some images with a model and save
    if args.command == 'pred':
        dataset = novaDataset()
        dataset.load_nova(args.dataset,'eval')
        dataset.prepare()

        class inferNovaConfig(novaConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = inferNovaConfig()
        model = modellib.MaskRCNN(mode="inference", model_dir=args.logdir, config=config)

        if args.model != '':
            print('Loading weights from',args.model)
            model.load_weights(args.model,by_name=True)

        class_names = dataset.class_names

        image_ids = dataset.image_ids
        if not args.limit == 0:
            #image_ids = np.random.choice(image_ids,args.limit)
            image_ids = np.arange(args.limit)

        for i in image_ids:
            image = dataset.load_image(i)

            #r = nova_detect(model, image)
            results = model.detect([image], verbose=0)
            r = results[0]

            bandw = dataset.load_image_BonW(i)

            visualize.display_instances(bandw, r['rois'], r['masks'], r['class_ids'], class_names, \
                            r['scores'],title=dataset.image_info[i]['name'][8:],save=1)

    # Show the ground truth images
    if args.command == 'gt':
        dataset = novaDataset()
        dataset.load_nova(args.dataset,'eval')
        dataset.prepare()

        image_ids = dataset.image_ids
        if not args.limit == 0:
            #image_ids = np.random.choice(image_ids,args.limit)
            image_ids = np.arange(args.limit)

        for i in image_ids:
            image = dataset.load_image(i)
            mask, class_ids = dataset.load_mask(i)
            bbox = utils.extract_bboxes(mask)

            bandw = dataset.load_image_BonW(i)

            visualize.display_instances(bandw, bbox, mask, class_ids, \
                            dataset.class_names, title=dataset.image_info[i]['name'][8:]+'_gt',save=1)

    # Train a network
    if args.command == 'train':
        config = novaConfig()
        config.display()

        dataset_train = novaDataset()
        dataset_train.load_nova(args.dataset,'train')
        dataset_train.prepare()

        dataset_eval = novaDataset()
        dataset_eval.load_nova(args.dataset,'eval')
        dataset_eval.prepare()

        model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logdir)

        if args.model != '':
            print('Loading weights from',args.model)
            model.load_weights(args.model,by_name=True)
        '''
        # Coco training schedule:
        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_eval,
                    learning_rate=config.LEARNING_RATE,
                    epochs=1,
                    layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_eval,
                    learning_rate=config.LEARNING_RATE,
                    epochs=3,
                    layers='4+')

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_eval,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=4,
                    layers='all')
        '''
        # Simple training
        model.train(dataset_train, dataset_eval,
                    learning_rate=config.LEARNING_RATE,
                    epochs=250,
                    layers='all')

    # Evaluate a model by computing mAP
    if args.command == 'eval':
        dataset = novaDataset()
        dataset.load_nova(args.dataset,'eval')
        dataset.prepare()

        class inferNovaConfig(novaConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = inferNovaConfig()

        model = modellib.MaskRCNN(mode="inference", model_dir=args.logdir, config=config)

        if args.model != '':
            print('Loading weights from',args.model)
            model.load_weights(args.model,by_name=True)

        image_ids = dataset.image_ids
        if not args.limit == 0:
            image_ids = np.random.choice(image_ids,args.limit)

        # Compute VOC-Style mAP @ IoU=0.5
        APs = []
        for image_id in image_ids:
            # Load image and ground truth data
            image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config,
                                image_id, use_mini_mask=False)
            molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
            # Run object detection
            r = nova_detect(model, image)

            # Compute AP
            AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                            r["rois"], r["class_ids"], r["scores"], r['masks'])
            APs.append(AP)

        print("mAP: ", np.mean(APs))

    # Compute some model metris
    if args.command == 'metric':
        dataset = novaDataset()
        dataset.load_nova(args.dataset,['fluxswap/eval','nonswap/eval'])
        dataset.prepare()

        class inferNovaConfig(novaConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = inferNovaConfig()
        model = modellib.MaskRCNN(mode="inference", model_dir=args.logdir, config=config)

        if args.model != '':
            print('Loading weights from',args.model)
            model.load_weights(args.model,by_name=True)

        class_names = dataset.class_names

        image_ids = dataset.image_ids
        if not args.limit == 0:
            #image_ids = np.random.choice(image_ids,args.limit)
            #image_ids = np.arange(args.limit)
            image_ids = np.arange(0,3000,10)

        mult_res = np.array([])
        APs = []
        true_purs = []
        true_effs = []
        pred_purs = []
        pred_effs = []

        for image_id in image_ids:
            image = dataset.load_image(image_id)

            gt_mask, gt_class_id = dataset.load_mask(image_id)
            gt_mask,_,_ = subsample(gt_mask)
            gt_bbox = utils.extract_bboxes(gt_mask)

            r = nova_detect(model,image)

            bandw = dataset.load_image_BonW(i)

            visualize.display_instances(bandw, r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'],title=dataset.image_info[i]['name'][8:]+'_triple',save=1)

            AP, precisions, recalls, overlaps =\
                utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                            r['rois'], r['class_ids'], r['scores'], r['masks'])
            APs.append(AP)

            truemult = gt_class_id.shape[0]
            predmult = r['class_ids'].shape[0]

            mult_res = np.append(mult_res, [predmult - truemult])

            gt_match, pred_match, overlaps = utils.compute_matches(gt_bbox, gt_class_id, gt_mask,
                                r['rois'], r['class_ids'], r['scores'], r['masks'])

            for gt_id, pred_id in enumerate(gt_match.astype(int)):
                if pred_id==-1: continue
                pred_area, gt_area = np.sum(r['masks'][:,:,pred_id]), np.sum(gt_mask[:,:,gt_id])
                diff = r['masks'][:,:,pred_id].astype(int) - gt_mask[:,:,gt_id].astype(int)

                ones = np.count_nonzero(diff == 1)
                negs = np.count_nonzero(diff == -1)

                pur = (pred_area-ones)/pred_area
                eff = (gt_area-negs)/gt_area

                true_purs.append(pur)
                true_effs.append(eff)

            for pred_id, gt_id in enumerate(pred_match.astype(int)):
                if gt_id==-1: continue
                pred_area, gt_area = np.sum(r['masks'][:,:,pred_id]), np.sum(gt_mask[:,:,gt_id])
                diff = r['masks'][:,:,pred_id].astype(int) - gt_mask[:,:,gt_id].astype(int)

                ones = np.count_nonzero(diff == 1)
                negs = np.count_nonzero(diff == -1)

                pur = (pred_area-ones)/pred_area
                eff = (gt_area-negs)/gt_area

                pred_purs.append(pur)
                pred_effs.append(eff)

        import matplotlib.pyplot as plt

        plt.figure(1)
        plt.hist(mult_res, 30, (-15,15))
        plt.xlabel('Reco-True Multiplicity')
        plt.ylabel('Events')
        plt.savefig('multres_all.png')

        plt.figure(2)
        plt.hist(true_purs, 25, (0,1))
        plt.xlabel('Purity')
        plt.ylabel('True Particles')
        plt.savefig('true_purity.png')

        plt.figure(3)
        plt.hist(true_effs, 25, (0,1))
        plt.xlabel('Efficiency')
        plt.ylabel('True Particles')
        plt.savefig('true_efficiency.png')

        plt.figure(4)
        plt.hist(pred_purs, 25, (0,1))
        plt.xlabel('Purity')
        plt.ylabel('Predicted Particles')
        plt.savefig('pred_purity.png')

        plt.figure(5)
        plt.hist(pred_effs, 25, (0,1))
        plt.xlabel('Efficiency')
        plt.ylabel('Predicted Particles')
        plt.savefig('pred_efficiency.png')

        plt.figure(6)
        plt.hist(APs, 25, (0,1))
        plt.xlabel('Average Precision')
        plt.ylabel('Events')
        plt.savefig('ap.png')

        plt.text(0.5, 0.65, datamu, color='k', fontsize=12, horizontalalignment='left', verticalalignment='center', \
            transform=plt.gca().transAxes)

    if args.command == 'save':
        class inferNovaConfig(novaConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = inferNovaConfig()
        model = modellib.MaskRCNN(mode="inference", model_dir=args.logdir, config=config)

        if args.model != '':
            print('Loading weights from',args.model)
            model.load_weights(args.model,by_name=True)

        model_keras = model.keras_model
        K.set_learning_phase(0)
        num_output = 7

        pred_node_names = ["detections", "mrcnn_class", "mrcnn_bbox", "mrcnn_mask",
                            "rois", "rpn_class", "rpn_bbox"]
        pred_node_names = ["output_" + name for name in pred_node_names]
        pred = [tf.identity(model_keras.outputs[i], name = pred_node_names[i])
                    for i in range(num_output)]
        sess = K.get_session()

        od_graph_def = graph_util.convert_variables_to_constants(sess,
                                                         sess.graph.as_graph_def(),
                                                         pred_node_names)

        frozen_graph_path = 'model.pb'
        with tf.gfile.GFile(frozen_graph_path, 'wb') as f:
            f.write(od_graph_def.SerializeToString())

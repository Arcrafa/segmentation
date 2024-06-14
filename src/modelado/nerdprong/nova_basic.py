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
from sklearn.model_selection import train_test_split

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Create a Config class
class novaConfig(Config):
    NAME = "NOvA"

    NUM_CLASSES = 1 + 6

    GPU_COUNT = 1
    IMAGES_PER_GPU = 12

    LEARNING_RATE = 0.05

    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 2.
    }

    BACKBONE = 'resnet50'

    IMAGE_RESIZE_MODE = 'pad64'
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    PIXEL_SCALING = 0.05

    TRAIN_ROIS_PER_IMAGE = 100
    MAX_GT_INSTANCES = 50

    USE_MINI_MASK = False

    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 10

    TEST_FRACTION = 0.1

    IMAGE_CHANNEL_COUNT = 2
    MEAN_PIXEL = np.array([0,0])

    MASK_THRESHOLD = 0.1

    DETECTION_MIN_CONFIDENCE = 0

    POST_NMS_ROIS_INFERENCE = 250
###################################################

# A class for loading nova images from numpy arrays in an hdf5 file
class novaDataset(utils.Dataset):
    # Load the nova classes, and set the the image ids
    def load_nova(self,filelist):
        n = 0 # keep tracking of the total number of events
        for f in filelist:
            h5 = h5py.File(f,'r')
            labs = h5['label'][:]
            Es = h5['energy'][:]
            
            for i,(lab,E) in enumerate(zip(labs,Es)):
                if lab == 15: # omit cosmics
                    continue
                if E > 10: # omit real crazy events
                    continue
                # use both x and y views together
                self.add_image('NOvA',n,f,idx=i,view='X')
                n+=1
                self.add_image('NOvA',n,f,idx=i,view='Y')
                n+=1
            h5.close()

        print('Loaded',n,'images.')

        # Same class list as prong CVN
        self.add_class("NOvA", 1, "Electron")
        self.add_class("NOvA", 2, "Muon")
        self.add_class("NOvA", 3, "Proton")
        self.add_class("NOvA", 4, "Photon")
        self.add_class("NOvA", 5, "Pion")
        self.add_class("NOvA", 6, "Other")

    # Transform the 1D pixelmap array to one of the views
    def transform_pm(self, pm, view):
        # if pixel map geometry changes so will this
        newpm = pm.reshape(2,100,80)
        if view=='X':
            return newpm[0]
        else:
            return newpm[1]

    # Useful for visualizing results
    def load_image_BonW(self,image_id):
        info = self.image_info[image_id]
        hf = h5py.File(info['path'],'r')
        pm = hf['cvnmap'][info['idx']]
        pm = self.transform_pm(pm, info['view'])
        hf.close()

        image = np.zeros(pm.shape+(3,))

        image[pm > 0] = [255,255,255]

        return image

    # Pixel map of the hits
    def load_image(self,image_id):
        info = self.image_info[image_id]
        hf = h5py.File(info['path'],'r')
        pm = hf['cvnmap'][info['idx']]
        pm = self.transform_pm(pm, info['view'])
        hf.close()

        # second channel is boolean for hit or no hit
        blue = np.zeros_like(pm)
        blue[pm > 0] = 255
        
        image = np.zeros(pm.shape+(2,))
        image[:,:,0] = pm
        image[:,:,1] = blue

        return self.pm_resize(image)

    # Turn each cell into 9 (3x3) and remove the last 15 planes.
    # Keeps input size under 256x256
    def pm_resize(self,image,n=3):
        return image[:-15,:].repeat(n,axis=0).repeat(n,axis=1)

    # Create the binary mask from the label and object instance of each pixel
    def load_mask(self,image_id):
        info = self.image_info[image_id]
        hf = h5py.File(info['path'],'r')
        object_array = hf['cvnobjmap'][info['idx']]
        object_array = self.transform_pm(object_array, info['view'])
        label_array = hf['cvnlabmap'][info['idx']]
        label_array = self.transform_pm(label_array, info['view'])
        hf.close()

        # Some objects never make it as a max contributor to any hits
        # Remove them
        max_object = np.max(object_array)
        for i in range(max_object, 0, -1):
            if object_array[object_array==i].shape[0]==0:
                object_array[object_array >= i] -= 1

        max_object = np.max(object_array)
        mask = to_categorical(object_array, num_classes=max_object+1).astype(dtype=np.bool)
        mask = mask[:,:,1:]
        label = np.zeros(max_object,dtype=np.int32)

        for i in range(max_object):
            # Sometimes we get an object with different labels at each hit
            labs = label_array[object_array == i+1]
            # Use the most common
            id = np.argmax(np.bincount(labs))

            # Redefine labels
            if id==7:
                id=4
            elif id>7 or id==4:
                id=6
            label[i] = id

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
# Similar to what is done in the art module
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

            r = nova_detect(model, image)

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
        """
        FHCFiles = os.listdir(os.path.join(args.dataset,'FHC'))
        FHCFlux  = [os.path.join(args.dataset,'FHC',f) for f in FHCFiles if 'fluxswap' in f]
        FHCNon   = [os.path.join(args.dataset,'FHC',f) for f in FHCFiles if 'nonswap' in f]
        FHCTau   = [os.path.join(args.dataset,'FHC',f) for f in FHCFiles if 'tau' in f]

        fftrain,ffeval = train_test_split(FHCFlux, test_size=config.TEST_FRACTION)
        fntrain,fneval = train_test_split(FHCNon,  test_size=config.TEST_FRACTION)
        fttrain,fteval = train_test_split(FHCTau,  test_size=config.TEST_FRACTION)

        RHCFiles = os.listdir(os.path.join(args.dataset,'RHC'))
        RHCFlux  = [os.path.join(args.dataset,'RHC',f) for f in RHCFiles if 'fluxswap' in f]
        RHCNon   = [os.path.join(args.dataset,'RHC',f) for f in RHCFiles if 'nonswap' in f]
        RHCTau   = [os.path.join(args.dataset,'RHC',f) for f in RHCFiles if 'tau' in f]

        rftrain,rfeval = train_test_split(RHCFlux, test_size=config.TEST_FRACTION)
        rntrain,rneval = train_test_split(RHCNon,  test_size=config.TEST_FRACTION)
        rttrain,rteval = train_test_split(RHCTau,  test_size=config.TEST_FRACTION)
        """
        files = os.listdir(args.dataset)
        files   = [os.path.join(args.dataset,f) for f in files ]
        ftrain,feval = train_test_split(files, test_size=0.2)
        
        dataset_train = novaDataset()
        dataset_train.load_nova(ftrain)
        dataset_train.prepare()

        dataset_eval = novaDataset()
        dataset_eval.load_nova(feval)
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
                    epochs=5,
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

    if args.command == 'viewmodel':
        class inferNovaConfig(novaConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = inferNovaConfig()
        model = modellib.MaskRCNN(mode="inference", model_dir=args.logdir, config=config)
        model = model.keras_model

        #model.summary()

        print('Model has ',model.count_params(),' parameters')

        def count_flops(model):
            run_meta = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()

            # We use the Keras session graph in the call to the profiler.
            flops = tf.profiler.profile(graph=K.get_session().graph,
                                        run_meta=run_meta, cmd='op', options=opts)

            return flops.total_float_ops  # Prints the "flops" of the model.

        print('Model has ',count_flops(model),' flops')

        #plot_model(model, to_file='plots/model_'+config.name+'.png')

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

        frozen_graph_path = 'model'+config.NAME+'.pb'
        with tf.gfile.GFile(frozen_graph_path, 'wb') as f:
            f.write(od_graph_def.SerializeToString())

    if args.command == 'anchors':
        class inferNovaConfig(novaConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = inferNovaConfig()

        RHCFiles = os.listdir(os.path.join(args.dataset,'RHC'))
        RHCFlux  = [os.path.join(args.dataset,'RHC',f) for f in RHCFiles if 'fluxswap' in f][0:1]

        dataset = novaDataset()
        dataset.load_nova(RHCFlux)
        dataset.prepare()

        model = modellib.MaskRCNN(mode="inference", model_dir=args.logdir, config=config)

        images = [dataset.load_image(0)]
        molded_images, image_metas, windows = model.mold_inputs(images)
        print("molded_images.shape",molded_images.shape)
        image_shape = molded_images[0].shape
        # Anchors
        anchors = model.get_anchors(image_shape)

        with open('anchors.txt','w') as f:
            for i in anchors:
                for j in i:
                    f.write(str(j)+' ')
                f.write('\n')

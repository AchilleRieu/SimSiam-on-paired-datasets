import cv2
from matplotlib import pyplot as plt
import numpy as np
from torchvision import datasets
import tqdm
import os.path
from PIL import Image

class CropLayer(object):
    def __init__(self, params, blobs):
        # initialize our starting and ending (x, y)-coordinates of
        # the crop
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0

    def getMemoryShapes(self, inputs):
        # the crop layer will receive two inputs -- we need to crop
        # the first input blob to match the shape of the second one,
        # keeping the batch size and number of channels
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])

        # compute the starting and ending crop coordinates
        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W
        self.endY = self.startY + H

        # return the shape of the volume (we'll perform the actual
        # crop during the forward pass
        return [[batchSize, numChannels, H, W]]

    def forward(self, inputs):
        # use the derived (x, y)-coordinates to perform the crop
        return [inputs[0][:, :, self.startY:self.endY,
                self.startX:self.endX]]


def HED_paired_dataset(data, dataset_path, split):
    # The pre-trained model that OpenCV uses has been trained in Caffe framework
    #Download from the link above

    if(not os.path.isfile(dataset_path+split+"_hed.npy")):

        protoPath = "./Code/Paired_Dataset_SimSiam/hed_model/deploy.prototxt"
        modelPath = "./Code/Paired_Dataset_SimSiam/hed_model/hed_pretrained_bsds.caffemodel"
        net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

        cv2.dnn_registerLayer("Crop", CropLayer)

        # load the input image and grab its dimensions, for future use while defining the blob
        nb_img = data.shape[0]
        (H, W) = data.shape[2:4]

        dataset_HED = np.zeros((nb_img,H,W), dtype=np.uint8)

        # construct a blob out of the input image 
        #blob is basically preprocessed image. 
        #OpenCV’s new deep neural network (dnn ) module contains two functions that 
        #can be used for preprocessing images and preparing them for 
        #classification via pre-trained deep learning models.
        # It includes scaling and mean subtraction
        mean_pixel_values_train = np.average(data, axis = (2,3))

        for i,img in enumerate(tqdm.tqdm(data)):
            mean=(mean_pixel_values_train[i][0], mean_pixel_values_train[i][1], mean_pixel_values_train[i][2])
            img2 = np.moveaxis(img, 0, -1)
            blob_train = cv2.dnn.blobFromImage(img2, scalefactor=0.7, size=(W, H), mean=mean, swapRB= False, crop=False)
            net.setInput(blob_train)
            hed = net.forward()
            hed = hed[0,0,:,:]
            hed = (255 * hed).astype("uint8")
            dataset_HED[i] = hed
        
        try:
            os.makedirs(dataset_path)
        except:
            print("Target dir already exist")
        np.save(dataset_path+split+"_hed", dataset_HED, allow_pickle=False)
        return(dataset_HED)
    
    else:
        print("Dataset already created")
        data_hed = np.load(dataset_path+split+"_hed.npy").astype("uint8")

        return(data_hed)

def main():
    train_dataset = datasets.STL10("./datasets", split='train+unlabeled', download=False)
    # HED_dataset = HED_paired_dataset(train_dataset.data, train_dataset.root+"/stl10_hed_npy/",train_dataset.split)
    HED_dataset_loaded = np.load("./datasets/stl10_hed_npy/train+unlabeled_hed.npy")
    f, axarr = plt.subplots(3, 2)
    img = np.transpose(train_dataset.data[58], (1, 2, 0))
    img_hed = HED_dataset_loaded[58]
    img_gs = Image.fromarray(img)
    img_gs = img_gs.convert('L')
    axarr[0,0].imshow(img)
    axarr[1,0].imshow(img_gs, cmap='gray', vmin=0, vmax=255)
    axarr[2,0].imshow(img_hed, cmap='gray', vmin=0, vmax=255)

    img2 = np.transpose(train_dataset.data[0], (1, 2, 0))
    img_hed2 = HED_dataset_loaded[0]
    img_gs2 = Image.fromarray(img2)
    img_gs2 = img_gs2.convert('L')
    axarr[0,1].imshow(img2)
    axarr[1,1].imshow(img_gs2, cmap='gray', vmin=0, vmax=255)
    axarr[2,1].imshow(img_hed2, cmap='gray', vmin=0, vmax=255)
    plt.show()
    print("END")


if __name__ == "__main__":
    main()





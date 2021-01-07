import numpy as np
import os
import glob
import tqdm
import cv2 as cv
# from PIL import Image
# import cv2 as cv
# from keras.utils.np_utils import to_categorical
# import tifffile as tiff
# import matplotlib.image as mpimg


def get_labels():
    """Load the mapping that associates classes with label colors

    Returns:
        np.ndarray with dimensions (13, 3)
    """
    return np.asarray(
        [
            [0, 0, 0],  # 0，其他类别
            [255,255,255]
        ]
    )
# ["其他类别","水田","水浇地","旱耕地","园林","乔木林地","灌木林地","天然草地",
#  "人工草地","工业用地","城市住宅","村镇住宅","交通运输","河流","湖泊","坑塘"]
def encode_segmap(mask):
    """Encode segmentation label images as pascal classes

    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.

    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(np.uint8)
    return label_mask


def decode_segmap(label_mask, n_classes):
    """Decode segmentation class labels into a color image

    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.

    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    label_colours = get_labels()
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3),dtype=np.uint8)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb

path = "./predict_UNet_55"
output_path ="./test_tif"
if not os.path.exists(output_path):
    os.makedirs(output_path)
labelList = glob.glob(f"{path}/*png")
num = len(labelList)
for i in tqdm.tqdm(range(num)):
    label = cv.imread(labelList[i], 0)
    # label = np.array(Image.open(labelList[i]))
    # label = encode_segmap(label)
    # shape = label.shape
    # print(shape)
    name = os.path.split(labelList[i])[-1].split('.')[0]
    # np.save(f"{output_path}/{name}.npy",label)
    cv.imwrite(f"{output_path}/{name}.tif", label)

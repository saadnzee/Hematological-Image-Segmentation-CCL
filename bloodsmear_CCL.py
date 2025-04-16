import numpy as np
import cv2
import matplotlib.pyplot as plt

# Plotting Histogram for each image to analyze frequency count of pixel grayscale values to better define V-Set.
# e.g. nucleus usually covers very few pixels so pixels having less frequency values correspond to nucleus.
def plot_histogram(image):
    plt.figure()
    plt.title("Histogram Plot")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.hist(image.ravel(), bins=256, range=[0,256], color='red', alpha=0.7)
    plt.show()

# Padding Function for CCA - to avoid out of bound errors.
def padding(original):
    [h_Original, w_Original] = np.shape(original)
    border_image = np.ones((h_Original+2, w_Original+2), dtype=np.uint8) * 255
    [h_Border, w_Border] = np.shape(border_image)
    border_image[1:h_Border-1, 1:w_Border-1] = original
    return border_image

# CCA with 8-connectivity. Range for V-Set is passed as lower_th & and upper_th.
def CCA(img, lower_th, upper_th):
    [rows, cols] = np.shape(img)   # Dimensions of the original image
    border_img = padding(img)   # Bordered Image
    label_img = np.zeros((rows, cols), np.uint8)   # Blank canvas for label_image.
    label = 10   # Random value which will serve as starting label.
    label_count = 0
    eq_list = {}    # Mantains labels for each new component.
    label_map = {}    # Mantains mapping of equivalent labels.
    # Sets range of values for V-Set (treated as foreground) i.e. will check neighbors etc.
    # e.g. it will V-Set as (0-200) for separating cytoplasm from background.
    V_set = list(range(lower_th, upper_th))  
    
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            # only check if in V-Set
            if border_img[i, j] in V_set:
                neighbors = []
                # Check bounds before accessing neighbors (avoiding negative-indexing)
                if i - 2 >= 0:
                    neighbors.append(label_img[i-2, j-1])  # Top
                    if j - 2 >= 0:
                        neighbors.append(label_img[i-2, j-2])  # Top Left
                    if j < cols:
                        neighbors.append(label_img[i-2, j])  # Top Right
                
                if j - 2 >= 0:
                    neighbors.append(label_img[i-1, j-2])  # Left

                # All non-zero neighbors are placed in nonzero_neighbors (label values).
                nonzero_neighbors = [n for n in neighbors if n != 0]

                # When all neighbors are zero
                if not nonzero_neighbors:
                    label = label + 1
                    label_count = label_count + 1
                    eq_list[label_count] = label
                    label_img[i-1, j-1] = label
                
                # When neighbor has a value
                else:
                    min_label = min(nonzero_neighbors)
                    label_img[i-1, j-1] = min_label
                    # Mantaining Equivalency
                    for n in nonzero_neighbors:
                        if n != min_label:
                            label_map[n] = min_label

    # Resolving equivalences
    for k, v in label_map.items():
        val = v
        while val in label_map:
            val = label_map[val]
        label_map[k] = val

    # Replacing equivalent labels
    for i in range(rows):
        for j in range(cols):
            if label_img[i, j] in label_map:
                label_img[i, j] = label_map[label_img[i, j]]

    return label_img, len(np.unique(label_img)) 

# Keeping only the two largest components (background + WBC)
# This is to remove the noise/components surrounding the WBC
def post_process(label_img):
    unique_labels, counts = np.unique(label_img, return_counts=True)
    largest_labels = unique_labels[np.argsort(counts)[-2:]]
    label_img = np.where(np.isin(label_img, largest_labels), label_img, 0)
    img_coloured = cv2.applyColorMap(label_img, cv2.COLORMAP_JET)
    return label_img, img_coloured, len(np.unique(label_img)) 

# To map CCA-Output Gray Scale values to Mask Image Gray Scale Values.
# This function is used to map cytoplasm gray scale values to 128 (Gray Color).
def threshold(label_img):
    [h,w] = np.shape(image)
    for i in range(0, h):
        for j in range(0,w):
            if(label_img[i][j]>=10):
                label_img[i][j] = 128
            else:
                label_img[i][j] = 0
    return len(np.unique(label_img))

# Function used to combine the two CCA outputs, CCA (Cytoplasm) and CCA (Nucleus)
def concat(label_img1, label_img2):
    [h,w] = np.shape(label_img1)
    for i in range(0, h):
        for j in range(0,w):
            if(label_img2[i][j]==255):
                label_img1[i][j] = 255
    return label_img1

# Computing dice coefficient between my CCA Output and Mask Image.
def dice_coefficient(label_img, mask_img):
    [h,w] = np.shape(label_img)
    count = 0
    for i in range(0, h):
        for j in range(0,w):
            if(label_img[i][j]==mask_img[i][j]):
                count += 1
    dice = count / (h*w)
    return dice
    
# Main Code
image = cv2.imread("C:\\Users\\Saadn\\Desktop\\003.bmp", 0)
cv2.imshow('Original Image', image)
cv2.waitKey()

# Histogram Plot
plot_histogram(image)

# Could take V-Set Range as input from the user as-well!

# CCA for Cytoplasm and Background (0-200) V-Set Range
[CCA_matrix, objects] = CCA(image,0,200)

print("\n=========================================")
print("=============Cytoplasm Case:=============")
print("=========================================")

print("Before Post Processing: Number of objects: ", objects)
cv2.imshow('Before Post Processing: CCA Matrix', CCA_matrix)
cv2.waitKey()

[CCA_matrix_processed, CCA_colored_processed, objects_processed] = post_process(CCA_matrix)
print("After Post Processing: Number of objects: ", objects_processed)
cv2.imshow('After Post Processing: CCA Matrix', CCA_matrix_processed)
cv2.waitKey()

# to map cytoplasm gray scale values (label value) to 128 (Gray Color) to match mask image.
objects_processed = threshold(CCA_matrix_processed)

print("After Thresholding: Number of objects: ", objects_processed)
cv2.imshow('After Thresholding: CCA Matrix', CCA_matrix_processed)
cv2.waitKey()

# CCA for Nucleus
[CCA_matrix_nucleus, objects_nucleus] = CCA(image,0,70)

print("\n=========================================")
print("==============Nucleus Case:==============")
print("=========================================")

print("Before Post Processing: Number of objects: ", objects_nucleus)
cv2.imshow('Before Post Processing: CCA Matrix', CCA_matrix_nucleus)
cv2.waitKey()

# to map nucleus gray scale values (label value) to 255 (White Color) to match mask image.
_, CCA_matrix_nucleus = cv2.threshold(CCA_matrix_nucleus, 10, 255, cv2.THRESH_BINARY)

print("After Thresholding: Number of objects: ", len(np.unique(CCA_matrix_nucleus)))
cv2.imshow('After Thresholding: CCA Matrix', CCA_matrix_nucleus)
cv2.waitKey()

# Combining the CCA for cytoplasm and nucleus
Final_CCA_Img = concat(CCA_matrix_processed, CCA_matrix_nucleus)
print("Final Number of Objects After Concat: ", len(np.unique(Final_CCA_Img)))
cv2.imshow('Final CCA Image', Final_CCA_Img)
cv2.waitKey()

# Dice-Coefficient Computation
print("\n=========================================")
print("===========Dice Coefficient:=============")
print("=========================================")
mask_image = cv2.imread("C:\\Users\\Saadn\\Desktop\\003_Mask.png", 0)
dice = dice_coefficient(Final_CCA_Img, mask_image)
print("Dice Coefficient is: ", dice)
print("Dice Percentage is: ", dice*100 ,"%")

"""Since they are just RGB images then we can use proxy instead of the specific bands used for the indeces

Normalized Difference Water Index
"""

import csv
import os
import numpy as np
from skimage import io, color, img_as_ubyte
from skimage.measure import label,regionprops
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
import cv2
from skimage.color import label2rgb

def calculate_ndwi(image_path):
    # read the image 
    img = io.imread(image_path)

    # Convert the image to float64 data type
    img = img_as_ubyte(img)

    # Calculate the NDWI
    green = img[:, :, 1].astype('float64')
    blue = img[:, :, 2].astype('float64')

    # np.seterr(divide='ignore', invalid='ignore')
    # if there is zero value for np.add(green, blue) then put 0 at this position
    if np.any(np.add(green, blue) == 0):
        ndwi = np.divide(np.subtract(green, blue), np.add(green, blue) + 0.0000001)
    else:
        ndwi = np.divide(np.subtract(green, blue), np.add(green, blue))

    list_ndwi = []
    for i in range(len(ndwi)):
        for j in range(len(ndwi[i])):
            list_ndwi.append(ndwi[i][j])

    return list_ndwi

"""Normalized Difference Vegetation Index"""

def calculate_ndvi(image_path):

    # read the image
    img = io.imread(image_path)

    # Convert the image to float64 data type
    img = img_as_ubyte(img)

    # Calculate the NDVI
    # Extract the Red and NIR bands
    red = img[:, :, 0].astype('float64')
    nir = img[:, :, 1].astype('float64')

    if np.any(np.add(nir, red) == 0):
        ndvi = np.divide(np.subtract(nir, red), np.add(nir, red) + 0.0000001)
    else:
        ndvi = np.divide(np.subtract(nir, red), np.add(nir, red))

    list_ndvi = []
    for i in range(len(ndvi)):
        for j in range(len(ndvi[i])):
            list_ndvi.append(ndvi[i][j])

    return list_ndvi

"""Color features:
* color averaging
* color varience
* color histogram

***Color varience significintaly differs between flooded and non-flooded images.***

***Difference in blue color value between 2 flooded image is too small, while this difference between flood and non-flooded images is large.***

one image
"""

def color_feature(image_path):
    # Load the flooded image
    image = Image.open(image_path)

    # Convert the image to a numpy array
    img_arr_flooded = np.array(image)

    # Extract the average color
    avg_color = np.mean(img_arr_flooded, axis=(0, 1))

    # Extract the color variance
    color_var = np.var(img_arr_flooded, axis=(0, 1))

    # Extract the color histogram
    hist, bins = np.histogramdd(img_arr_flooded.reshape(-1, 3), bins=256, range=((0, 255), (0, 255), (0, 255)))
    hist_norm = hist / np.sum(hist)

    hist_list = []
    for i in range(len(hist_norm)):
        for j in range(len(hist_norm[i])):
            for k in range(len(hist_norm[i][j])):
                hist_list.append(hist_norm[i][j][k])

    return avg_color, color_var, hist_list

"""Extract texture features:
* texture gradient
* texture energy
* texture correlation
* texture homogenity

***texture energy and texture homogenity have slight difference with 2 flooded images, while it differs with the flooded and non-flooded images***


"""

def texture_feature(image_path):
    # read the image
    img = io.imread(image_path)
    # Convert the image to grayscale
    img_gray = color.rgb2gray(img)

    # Convert the image to uint8
    img_gray = img_as_ubyte(img_gray)

    # Calculate the texture gradient
    texture_gradient = np.mean(np.gradient(img_gray)[0])

    # Calculate the texture energy
    texture_energy = np.mean(graycoprops(graycomatrix(img_gray, distances=[1], angles=[0], levels=256), 'energy'))

    # Calculate the texture correlation
    texture_correlation = np.mean(
        graycoprops(graycomatrix(img_gray, distances=[1], angles=[0], levels=256), 'correlation'))

    # Calculate the texture homogeneity
    texture_homogeneity = np.mean(
        graycoprops(graycomatrix(img_gray, distances=[1], angles=[0], levels=256), 'homogeneity'))


    return texture_gradient, texture_energy, texture_correlation, texture_homogeneity

"""Extract shape features:
* area
* perimeter
* compactness
* eccentricity.

--> summation compactness and summation of eccentricity have significant difference between flooded and non-flooded, while 2 flooded have no difference

"""

# extract shape features the area, perimeter, compactness, or eccentricity.

def shape_feature(img_flooded):

    # Load the flooded image
    img_flooded = io.imread(img_flooded)

    # Convert the image to grayscale
    img_flooded_gray = color.rgb2gray(img_flooded)

    # Convert the image to uint8
    img_flooded_gray = img_as_ubyte(img_flooded_gray)

    # Extract the regionprops
    region_props = regionprops(img_flooded_gray)

    # Print the summation area of the all regions
    area = np.sum([region.area for region in region_props])

    # Print the summation perimeter of all regions
    perimeter = np.sum([region.perimeter for region in region_props])

    # Print the summation compactness of all regions
    compactness = np.sum([region.perimeter ** 2 / region.area for region in region_props])

    # Print the summation eccentricity of all regions
    eccentricity = np.sum([region.eccentricity for region in region_props])

    return area, perimeter, compactness, eccentricity

def segment_feature(image_path):

    # Load the image
    image = io.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding to segment the image into foreground and background
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Perform connected component labeling to segment the image into regions
    label_image = label(thresh)

    # Find the properties of each region
    regions = regionprops(label_image)

    # Find the region with the largest area (assuming it's the flooded region)
    flooded_region = max(regions, key=lambda x: x.area)

    # Create a binary mask for the flooded region
    mask = np.zeros_like(gray)
    mask[label_image == flooded_region.label] = 255

    flood_image = image.copy()
    flood_image[mask == 255] = (255,0,0)

    return flood_image

def extract_all_features(class_name, class_dir):
    # check if the file already exists
    features_file = class_name + '_features.csv'
    # file_exists = os.path.isfile(features_file)
    # # if the file doesn't exist yet, create it with the header
    # if not file_exists:
    #     with open(features_file, 'w', newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(['ndwi', 'ndvi', 'color', 'var', 'hist', 'texture_gradient', 'texture_energy', 'texture_correlation', 'texture_homogeneity', 'area', 'perimeter', 'compactness', 'eccentricity'])
    # process each image in the directory and append the features to the CSV file
    with open(features_file, 'a', newline='') as f:
        writer = csv.writer(f)
        for filename in os.listdir(class_dir):
            ndwi = calculate_ndwi(os.path.join(class_dir, filename))
            ndvi = calculate_ndvi(os.path.join(class_dir, filename))
            color, var, hist = color_feature(os.path.join(class_dir, filename))
            texture_gradient, texture_energy, texture_correlation, texture_homogeneity = texture_feature(os.path.join(class_dir,filename))
            area, perimeter, compactness, eccentricity = shape_feature(os.path.join(class_dir, filename))

            # merge all to be in one list
            features = ndwi + ndvi + hist + [color[0], color[1], color[2], var[0], var[1], var[2], texture_gradient, texture_energy, texture_correlation, texture_homogeneity, area, perimeter, compactness, eccentricity]
            # append the features to the CSV file
            writer.writerow(features)
# Satalleite-images-project
Satellite Imagery project aims to detect if a flooding occurs or not from the post-flooding damages.

## Preprocessing 
Apply preprocessing on images before classification
* Contrast Enhancement
  * to distinguish better between water regions and land regions.
* Resize images
  * All images are from different sizes then resizing each image to 256 * 256.
* Data Augmentation 
  * the data size is small and not much suitable for learning, especially deep learning techniques. Data augmentation is done by rotating each image 90 degrees and 180 degrees. Thus, increasing the size 3 times more.

## Feature Extraction
* Indexes
  Extracting indices is common for more than 3-band stallite images but we have RGB images 
  * Normalized difference water index (NDWI)
    * Calculated using Green & Blue channels
  * Normalized difference vegetation index (NDVI)
    * Calculated using Green & Red channels
* Color Features
  * Color averaging
  * Color varience
  * Color Histogram
* Texture Features
  * Gradient
  * Energy
  * Correlation
  * Homogeneity
* Shape Feature
  Dividing the image into regions then get diiferent following values: 
  * Area
  * Perimeter
  * Compactness
  * Eccentricity
  
All those features have distguinshable values between flooded and non-flooded images.

## Classification
### Classical classification
![image](https://github.com/reem-atalah/Satalleite-images-project/assets/55799245/580c0461-d93d-45fd-af7a-44c924462ae5)

### DL extract features and classical classifiers
### DL
![image](https://github.com/reem-atalah/Satalleite-images-project/assets/55799245/9b6b1e11-fd9b-4eef-8f51-747678c4a7bd)

## Segmentation of flooded regions
Use different clustering algorithms
![image](https://github.com/reem-atalah/Satalleite-images-project/assets/55799245/6559ec29-462f-4eaa-bf42-6d4bb29d5e2e)

* Binarization
![image](https://github.com/reem-atalah/Satalleite-images-project/assets/55799245/96ab6c1d-af3d-436f-b462-205e27e47ce9)

* Kmeans (K=2)
![image](https://github.com/reem-atalah/Satalleite-images-project/assets/55799245/e56a3635-9e13-46e0-9245-31658ed4b5db)

* Kmeans (K=3)
![image](https://github.com/reem-atalah/Satalleite-images-project/assets/55799245/d75dcf62-db1d-4efe-82a8-ddc2d3b96c64)

* Isodata
![image](https://github.com/reem-atalah/Satalleite-images-project/assets/55799245/bc0ebde7-2a45-40c1-8e1a-9ffeedeedf69)

* Region Growing
![image](https://github.com/reem-atalah/Satalleite-images-project/assets/55799245/f7e3f78d-67ea-4254-8011-b8ddbbeab95d)

The best of them was **Region Growing**

For a visualized summary about the project check file Team_20_Presentation.pptx

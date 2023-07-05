# TensorflowSwinUNet-Segmentation-MultipleMyeloma (Updated: 2023/07/06)

<h2>
1 TensorflowSwinUNet-Segmentation-MultipleMyelom
</h2>
<p>
This is an experimental project detect MultipleMyeloma by using 
TensorflowSwinUNetModel,
which is a Tensorflow 2 implementation of
<a href="https://arxiv.org/pdf/2105.05537.pdf">
<b>Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation</b>
</a>.
</p>
In order to write the <a href="./TensorflowSwinUNet.py">TensorflowSwinUNet</a> Python class, we have used the Python code in the following web sites.
</p>
<pre>
1.keras-unet-collection
 https://github.com/yingkaisha/keras-unet-collection/tree/main/keras_unet_collection
</pre>

<p>
Please see also:
</p>
<pre>
2. U-Net: Convolutional Networks for Biomedical Image Segmentation
 https://arxiv.org/pdf/1505.04597.pdf
</pre>
<pre>
3.Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation
https://arxiv.org/pdf/2105.05537.pdf
</pre>

The original dataset used here has been take from the following  web site:<br><br>
<b>SegPC-2021-dataset</b><br>
SegPC-2021: Segmentation of Multiple Myeloma Plasma Cells in Microscopic Images<br>
<pre>
https://www.kaggle.com/datasets/sbilab/segpc2021dataset
</pre>
Citation:<br>
<pre>
Anubha Gupta, Ritu Gupta, Shiv Gehlot, Shubham Goswami, April 29, 2021, "SegPC-2021: Segmentation of Multiple Myeloma Plasma Cells 
in Microscopic Images", IEEE Dataport, doi: https://dx.doi.org/10.21227/7np1-2q42.
BibTex
@data{segpc2021,
doi = {10.21227/7np1-2q42},
url = {https://dx.doi.org/10.21227/7np1-2q42},
author = {Anubha Gupta; Ritu Gupta; Shiv Gehlot; Shubham Goswami },
publisher = {IEEE Dataport},
title = {SegPC-2021: Segmentation of Multiple Myeloma Plasma Cells in Microscopic Images},
year = {2021} }
IMPORTANT:
If you use this dataset, please cite below publications-
1. Anubha Gupta, Rahul Duggal, Shiv Gehlot, Ritu Gupta, Anvit Mangal, Lalit Kumar, Nisarg Thakkar, and Devprakash Satpathy, 
 "GCTI-SN: Geometry-Inspired Chemical and Tissue Invariant Stain Normalization of Microscopic Medical Images," 
 Medical Image Analysis, vol. 65, Oct 2020. DOI: 
 (2020 IF: 11.148)
2. Shiv Gehlot, Anubha Gupta and Ritu Gupta, 
 "EDNFC-Net: Convolutional Neural Network with Nested Feature Concatenation for Nuclei-Instance Segmentation,"
 ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 
 Barcelona, Spain, 2020, pp. 1389-1393.
3. Anubha Gupta, Pramit Mallick, Ojaswa Sharma, Ritu Gupta, and Rahul Duggal, 
 "PCSeg: Color model driven probabilistic multiphase level set based tool for plasma cell segmentation in multiple myeloma," 
 PLoS ONE 13(12): e0207908, Dec 2018. DOI: 10.1371/journal.pone.0207908
License
CC BY-NC-SA 4.0
</pre>

<h2>
2. Prepare MultipleMyelom dataset
</h2>
<h3>
2.1. Download 
</h3>
Please download original <b>Multiple Myeloma Plasma Cells</b> dataset from the following link.
<b>SegPC-2021-dataset</b><br>
SegPC-2021: Segmentation of Multiple Myeloma Plasma Cells in Microscopic Images<br>
<pre>
https://www.kaggle.com/datasets/sbilab/segpc2021dataset
</pre>
The folder structure of the dataset is the following.<br>
<pre>
TCIA_SegPC_dataset
├─test
│  └─x
├─train
│  ├─x
│  └─y
└─valid
    ├─x
    └─y
</pre>
Each <b>x</b> folder of the dataset contains the ordinary image files of Multiple Myeloma Plasma Cells,
and <b>y</b> folder contains the mask files to identify each Cell of the ordinary image files.
  Both the image size of all files in <b>x</b> and <b>y</b> is 2560x1920 (2.5K), which is apparently too large to use 
for our TensoflowUNet Model.<br>

Sample images in train/x:<br>
<img src="./asset/train_x.png" width="720" height="auto"><br>
Sample masks in train/y:<br>
<img src="./asset/train_y.png" width="720" height="auto"><br>

<h3>
2.2. Generate MultipleMyeloma Image Dataset
</h3>
 We have created Python script <a href="./projects/MultipleMyeloma/generator/MultipleMyelomaImageDatasetGenerator.py">
 <b>MultipleMyelomaImageDatasetGenerator.py</b></a> to create images and masks dataset.<br>
 This script will perform following image processing.<br>
 <pre>
 1 Resize all bmp files in <b>x</b> and <b>y</b> folder to 256x256 square image.
 2 Create clear white-black mask files from the original mask files.
 3 Create cropped images files corresponding to each segmented region in mask files in <b>y</b> folders.
</pre>

See also the following web-site on Generation of MultipleMyeloma Image Dataset.<br>
<a href="https://github.com/atlan-antillia/Image-Segmentation-Multiple-Myeloma">Image-Segmentation-Multiple-Myeloma </a>
<br>

<h3>
2.3 Generated MultipleMyeloma dataset.<br>
</h3>
Finally, we have generated the resized jpg files dataset below.<br> 
<pre>
└─projects
    └─MultipleMyeloma
        └─MultipleMyeloma
            ├─train
            │  ├─images
            │  └─masks
            └─valid
                ├─images
                └─masks
</pre>



<h2>
3. Create TensorflowSwinUNet Model
</h2>
 You can customize your <a href="./TensorflowSwinUNet.py">TensorflowSwinUNet</a> model by using 
 a configration file.<br>
 The following is the case of MultipleMyeloma segmentation dataset.<br>
Please note that image width, image_height, loss and metrics functions in model section.<br>
<pre>
[model]
image_width    = 384
image_height   = 384
loss           = "binary_crossentropy"
metrics        = ["binary_accuracy"]
</pre>
<b>train_eval_infer.config</b><br>
<pre>
; train_eval_infer.config
; 2023/07/06 antillia.com
; Modified to use loss and metric
;
; #Oxford IIIT image segmentation with SwinUNET
; #https://github.com/yingkaisha/keras-vision-transformer/blob/main/examples/Swin_UNET_oxford_iiit.ipynb

[model]
image_width    = 384
image_height   = 384
image_channels = 3
num_classes    = 1
filter_num_begin = 128   
; number of channels in the first downsampling block; it is also the number of embedded dimensions

depth = 4
; the depth of SwinUNET; depth=4 means three down/upsampling levels and a bottom level 

stack_num_down = 2         
; number of Swin Transformers per downsampling level

stack_num_up = 2
; number of Swin Transformers per upsampling level

patch_size = (4, 4)        
; Extract 4-by-4 patches from the input image. Height and width of the patch must be equal.  

num_heads = [4, 8, 8, 8]   
;number of attention heads per down/upsampling level

window_size = [4, 2, 2, 2] 
;the size of attention window per down/upsampling level

num_mlp = 512              
; number of MLP nodes within the Transformer

shift_window=True          
;Apply window shifting, i.e., Swin-MSA

;Optimizer Adam 1e-4?
learning_rate  = 0.0001
clipvalue      = 0.2
;loss           = "bce_iou_loss"
;loss           = "dice_loss"
;loss           = "iou_loss"
loss           = "binary_crossentropy"
metrics        = ["binary_accuracy"]
;metrics        = ["iou_coef"]
show_summary   = False

[train]
epochs        = 100
batch_size    = 2
patience      = 10
metrics       = ["binary_accuracy", "val_binary_accuracy"]
;metrics       = ["iou_coef", "val_iou_coef"]
model_dir     = "./models"
eval_dir      = "./eval"

image_datapath = "./MultipleMyeloma/train/images/"
mask_datapath  = "./MultipleMyeloma/train/masks/"
create_backup  = True

[eval]
image_datapath = "./MultipleMyeloma/valid/images/"
mask_datapath  = "./MultipleMyeloma/valid/masks/"

[infer] 
images_dir    = "./mini_test" 
output_dir    = "./mini_test_output"
merged_dir    = "./mini_test_output_merged"

[tiledinfer] 
overlapping  = 0
images_dir   = "./4k_mini_test"
output_dir   = "./4k_tiled_mini_test_output"

[mask]
blur        = True
binarize    = True
threshold   = 128 
</pre>

<br>
You will pass the filename of this configuration file to <a href="./TensorflowSwinUNet.py">TensorflowSwinUNet</a> constructor to create your model 
in the following way:<br>
<pre>
  config_file = "./train_eval_infer.config"
  model       = TensorflowSwinUNet(config_file)
</pre>


<h2>
4 Train TensorflowSwinUNet Model
</h2>
 We can create and train your TensorflowSwinUNet model by MultipleMyeloma dataset defined in the <b>train_eval_infer.config</b> file.<br>
<p>

We can use <a href="./ImageMaskDataset.py">ImageMaskDataset</a> class to create <b>train</b> and <b>test</b> dataset from the 
the original downloaded file.<br>
Please move to <b>./projects/MultipleMyeloma/</b>, and run the following bat file.
<pre>
>1.train.bat
</pre>
which simply runs the Python script 
<a href="./TensorflowSwinUNetTrainer.py">TensorflowSwinUNetTrainer.py</a>
<pre>
python ../../TensorflowSwinUNetTrainer.py ./train_eval_infer.config
</pre>

<b>Training console output</b><br>

<img src="./asset/train_console_output_at_epoch_32_0706.png" width="720" height="auto"><br>
<br>
<b>Train metrics: iou_coef</b>, which shows Excessive Fluctuation.<br>
<img src="./asset/train_metrics.png" width="720" height="auto"><br>

<br>
<b>Train losses: iou_loss</b>, which shows Excessive Fluctuation<br>
<img src="./asset/train_losses.png" width="720" height="auto"><br>


<h2>
5 Evaluation
</h2>
 We can evaluate the prediction(segmentation) accuracy in <b>test</b> dataset by using our Trained TensorflowSwinUNet Model,
and <b>train_eval_infer.config</b> file.<br>
Please move to <b>./projects/MultipleMyeloma/</b>, and run the following bat file.
<pre>
>2.evaluate.bat
</pre>
, which runs the Python script 
<a href="./TensorflowSwinUNetEvaluator.py">TensorflowSwinUNetEvaluator.py</a>
<pre>
python ../../TensorflowSwinUNetEvaluator.py ./train_eval_infer.config
</pre>
<b>Evaluation console output</b><br>
<img src="./asset/evaluate_console_output_at_epoch_32_0706.png" width="720" height="auto"><br>
<br>

<h2>
6 Non Tiled Image Segmentation
</h2>
Please move to <b>./projects/MultipleMyeloma/</b>, and run the following bat file.<br>
<pre>
>3.infer.bat
</pre>
which runs the Python script 
<a href="./TensorflowSwinUNetInferencer.py">TensorflowSwinUNetInferencer.py</a>
<pre>
python ../../TensorflowSwinUNetInferencer.py  ./train_eval_infer.config
</pre>

<b>Input images 2.5K (mini_test)</b><br>
<img src="./asset/mini_test.png" width="1024" height="auto"><br>
<br>
<b>Infered images (mini_test_output)</b><br>

<img src="./asset/mini_test_output.png" width="1024" height="auto"><br><br>


<h2>
7 Tiled Image Segmentation 
</h2>
Please move to <b>./projects/MultipleMyeloma/</b>, and run the following bat file.<br>
<pre>
>4.tiled_infer.bat
</pre>
which runs the Python script 
<a href="./TensorflowSwinUNetTiledInferencer.py">TensorflowSwinUNetTiledInferencer.py</a>
<pre>
python ../../TensorflowSwinUNetTiledInferencer.py  ./train_eval_infer.config
</pre>
Please note that overlalling is 0 in this case.<br>
<pre>
[tiledinfer] 
overlapping  = 0
</pre>
<b>Input images 4K (4k_mini_test)</b><br>

<img src="./asset/4k_mini_test.png" width="1024" height="auto"><br>
<br>
<b>Tiled-Image-Segmentation: Infered images (4k_mini_test_output)</b><br>

<img src="./asset/4k_mini_test_image_output_tiled.png" width="1024" height="auto"><br><br>

 Please refer to our experiment on <b>Overlapped-Tiled-Image-Segmentation</b> based on UNet3Plus Model.<br>
<a href="https://github.com/sarah-antillia/TensorflowUNet3Plus-Segmentation-MultipleMyeloma">
TensorflowUNet3Plus-Segmentation-MultipleMyeloma</a>
<br> <br>
 See also Tiled-Image-Segmentation based on <b>classic UNet Model without BatchNormalization</b>.<br>
<a href="https://github.com/atlan-antillia/Tiled-Image-Segmentation-Multiple-Myeloma">Tiled-Image-Segmentation-Multiple-Myeloma</a>

<br>
<br>
<h3>
References
</h3>

<b>1.Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation</b><br>
Hu Cao, Yueyue Wang, Joy Chen, Dongsheng Jiang, Xiaopeng Zhang, Qi Tian, and Manning Wang<br>
<pre>
 https://www.sciencedirect.com/science/article/abs/pii/S0893608019302503
</pre>


<b>2.keras-unet-collection</b><br>
<pre>
 https://github.com/yingkaisha/keras-unet-collection/tree/main/keras_unet_collection
</pre>

<b>3.Tiled-Image-Segmentation-MultipleMyeloma</b><br>
Toshiyuki Arai @antillia.com
<br>
<pre>
https://github.com/atlan-antillia/Tiled-Image-Segmentation-Multiple-Myeloma
</pre>

<b>4.TensorflowUNet3Plus-Segmentation-MultipleMyeloma</b><br>
Toshiyuki Arai @antillia.com
<br>
<pre>
https://github.com/sarah-antillia/TensorflowUNet3Plus-Segmentation-MultipleMyeloma
</pre>
<b>5.TensorflowMultiResUNet-Segmentation-MultipleMyeloma</b><br>
Toshiyuki Arai @antillia.com
<br>
<pre>
https://github.com/sarah-antillia/TensorflowMultiResUNet-Segmentation-MultipleMyeloma
</pre>

<b>6. Semantic-Segmentation-Loss-Functions (SemSegLoss)</b><br>
<pre>
https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions
</pre>
<pre>
Citation
@inproceedings{jadon2020survey,
  title={A survey of loss functions for semantic segmentation},
  author={Jadon, Shruti},
  booktitle={2020 IEEE Conference on Computational Intelligence in Bioinformatics and Computational Biology (CIBCB)},
  pages={1--7},
  year={2020},
  organization={IEEE}
}
@article{JADON2021100078,
title = {SemSegLoss: A python package of loss functions for semantic segmentation},
journal = {Software Impacts},
volume = {9},
pages = {100078},
year = {2021},
issn = {2665-9638},
doi = {https://doi.org/10.1016/j.simpa.2021.100078},
url = {https://www.sciencedirect.com/science/article/pii/S2665963821000269},
author = {Shruti Jadon},
keywords = {Deep Learning, Image segmentation, Medical imaging, Loss functions},
abstract = {Image Segmentation has been an active field of research as it has a wide range of applications, 
ranging from automated disease detection to self-driving cars. In recent years, various research papers 
proposed different loss functions used in case of biased data, sparse segmentation, and unbalanced dataset. 
In this paper, we introduce SemSegLoss, a python package consisting of some of the well-known loss functions 
widely used for image segmentation. It is developed with the intent to help researchers in the development 
of novel loss functions and perform an extensive set of experiments on model architectures for various 
applications. The ease-of-use and flexibility of the presented package have allowed reducing the development 
time and increased evaluation strategies of machine learning models for semantic segmentation. Furthermore, 
different applications that use image segmentation can use SemSegLoss because of the generality of its 
functions. This wide range of applications will lead to the development and growth of AI across all industries.
}
}

<br>

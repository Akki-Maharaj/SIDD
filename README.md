SMARTPHONE IMAGE DENOISING (SIDD)
=================================

ABOUT
-----
This repository demonstrates an end-to-end pipeline for image denoising using the Smartphone Image Denoising Dataset (SIDD). The goal is to remove real-world noise from smartphone images while preserving details and textures.

The project showcases how to load noisy-clean image pairs, preprocess the data, build a Convolutional Neural Network (CNN)-based denoising model (such as DnCNN or U-Net), train it, and evaluate its performance.

TECHNOLOGIES AND LIBRARIES USED
-------------------------------
- Python 3.x
- TensorFlow / Keras : for building and training the deep learning model
- NumPy : for numerical operations
- OpenCV : for image reading and processing
- Matplotlib : for visualization of images and results
- SIDD Dataset : Smartphone Image Denoising Dataset (by Abhiram V. et al.)

WORKFLOW / APPROACH
-------------------
1. DATA LOADING
   - Load pairs of noisy and corresponding ground truth (clean) images from the SIDD dataset.
   - Perform basic sanity checks to ensure data integrity.

2. PREPROCESSING
   - Resize or crop images to a fixed patch size suitable for training.
   - Normalize pixel values to [0, 1] range.

3. MODEL BUILDING
   - Build an image denoising neural network, such as a DnCNN or U-Net.
   - The model learns to map noisy input images to clean targets.

4. TRAINING
   - Train the model on noisy-clean pairs using Mean Squared Error (MSE) loss.
   - Use callbacks like early stopping and model checkpointing to prevent overfitting.

5. EVALUATION
   - Test the trained model on unseen noisy images.
   - Visualize denoised results and compare them with ground truth.
   - Compute metrics like PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index).

HOW TO RUN
----------
1. Clone this repository.
2. Download the SIDD dataset and organize the data into appropriate folders.
3. Install required dependencies using pip:
   pip install tensorflow numpy opencv-python matplotlib
4. Open the notebook (SIDD.ipynb) in Jupyter Notebook or Google Colab.
5. Run each cell step-by-step to preprocess data, train the model, and evaluate results.

CITATIONS
---------
- Abhiram V., Abdelrahman Abdelhamed, Stephen Lin, Michael S. Brown. "A High-Quality Denoising Dataset for Smartphone Cameras." The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.
- Zhang, K., Zuo, W., Chen, Y., Meng, D., & Zhang, L. (2017). Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising. IEEE Transactions on Image Processing.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.

ACKNOWLEDGEMENTS
----------------
Thanks to the original authors of the SIDD dataset and the open-source community for sharing models and ideas for image denoising.

For improvements or suggestions, feel free to create an issue or pull request.

HAPPY CODING! 📸✨

# Image_colorization
A simple implementation of auto-encoder for colorizing black and white images
(Autoencoder+InceptionResnetV2)

@theamitnikhade

****Requirements****

Python3

Install the requirements from the requirements.txt file.

pip install -r requirements.txt

you can download the Inception model from the below link or it will automatically do so if you don't have it.

link- 

****Dataset****

I have used the dataset from kaggle called art-images-drawings-painting-sculpture-engraving, it consists of multiple data's of various arts. I used just a single data folder from it, as I don't have enough RAM.

***TRAINING WITH CUSTOM DATA****

To train the autoencoder with custom data, simply edit the prameters and add the path to your dataset and run the main.py file.

****generating colorized images****

To make the b&w images colourful, just add the test images path to the parameters.json file and run the colorize.py file.

I tried the same colorization method with variational autoencoder but they disappointed me, So when i tried with the autoencoder this gave me a better result as compared to the VAE. 


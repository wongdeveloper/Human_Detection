# Human_Detection
Human Detection using HOG as preprocessing and SVM as classifier.
Dataset:
https://drive.google.com/drive/folders/1sxhKn6o9Nnhjy0kHA84yVDJq50eHK0hy?usp=sharing
Reference:
https://github.com/vinay0410/Pedestrian_Detection
Laporan:
https://docs.google.com/document/d/1fPaniGnf3IL9TYPEtRpE5gAeZghKJ0hmJsrEY6Av8wk/edit?pli=1
Dependencies
OpenCV
scikit-image pip install scikit-image==0.14.1
scikkit-learn pip install scikit-learn==0.20.2
To test on images, simply run, python detectmulti.py -i <path to image>
For example, python detectmulti.py -i sample_images/pedestrian.jpg
To train just run:
python train.py --pos <path to positive images> --neg <path to negative images>
After successful training just run:
python test.py --pos <path to positive images> --neg <path to negative images>

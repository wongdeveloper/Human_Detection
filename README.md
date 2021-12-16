# Human_Detection
Human Detection using HOG as preprocessing and SVM as classifier.<br>
Dataset:<br>
https://drive.google.com/drive/folders/1sxhKn6o9Nnhjy0kHA84yVDJq50eHK0hy?usp=sharing<br>
Reference:<br>
https://github.com/vinay0410/Pedestrian_Detection<br>
Laporan:<br>
https://docs.google.com/document/d/1fPaniGnf3IL9TYPEtRpE5gAeZghKJ0hmJsrEY6Av8wk/edit?pli=1<br>
Dependencies<br>
OpenCV<br>
scikit-image pip install scikit-image==0.14.1<br>
scikkit-learn pip install scikit-learn==0.20.2<br>
To test on images, simply run, python detectmulti.py -i <path to image><br>
For example, python detectmulti.py -i sample_images/pedestrian.jpg<br>
To train just run:<br>
python train.py --pos <path to positive images> --neg <path to negative images><br>
After successful training just run:<br>
python test.py --pos <path to positive images> --neg <path to negative images><br>

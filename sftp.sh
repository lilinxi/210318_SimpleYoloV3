rm -rf /Users/limengfan/PycharmProjects/210318_SimpleYoloV3Sftp

mkdir /Users/limengfan/PycharmProjects/210318_SimpleYoloV3Sftp
cp /Users/limengfan/PycharmProjects/210318_SimpleYoloV3/train.py /Users/limengfan/PycharmProjects/210318_SimpleYoloV3Sftp
cp /Users/limengfan/PycharmProjects/210318_SimpleYoloV3/run.sh /Users/limengfan/PycharmProjects/210318_SimpleYoloV3Sftp
cp /Users/limengfan/PycharmProjects/210318_SimpleYoloV3/view.sh /Users/limengfan/PycharmProjects/210318_SimpleYoloV3Sftp

mkdir /Users/limengfan/PycharmProjects/210318_SimpleYoloV3Sftp/conf
cp /Users/limengfan/PycharmProjects/210318_SimpleYoloV3/conf/coco.names /Users/limengfan/PycharmProjects/210318_SimpleYoloV3Sftp/conf
cp /Users/limengfan/PycharmProjects/210318_SimpleYoloV3/conf/config.py /Users/limengfan/PycharmProjects/210318_SimpleYoloV3Sftp/conf
cp /Users/limengfan/PycharmProjects/210318_SimpleYoloV3/conf/voc.names /Users/limengfan/PycharmProjects/210318_SimpleYoloV3Sftp/conf

mkdir /Users/limengfan/PycharmProjects/210318_SimpleYoloV3Sftp/dataset
cp /Users/limengfan/PycharmProjects/210318_SimpleYoloV3/dataset/dataset_utils.py /Users/limengfan/PycharmProjects/210318_SimpleYoloV3Sftp/dataset
cp /Users/limengfan/PycharmProjects/210318_SimpleYoloV3/dataset/pennfudan_dataset.py /Users/limengfan/PycharmProjects/210318_SimpleYoloV3Sftp/dataset
cp /Users/limengfan/PycharmProjects/210318_SimpleYoloV3/dataset/voc_dataset.py /Users/limengfan/PycharmProjects/210318_SimpleYoloV3Sftp/dataset

mkdir /Users/limengfan/PycharmProjects/210318_SimpleYoloV3Sftp/model
cp /Users/limengfan/PycharmProjects/210318_SimpleYoloV3/model/base_model.py /Users/limengfan/PycharmProjects/210318_SimpleYoloV3Sftp/model
cp /Users/limengfan/PycharmProjects/210318_SimpleYoloV3/model/darknet.py /Users/limengfan/PycharmProjects/210318_SimpleYoloV3Sftp/model
cp /Users/limengfan/PycharmProjects/210318_SimpleYoloV3/model/yolov3.py /Users/limengfan/PycharmProjects/210318_SimpleYoloV3Sftp/model
cp /Users/limengfan/PycharmProjects/210318_SimpleYoloV3/model/yolov3decode.py /Users/limengfan/PycharmProjects/210318_SimpleYoloV3Sftp/model
cp /Users/limengfan/PycharmProjects/210318_SimpleYoloV3/model/yolov3loss.py /Users/limengfan/PycharmProjects/210318_SimpleYoloV3Sftp/model
cp /Users/limengfan/PycharmProjects/210318_SimpleYoloV3/model/yolov3net.py /Users/limengfan/PycharmProjects/210318_SimpleYoloV3Sftp/model

# mkdir /Users/limengfan/PycharmProjects/210318_SimpleYoloV3Sftp/weights
# cp /Users/limengfan/PycharmProjects/210318_SimpleYoloV3/weights/demo_darknet53_weights.pth /Users/limengfan/PycharmProjects/210318_SimpleYoloV3Sftp/weights

mkdir /Users/limengfan/PycharmProjects/210318_SimpleYoloV3Sftp/logs

sftp lab2 << EOF
put -r /Users/limengfan/PycharmProjects/210318_SimpleYoloV3Sftp /home/lenovo/data/lmf
EOF

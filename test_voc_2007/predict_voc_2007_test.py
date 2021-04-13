import torchvision

import conf.config
import model.yolov3
import dataset.bak_voc_dataset

if __name__ == "__main__":

    Config = conf.config.DefaultCocoConfig

    yolov3 = model.yolov3.YoloV3(Config)

    EPOCH = 1
    BATCH_SIZE = 10

    voc_dataloader = dataset.bak_voc_dataset.get_voc_dataloader(
        config=Config,
        root="/Users/limengfan/Dataset/VOC/VOC2012Train",
        batch_size=BATCH_SIZE
    )

    for epoch in range(EPOCH):
        print("Epoch:", epoch)
        for step, (tensord_images, tensord_target_list) in enumerate(voc_dataloader):
            print("step:", step)
            for batch_index in range(BATCH_SIZE):
                print("batch:", batch_index)
                predict_annotation = yolov3.predict_annotation(tensord_images[batch_index])
                [print(anno) for anno in predict_annotation]

            exit(-1)

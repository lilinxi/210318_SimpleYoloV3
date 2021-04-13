if __name__ == "__main__":
    import conf.config
    import dataset.bak_voc_dataset
    import model.yolov3

    Config = conf.config.VocConfig

    # 1. 初始化模型
    yolov3 = model.yolov3.YoloV3(Config)

    # 2. 遍历数据集
    EPOCH = 1
    BATCH_SIZE = 10

    voc_dataloader = dataset.bak_voc_dataset.get_voc_train_dataloader(
        config=Config,
        batch_size=BATCH_SIZE
    )

    for epoch in range(EPOCH):
        print("Epoch:", epoch)
        for step, (tensord_images, tensord_target_list) in enumerate(voc_dataloader):
            print("step:", step)
            for batch_index in range(BATCH_SIZE):
                print("batch:", batch_index)
                print(tensord_images[batch_index].shape)
                # exit(13)
                # 3. 预测结果并比较
                predict_image = yolov3.predict(tensord_images[batch_index], tensord_target_list[batch_index])
                predict_image.show()

            exit(-1)

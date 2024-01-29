rain_acdc_type = "CityscapesDataset"
rain_acdc_root = "data/acdc/"
rain_acdc_crop_size = (1024, 1024)
rain_acdc_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(
        type="RandomChoiceResize",
        scales=[int(1080 * x * 0.1) for x in range(5, 21)],
        resize_type="ResizeShortestEdge",
        max_size=2048,
    ),
    dict(type="RandomCrop", crop_size=rain_acdc_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
rain_acdc_val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1920, 1080), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
rain_acdc_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1920, 1080), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="PackSegInputs"),
]
train_rain_acdc = dict(
    type=rain_acdc_type,
    data_root=rain_acdc_root,
    data_prefix=dict(
        img_path="rgb_anon/rain/train",
        seg_map_path="gt/rain/train",
    ),
    img_suffix="_rgb_anon.png",
    seg_map_suffix="_gt_labelTrainIds.png",
    pipeline=rain_acdc_train_pipeline,
)
val_rain_acdc = dict(
    type=rain_acdc_type,
    data_root=rain_acdc_root,
    data_prefix=dict(
        img_path="rgb_anon/rain/val",
        seg_map_path="gt/rain/val",
    ),
    img_suffix="_rgb_anon.png",
    seg_map_suffix="_gt_labelTrainIds.png",
    pipeline=rain_acdc_val_pipeline,
)
test_rain_acdc = dict(
    type=rain_acdc_type,
    data_root=rain_acdc_root,
    data_prefix=dict(
        img_path="rgb_anon/rain/test",
        seg_map_path="gt/rain/test",
    ),
    img_suffix="_rgb_anon.png",
    seg_map_suffix="_gt_labelTrainIds.png",
    pipeline=rain_acdc_test_pipeline,
)


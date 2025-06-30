
from isegm.utils.exp_imports.default import *
from isegm.data.datasets.oct import OCTDataset
from isegm.data.points_sampler import *
from isegm.engine.trainer_ICL_mfp import *
from isegm.model.is_ukan_hicam_mfp import UKANModel_hicam_fusion_mfp
MODEL_NAME = 'oct_ikan'


def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg)


def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (256, 256)
    model_cfg.num_max_points = 24
    model_cfg.mfp_N = 100
    model_cfg.mfp_R_max = 25
    model_cfg.no_kan = True

    model = UKANModel_hicam_fusion_mfp(num_classes=1, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3, embed_dims=[256, 320, 512], no_kan=model_cfg.no_kan,
    drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1],use_rgb_conv=True, use_disks=True, norm_radius=5,
    with_prev_mask=True,N=model_cfg.mfp_N,R_max=model_cfg.mfp_R_max)

    model.to(cfg.device)
    # model.apply(initializer.XavierGluon(rnd_type='gaussian', magnitude=2.0))
    # model.feature_extractor.load_pretrained_weights(cfg.IMAGENET_PRETRAINED_MODELS.HRNETV2_W18)

    return model, model_cfg


def train(model, cfg, model_cfg):
    cfg.batch_size = 32 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.instance_loss =SoftIoU()
    loss_cfg.instance_loss_weight = 1.0
    loss_cfg.instance_aux_loss = SigmoidBinaryCrossEntropyLoss()
    loss_cfg.instance_aux_loss_weight = 0.0

    train_augmentator = Compose([
        UniformRandomResize(scale_range=(0.75, 1.40)),
        HorizontalFlip(),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size),
        RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75)
    ], p=1.0)

    val_augmentator = Compose([
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size)
    ], p=1.0)

    points_sampler = MultiPointSampler(model_cfg.num_max_points, prob_gamma=0.80,
                                        merge_objects_prob=0.15,
                                        max_num_merged_objects=2)

    trainset = OCTDataset(  # Use OCTDataset instead of SBDDataset
        cfg.OCT_PATH,  # Make sure the path is correct
        split='train',
        augmentator=train_augmentator,
        min_object_area=80,  # Adjust this parameter if necessary
        keep_background_prob=0.0,
        points_sampler=points_sampler,
    )

    valset = OCTDataset(  # Use OCTDataset for validation set
        cfg.OCT_PATH,  # Make sure the path is correct
        split='val',
        augmentator=val_augmentator,
        min_object_area=80,  # Adjust this parameter if necessary
        points_sampler=points_sampler,
        #epoch_len=2000
    )

    optimizer_params = {
        'lr': 5e-4, 'betas': (0.9, 0.999), 'eps': 1e-8
    }

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[200, 220], gamma=0.1)
    trainer = ISTrainer_mfp(model, cfg, model_cfg, loss_cfg,
                        trainset, valset,
                        optimizer='adam',
                        optimizer_params=optimizer_params,
                        lr_scheduler=lr_scheduler,
                        checkpoint_interval=[(0, 5), (200, 1)],
                        image_dump_interval=300,
                        metrics=[AdaptiveIoU()],
                        max_interactive_points=model_cfg.num_max_points,
                        max_num_next_clicks=3,
                        use_iterloss=True,
                        iterloss_weights=[1,2,3],
                        use_random_clicks=True,
                        N=model_cfg.mfp_N,
                        R_max=model_cfg.mfp_R_max)
    trainer.run(num_epochs=200)
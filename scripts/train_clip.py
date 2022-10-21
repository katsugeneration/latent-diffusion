import argparse, os, sys, glob, datetime, yaml
import torch
import time
import itertools
import numpy as np
from tqdm.auto import trange

from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from ldm.modules.losses.clip_loss import CLIPLoss
from ldm.modules.image_degradation import degradation_fn_bsr_light


class LocalImageDataset(Dataset):
    def __init__(self, img_dir, resolution, downscale_f):
        self.img_dir = img_dir
        valid_exts = [".png", ".jpg", ".jpeg"]
        self.file_list = [
            os.path.join(img_dir, file_name)
            for file_name in os.listdir(img_dir)
            if os.path.splitext(file_name)[1].lower() in valid_exts
        ]
        self.resolution = resolution
        self.downscale_f = downscale_f

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.file_list[idx])
        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.resolution, resample=Image.LANCZOS)
        image = np.array(image)
        lr_image = degradation_fn_bsr_light(image, sf=self.downscale_f)["image"]

        image = image.astype(np.float32) / 127.5 - 1.0
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image)

        lr_image = lr_image.astype(np.float32) / 127.5 - 1.0
        lr_image = lr_image.transpose(2, 0, 1)
        lr_image = torch.from_numpy(lr_image)
        return image, lr_image


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1.0, 1.0)
    x = (x + 1.0) / 2.0
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def compute_clip_direction(
    style_img_dir: str,
    clip_loss_func: CLIPLoss,
    src_class: str,
    target_class: str,
    use_target_class: bool = True,
    frozen_samples: torch.Tensor = None,
):
    if style_img_dir is None:
        return

    valid_exts = [".png", ".jpg", ".jpeg"]
    file_list = [
        os.path.join(style_img_dir, file_name)
        for file_name in os.listdir(style_img_dir)
        if os.path.splitext(file_name)[1].lower() in valid_exts
    ]

    with torch.no_grad():
        if frozen_samples is not None:
            direction = clip_loss_func.compute_img2img_direction(
                frozen_samples, file_list
            )
        elif use_target_class:
            direction = clip_loss_func.compute_txt2txt_and_img_direction(
                src_class, target_class, file_list
            )
        else:
            direction = clip_loss_func.compute_txt2img_direction(src_class, file_list)
        clip_loss_func.target_direction = direction


def run(
    model,
    model_frozen,
    logdir: str,
    modeldir: str,
    iter_num: int,
    save_log_interval: int,
    save_ckpt_interval: int,
    src_class: str,
    target_class: str,
    style_img_dir: str = None,
    use_target_class: bool = True,
    lr=2.0e-06,
    batch_size=50,
    l1_w=0.0,
    clip_model="ViT-B/32",
    resolution=256,
    custom_steps=None,
    eta=None,
    only_train_output=True,
    device="cpu",
    data=None,
):
    loss_func = CLIPLoss(device, clip_model=clip_model)

    with torch.no_grad():
        frozen_sampler = DDIMSampler(model_frozen)
        frozen_sampler.make_schedule(
            ddim_num_steps=custom_steps, ddim_eta=eta, verbose=False
        )
        sampler = DDIMSampler(model)
        sampler.make_schedule(ddim_num_steps=custom_steps, ddim_eta=eta, verbose=False)

    if style_img_dir is not None:
        with torch.no_grad():
            if data is not None:
                src_samples = torch.cat([im for im, _ in data])
            else:
                raise NotImplementedError()

            compute_clip_direction(
                style_img_dir,
                loss_func,
                src_class,
                target_class,
                use_target_class,
                src_samples,
            )

    if only_train_output:
        opt = torch.optim.AdamW(
            list(
                itertools.chain.from_iterable(
                    [b.parameters() for b in model.model.diffusion_model.output_blocks]
                )
            ),
            lr=lr,
        )
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr)

    tstart = time.time()

    for step in trange(1, iter_num + 1, desc="Training Batches"):
        opt.zero_grad()

        cond = None
        with torch.no_grad():
            if data is not None:
                _, lr_image = next(iter(data))
                if model.cond_stage_key is not None:
                    if model.cond_stage_key == "LR_image":
                        if not model.cond_stage_trainable:
                            cond = model_frozen.get_learned_conditioning(
                                lr_image.to(device)
                            )
                        else:
                            cond = lr_image.to(device)
                else:
                    raise NotImplementedError()
            noise = torch.randn(
                (
                    batch_size,
                    model.channels,
                    int(resolution // 4),
                    int(resolution // 4),
                )
            ).to(device)

        with torch.no_grad():
            frozen_latent = frozen_sampler.decode(noise, cond, custom_steps)
            frozen_sample = model_frozen.decode_first_stage(frozen_latent)

        latent = sampler.decode(noise, cond, custom_steps)
        sample = model.first_stage_model.decode(latent)

        clip_loss = loss_func(frozen_sample, src_class, sample, target_class)
        l1_loss = torch.abs(frozen_sample - sample).mean()
        loss = clip_loss + l1_w * l1_loss

        loss.backward()
        opt.step()

        if step % save_log_interval == 0:
            logs = {"sample": sample, "frozen_sample": frozen_sample}
            save_logs(logs, logdir, step, key="sample")
            save_logs(logs, logdir, step, key="frozen_sample")

        if step % save_ckpt_interval == 0:
            torch.save(
                {"state_dict": model.state_dict()},
                os.path.join(
                    modeldir,
                    f"model_{src_class}_to_{target_class}_lr{lr}_l1{l1_w}_{step:06}.ckpt",
                ),
            )

    print(
        f"Train {iter_num} iterations finished in {(time.time() - tstart) / 60.:.2f} minutes."
    )


def save_logs(logs, path, step, key="sample"):
    for k in logs:
        if k == key:
            batch = logs[key]
            n_saved = 0
            for x in batch:
                img = custom_to_pil(x)
                imgpath = os.path.join(path, f"{key}_step_{step:06}_{n_saved:06}.png")
                img.save(imgpath)
                n_saved += 1


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1e-4,
    )
    parser.add_argument(
        "-l", "--logdir", type=str, nargs="?", help="extra logdir", default="none"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=10,
    )
    parser.add_argument("--batch_size", type=int, nargs="?", help="the bs", default=100)
    parser.add_argument(
        "--iter_num", type=int, nargs="?", help="num of iteration", default=200
    )
    parser.add_argument(
        "--save_log_interval",
        type=int,
        nargs="?",
        help="interval of log saves",
        default=50,
    )
    parser.add_argument(
        "--save_ckpt_interval",
        type=int,
        nargs="?",
        help="interval of ckpt saves",
        default=50,
    )
    parser.add_argument(
        "--src_class", type=str, nargs="?", help="src class text", default=None
    )
    parser.add_argument(
        "--target_class", type=str, nargs="?", help="target class text", default=None
    )
    parser.add_argument(
        "--style_img_dir",
        type=str,
        nargs="?",
        help="Style image directory path",
        default=None,
    )
    parser.add_argument(
        "--use_target_class",
        action="store_true",
        help="Use target class for compute direction",
    )
    parser.add_argument(
        "--lr", type=float, nargs="?", help="initial learning rate", default=2.0e-06
    )
    parser.add_argument(
        "--l1_w", type=float, nargs="?", help="L1 loss weight", default=0.0
    )
    parser.add_argument(
        "--only_train_output",
        action="store_true",
        help="Only training unet output block",
    )
    parser.add_argument(
        "--train_img_dir",
        type=str,
        help="Train image directory path",
        default=None,
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        nargs="?",
        help="use clip pretained model",
        default="ViT-B/32",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs="?",
        help="training target resolution",
        default=256,
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="?",
        help="random seed variables",
        default=2,
    )
    return parser


def load_model_from_config(config, sd, device):
    model = instantiate_from_config(config)
    model.load_state_dict(sd, strict=False)
    model.to(device=device)
    return model


def load_model(config, ckpt, device, eval_mode):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model, pl_sd["state_dict"], device)
    if eval_mode:
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
    return model, global_step


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = None

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    if not os.path.exists(opt.resume):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.resume):
        # paths = opt.resume.split("/")
        try:
            logdir = "/".join(opt.resume.split("/")[:-1])
            # idx = len(paths)-paths[::-1].index("logs")+1
            print(f"Logdir is {logdir}")
        except ValueError:
            paths = opt.resume.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = opt.resume
    else:
        assert os.path.isdir(opt.resume), f"{opt.resume} is not a directory"
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")

    base_configs = sorted(glob.glob(os.path.join(logdir, "config.yaml")))
    opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if opt.logdir != "none":
        locallog = logdir.split(os.sep)[-1]
        if locallog == "":
            locallog = logdir.split(os.sep)[-2]
        print(
            f"Switching logdir from '{logdir}' to '{os.path.join(opt.logdir, locallog)}'"
        )
        logdir = os.path.join(opt.logdir, locallog)

    print(config)

    model_frozen, global_step = load_model(config, ckpt, device, True)
    model, global_step = load_model(config, ckpt, device, False)
    print(f"global step: {global_step}")
    print(75 * "=")
    print("logging to:")
    logdir = os.path.join(logdir, "samples", f"{global_step:08}", now)
    imglogdir = os.path.join(logdir, "img")
    modeldir = os.path.join(logdir, "model")

    os.makedirs(imglogdir)
    os.makedirs(modeldir)
    print(logdir)
    print(75 * "=")

    data = DataLoader(
        LocalImageDataset(opt.train_img_dir, (opt.resolution, opt.resolution), 4),
        batch_size=opt.batch_size,
        shuffle=True,
    )

    run(
        model=model,
        model_frozen=model_frozen,
        logdir=imglogdir,
        modeldir=modeldir,
        iter_num=opt.iter_num,
        save_log_interval=opt.save_log_interval,
        save_ckpt_interval=opt.save_ckpt_interval,
        src_class=opt.src_class,
        target_class=opt.target_class,
        style_img_dir=opt.style_img_dir,
        use_target_class=opt.use_target_class,
        lr=opt.lr,
        batch_size=opt.batch_size,
        l1_w=opt.l1_w,
        clip_model=opt.clip_model,
        resolution=opt.resolution,
        eta=opt.eta,
        custom_steps=opt.custom_steps,
        only_train_output=opt.only_train_output,
        device=device,
        data=data,
    )

    print("done.")

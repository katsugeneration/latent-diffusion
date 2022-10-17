import argparse, os, sys, glob, datetime, yaml
import torch
import time
import numpy as np
from tqdm import trange

from omegaconf import OmegaConf
from PIL import Image

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from ldm.modules.losses.clip_loss import CLIPLoss


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


def compute_clip_direction(style_img_dir: str, clip_loss_func, src_class, target_class):
    if style_img_dir is None:
        return

    valid_exts = [".png", ".jpg", ".jpeg"]
    file_list = [
        os.path.join(style_img_dir, file_name)
        for file_name in os.listdir(style_img_dir)
        if os.path.splitext(file_name)[1].lower() in valid_exts
    ]

    with torch.no_grad():
        direction = clip_loss_func.compute_txt2txt_and_img_direction(src_class, target_class, file_list)
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
    lr=2.0e-06,
    batch_size=50,
    l1_w=0.0,
    clip_model="ViT-B/32",
    custom_steps=None,
    eta=None,
    device="cpu",
):
    loss_func = CLIPLoss(device, clip_model=clip_model)
    if style_img_dir is not None:
        compute_clip_direction(style_img_dir, loss_func, src_class, target_class)

    with torch.no_grad():
        frozen_sampler = DDIMSampler(model_frozen)
        frozen_sampler.make_schedule(
            ddim_num_steps=custom_steps, ddim_eta=eta, verbose=False
        )
        sampler = DDIMSampler(model)
        sampler.make_schedule(ddim_num_steps=custom_steps, ddim_eta=eta, verbose=False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    tstart = time.time()

    for step in trange(1, iter_num + 1, desc="Training Batches"):
        opt.zero_grad()
        noise = torch.randn(
            (
                batch_size,
                model.model.diffusion_model.in_channels,
                model.model.diffusion_model.image_size,
                model.model.diffusion_model.image_size,
            )
        ).to(device)

        with torch.no_grad():
            frozen_latent = frozen_sampler.decode(noise, None, custom_steps)
            frozen_sample = model_frozen.decode_first_stage(frozen_latent)

        latent = sampler.decode(noise, None, custom_steps)
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
                    modeldir, f"model_{src_class}_to_{target_class}_lr{lr}_{step}.ckpt"
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
                imgpath = os.path.join(path, f"{key}_step_{step}_{n_saved:06}.png")
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
        "--style_img_dir", type=str, nargs="?", help="Style image directory path", default=None
    )
    parser.add_argument(
        "--lr", type=float, nargs="?", help="initial learning rate", default=2.0e-06
    )
    parser.add_argument(
        "--l1_w", type=float, nargs="?", help="L1 loss weight", default=0.0
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        nargs="?",
        help="use clip pretained model",
        default="ViT-B/32",
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
        lr=opt.lr,
        batch_size=opt.batch_size,
        l1_w=opt.l1_w,
        clip_model=opt.clip_model,
        eta=opt.eta,
        custom_steps=opt.custom_steps,
        device=device,
    )

    print("done.")

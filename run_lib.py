"""Training and evaluation for score-based generative models. """
import os
import time
import logging
from models import ncsnpp, ddpm
import losses
import sampling
from models import model_utils as mutils
from models.ema import ExponentialMovingAverage
import sde_lib
from absl import flags
import torch
from torch.utils import tensorboard
from utils.utils import *
import utils.datasets as datasets
#import tensorflow as tf
import json
import csv
import numpy as np
from utils.calc import Evaluation_metrics

FLAGS = flags.FLAGS


def train(config, workdir):
    """Runs the training pipeline.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """

    # The directory for saving test results during training
    sample_dir = os.path.join(workdir, "samples_in_train")
    os.makedirs(sample_dir, exist_ok=True)

    tb_dir = os.path.join(workdir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = tensorboard.SummaryWriter(tb_dir)

    # Initialize model.
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate
    )
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    initial_step = int(state["step"])

    # Build pytorch dataloader for training
    train_dl = datasets.get_dataset(config, "training")

    # Create data scaler and its inverse
    scaler = get_data_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == "vpsde":
        sde = sde_lib.VPSDE(config)
    elif config.training.sde.lower() == "subvpsde":
        sde = sde_lib.subVPSDE(config)
    elif config.training.sde.lower() == "vesde":
        sde = sde_lib.VESDE(config)
    elif config.training.sde.lower() == "hfssde":
        sde = sde_lib.HFS_SDE(config)
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(
        config,
        sde,
        train=True,
        optimize_fn=optimize_fn,
        reduce_mean=reduce_mean,
        continuous=continuous,
        likelihood_weighting=likelihood_weighting,
    )

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    logging.info("Starting training loop at step %d." % (initial_step,))

    for epoch in range(config.training.epochs):
        loss_sum = 0
        for step, batch in enumerate(train_dl):
            t0 = time.time()
            k0, csm = batch
            # TODO: mask condition
            label = Emat_xyt_complex(k0, True, csm, 1)  # 1x1x320x320
            label = c2r(label).type(torch.FloatTensor).to(config.device)
            label = scaler(label)

            # Execute one training step
            loss = train_step_fn(state, label)
            loss_sum += loss

            if step % 50 == 0:
                param_num = sum(param.numel() for param in state["model"].parameters())
                print(
                    "Epoch",
                    epoch + 1,
                    "/",
                    config.training.epochs,
                    "Step",
                    step,
                    "loss = ",
                    loss.cpu().data.numpy(),
                    "loss mean =",
                    loss_sum.cpu().data.numpy() / (step + 1),
                    "time",
                    time.time() - t0,
                    "param_num",
                    param_num,
                    flush=True,
                )

        print(
            "[EPOCH END]",
            "Epoch",
            epoch + 1,
            "/",
            config.training.epochs,
            "loss mean =",
            loss_sum.cpu().data.numpy() / (step + 1),
            flush=True,
        )

        # Save a checkpoint for every 5 epochs,改过优化参数，这个得改改
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                os.path.join(checkpoint_dir, f"checkpoint_{epoch + 1}.pth"), state
            )


def sample(config, workdir):
    """Generate samples.

    Args:
      config: Configuration to use.
      workdir: Working directory.
    """
    # Initialize model
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate
    )
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    checkpoint_dir = os.path.join(workdir, "checkpoints")

    # Support both:
    # 1) sampling.ckpt = "190"
    # 2) sampling.ckpt = "/abs/path/to/checkpoint_190.pth"
    ckpt_cfg = str(config.sampling.ckpt)

    if ckpt_cfg.endswith(".pth") or os.path.isabs(ckpt_cfg):
        ckpt_path = ckpt_cfg
    else:
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{ckpt_cfg}.pth")

    state = restore_checkpoint(ckpt_path, state, device=config.device)
    print("load weights:", ckpt_path)

    if FLAGS.config.sampling.datashift == "head":
        SAMPLING_FOLDER_ID = "_".join(
            [
                FLAGS.config.sampling.acc,
                FLAGS.config.sampling.acs,
                FLAGS.config.sampling.mask_type,
                "ckpt",
                str(config.sampling.ckpt),
                FLAGS.config.sampling.predictor,
                FLAGS.config.training.mean_equal,
                FLAGS.config.sampling.datashift,
                FLAGS.config.sampling.fft,
                str(config.sampling.snr),
                "predictor_mse",
                str(FLAGS.config.sampling.mse),
                "corrector_mse",
                str(FLAGS.config.sampling.corrector_mse),
                str(FLAGS.config.data.centered),
                str(
                    FLAGS.config.sampling.N
                    if FLAGS.config.sampling.accelerated_sampling
                    else ""
                ),
                "seed",
                str(FLAGS.config.seed),
            ]
        )
        test_dl = datasets.get_dataset(
            config, "datashift"
        )  # mode=test:90多张图，modex=sample:一张图，第十张
    elif FLAGS.config.sampling.datashift == "photom":
        SAMPLING_FOLDER_ID = "_".join(
            [
                FLAGS.config.sampling.acc,
                FLAGS.config.sampling.acs,
                FLAGS.config.sampling.mask_type,
                "ckpt",
                str(config.sampling.ckpt),
                FLAGS.config.sampling.predictor,
                FLAGS.config.training.mean_equal,
                FLAGS.config.sampling.datashift,
                FLAGS.config.sampling.fft,
                str(config.sampling.snr),
                "predictor_mse",
                str(FLAGS.config.sampling.mse),
                "corrector_mse",
                str(FLAGS.config.sampling.corrector_mse),
                str(FLAGS.config.data.centered),
                str(
                    FLAGS.config.sampling.N
                    if FLAGS.config.sampling.accelerated_sampling
                    else ""
                ),
                "photom",
                "seed",
                str(FLAGS.config.seed),
            ]
        )
        test_dl = datasets.get_dataset(
            config, "photom"
        )  # mode=test:90多张图，modex=sample:一张图，第十张
    else:
        SAMPLING_FOLDER_ID = "_".join(
            [
                FLAGS.config.sampling.acc,
                FLAGS.config.sampling.acs,
                FLAGS.config.sampling.mask_type,
                "ckpt",
                str(config.sampling.ckpt),
                FLAGS.config.sampling.predictor,
                FLAGS.config.training.mean_equal,
                str(config.sampling.snr),
                "predictor_mse",
                str(FLAGS.config.sampling.mse),
                "corrector_mse",
                str(FLAGS.config.sampling.corrector_mse),
                str(
                    FLAGS.config.data.centered,
                ),
                str(
                    FLAGS.config.sampling.N
                    if FLAGS.config.sampling.accelerated_sampling
                    else ""
                ),
                "--",
                "seed",
                str(FLAGS.config.seed),
            ]
        )
        test_dl = datasets.get_dataset(
            config, "test"
        )  # mode=test:90多张图，modex=sample:一张图，第十张

    FLAGS.config.sampling.folder = os.path.join(FLAGS.workdir, SAMPLING_FOLDER_ID)
    os.makedirs(FLAGS.config.sampling.folder, exist_ok=True)
    metrics_records = []
    zf_metrics_records = []

    # Create data scaler and its inverse
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == "vpsde":
        sde = sde_lib.VPSDE(config)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == "subvpsde":
        sde = sde_lib.subVPSDE(config)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == "vesde":
        sde = sde_lib.VESDE(config)
        sampling_eps = 1e-5
    elif config.training.sde.lower() == "hfssde":
        sde = sde_lib.HFS_SDE(config)
        sampling_eps = 1e-3  # TODO
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    atb_mask = get_mask(config, "sample")
    train_mask = get_mask(config, "sde")

    # Build the sampling function when sampling is enabled

    sampling_shape = (
        config.sampling.batch_size,
        config.data.num_channels,
        config.data.image_size,
        config.data.image_size,
    )
    sampling_fn = sampling.get_sampling_fn(
        config, sde, sampling_shape, inverse_scaler, sampling_eps, atb_mask, train_mask
    )

    for index, point in enumerate(test_dl):
        print("---------------------------------------------")
        print("---------------- point:", index, "------------------")
        print("---------------------------------------------")

        k0, csm = point
        k0 = k0.to(config.device)
        csm = csm.to(config.device)

        # 保存 complex 版本的 csm，后面算 label / zf 都用它
        csm_complex = csm

        # GT label
        label = Emat_xyt_complex(k0, True, csm_complex, 1.0).to(config.device)

        label_dir = os.path.join("results", FLAGS.config.sampling.datashift)
        os.makedirs(label_dir, exist_ok=True)
        save_mat(label_dir, label.to(label), "label", index, normalize=False)

        # undersampled k-space
        atb = k0 * atb_mask

        # zero-filled reconstruction，用 complex csm 算
        zf_complex = Emat_xyt_complex(atb, True, csm_complex, 1.0)

        print("[DBG BEFORE SAMPLE] label abs max/mean:",
              torch.abs(label).max().item(),
              torch.abs(label).mean().item())

        print("[DBG BEFORE SAMPLE] zf abs max/mean:",
              torch.abs(zf_complex).max().item(),
              torch.abs(zf_complex).mean().item())

        print("ZF metrics:")
        zf_ssim, zf_psnr, zf_nmse = Evaluation_metrics(label, zf_complex, False)

        zf_psnr = float(zf_psnr)
        zf_ssim = float(zf_ssim)
        zf_nmse = float(zf_nmse)

        zf_metrics_records.append({
            "index": int(index),
            "psnr": zf_psnr,
            "ssim": zf_ssim,
            "nmse": zf_nmse,
        })

        # 给 sampling 准备 atb_to_image
        atb_to_image = (
            c2r(Emat_xyt_complex(atb, True, csm_complex, 1))
            .type(torch.FloatTensor)
            .to(config.device)
        )

        # sampling 里面需要 real/imag csm
        csm_real = c2r(csm_complex).type(torch.FloatTensor).to(config.device)

        # diffusion reconstruction
        recon, n = sampling_fn(score_model, atb, atb_to_image, csm_real)
        recon = r2c(recon)

        print("[DBG AFTER SAMPLE] label abs max/mean:",
              torch.abs(label).max().item(),
              torch.abs(label).mean().item())

        print("[DBG AFTER SAMPLE] zf abs max/mean:",
              torch.abs(zf_complex).max().item(),
              torch.abs(zf_complex).mean().item())

        print("[DBG AFTER SAMPLE] recon abs max/mean:",
              torch.abs(recon).max().item(),
              torch.abs(recon).mean().item())

        save_mat(
            FLAGS.config.sampling.folder,
            recon.to(recon),
            "recon",
            index,
            normalize=False,
        )

        print("Recon metrics:")
        ssim, psnr, nmse = Evaluation_metrics(
            label,
            recon,
            True if FLAGS.config.sampling.datashift == "photom" else False,
        )

        psnr = float(psnr)
        ssim = float(ssim)
        nmse = float(nmse)

        metrics_records.append({
            "index": int(index),
            "psnr": psnr,
            "ssim": ssim,
            "nmse": nmse,
        })

        print(
            f"mse_{config.sampling.mse}_snr_{config.sampling.snr}_cmse_{config.sampling.corrector_mse}:"
        )
        print("nmse:", nmse)
        print("ssim:", ssim)
        print("psnr:", psnr)

    if len(metrics_records) > 0:
        psnrs = np.array([m["psnr"] for m in metrics_records], dtype=np.float64)
        ssims = np.array([m["ssim"] for m in metrics_records], dtype=np.float64)
        nmses = np.array([m["nmse"] for m in metrics_records], dtype=np.float64)

        zf_psnrs = np.array([m["psnr"] for m in zf_metrics_records], dtype=np.float64)
        zf_ssims = np.array([m["ssim"] for m in zf_metrics_records], dtype=np.float64)
        zf_nmses = np.array([m["nmse"] for m in zf_metrics_records], dtype=np.float64)

        summary = {
            "ckpt": str(config.sampling.ckpt),
            "num_samples": int(len(metrics_records)),

            "zf_psnr_mean": float(np.mean(zf_psnrs)),
            "zf_psnr_std": float(np.std(zf_psnrs)),
            "zf_ssim_mean": float(np.mean(zf_ssims)),
            "zf_ssim_std": float(np.std(zf_ssims)),
            "zf_nmse_mean": float(np.mean(zf_nmses)),
            "zf_nmse_std": float(np.std(zf_nmses)),

            "recon_psnr_mean": float(np.mean(psnrs)),
            "recon_psnr_std": float(np.std(psnrs)),
            "recon_ssim_mean": float(np.mean(ssims)),
            "recon_ssim_std": float(np.std(ssims)),
            "recon_nmse_mean": float(np.mean(nmses)),
            "recon_nmse_std": float(np.std(nmses)),

            "delta_psnr": float(np.mean(psnrs) - np.mean(zf_psnrs)),
            "delta_ssim": float(np.mean(ssims) - np.mean(zf_ssims)),
            "delta_nmse": float(np.mean(nmses) - np.mean(zf_nmses)),
        }

        print("\n========== Sampling Summary ==========", flush=True)
        print(f"Checkpoint: {config.sampling.ckpt}", flush=True)
        print(f"Num samples: {summary['num_samples']}", flush=True)
        print(
            f"ZF:    PSNR {summary['zf_psnr_mean']:.2f} ± {summary['zf_psnr_std']:.2f}, "
            f"SSIM {summary['zf_ssim_mean']:.4f} ± {summary['zf_ssim_std']:.4f}, "
            f"NMSE {summary['zf_nmse_mean']:.4f} ± {summary['zf_nmse_std']:.4f}",
            flush=True,
        )
        print(
            f"Recon: PSNR {summary['recon_psnr_mean']:.2f} ± {summary['recon_psnr_std']:.2f}, "
            f"SSIM {summary['recon_ssim_mean']:.4f} ± {summary['recon_ssim_std']:.4f}, "
            f"NMSE {summary['recon_nmse_mean']:.4f} ± {summary['recon_nmse_std']:.4f}",
            flush=True,
        )
        print(
            f"Delta: PSNR {summary['delta_psnr']:+.2f}, "
            f"SSIM {summary['delta_ssim']:+.4f}, "
            f"NMSE {summary['delta_nmse']:+.4f}",
            flush=True,
        )
        print("======================================\n", flush=True)

        metrics_json = os.path.join(FLAGS.config.sampling.folder, "metrics_summary.json")
        with open(metrics_json, "w") as f:
            json.dump(
                {
                    "zf_per_sample": zf_metrics_records,
                    "recon_per_sample": metrics_records,
                    "summary": summary,
                },
                f,
                indent=2,
            )

        metrics_csv = os.path.join(FLAGS.config.sampling.folder, "metrics_per_sample.csv")
        with open(metrics_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "index",
                    "zf_psnr", "zf_ssim", "zf_nmse",
                    "recon_psnr", "recon_ssim", "recon_nmse",
                    "delta_psnr", "delta_ssim", "delta_nmse",
                ],
            )
            writer.writeheader()
            for zf_row, recon_row in zip(zf_metrics_records, metrics_records):
                writer.writerow({
                    "index": recon_row["index"],
                    "zf_psnr": zf_row["psnr"],
                    "zf_ssim": zf_row["ssim"],
                    "zf_nmse": zf_row["nmse"],
                    "recon_psnr": recon_row["psnr"],
                    "recon_ssim": recon_row["ssim"],
                    "recon_nmse": recon_row["nmse"],
                    "delta_psnr": recon_row["psnr"] - zf_row["psnr"],
                    "delta_ssim": recon_row["ssim"] - zf_row["ssim"],
                    "delta_nmse": recon_row["nmse"] - zf_row["nmse"],
                })

        print(f"[Saved] {metrics_json}", flush=True)
        print(f"[Saved] {metrics_csv}", flush=True)
import os
import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


def to_complex(x):
    """
    支持两种格式：
    1. complex64 / complex128
    2. 最后一维为 2 的 real/imag 格式
    """
    x = np.asarray(x)
    if np.iscomplexobj(x):
        return x

    if x.shape[-1] == 2:
        return x[..., 0] + 1j * x[..., 1]

    return x.astype(np.complex64)


def ifft2c(kspace):
    """
    centered 2D IFFT.
    输入 shape: [..., H, W]
    输出 shape: [..., H, W]
    """
    return np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(kspace, axes=(-2, -1)),
            axes=(-2, -1),
            norm="ortho",
        ),
        axes=(-2, -1),
    )


def center_crop(x, crop_size):
    """
    对最后两个维度中心裁剪。
    支持 shape: [H, W] 或 [C, H, W]
    """
    if crop_size is None or crop_size <= 0:
        return x

    h, w = x.shape[-2:]
    crop_h = crop_size
    crop_w = crop_size

    if crop_h > h or crop_w > w:
        raise ValueError(
            f"crop_size={crop_size} 太大，但当前图像尺寸是 {h}x{w}。"
            f"例如你的 brain 是 640x320，就不能裁成 384x384。"
        )

    top = (h - crop_h) // 2
    left = (w - crop_w) // 2

    return x[..., top:top + crop_h, left:left + crop_w]


def normalize_img(x, percentile=99.5):
    """
    可视化归一化，只用于显示，不用于算指标。
    """
    x = np.abs(x)
    vmax = np.percentile(x, percentile)
    vmin = np.percentile(x, 1.0)

    if vmax <= vmin:
        return np.zeros_like(x)

    x = (x - vmin) / (vmax - vmin)
    x = np.clip(x, 0, 1)
    return x


def save_grid(imgs, titles, save_path, nrow=4, cmap="gray", suptitle=None):
    """
    imgs: list of 2D images
    titles: list of titles
    """
    n = len(imgs)
    ncol = nrow
    nrow_fig = int(np.ceil(n / ncol))

    plt.figure(figsize=(3.0 * ncol, 3.0 * nrow_fig))

    for i, img in enumerate(imgs):
        plt.subplot(nrow_fig, ncol, i + 1)
        plt.imshow(img, cmap=cmap)
        plt.title(titles[i], fontsize=9)
        plt.axis("off")

    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("========== Load kspace ==========")
    with h5py.File(args.kspace, "r") as f:
        print("kspace keys:", list(f.keys()))
        kspace_all = f["kspace"]
        print("kspace full shape:", kspace_all.shape, kspace_all.dtype)

        num_slices = kspace_all.shape[0]
        if args.slice < 0:
            slice_id = num_slices // 2
        else:
            slice_id = args.slice

        if slice_id >= num_slices:
            raise ValueError(f"slice={slice_id} 超出范围，总 slice 数为 {num_slices}")

        kspace = to_complex(kspace_all[slice_id])

        recon_rss_from_file = None
        if "reconstruction_rss" in f.keys():
            recon_rss_from_file = np.asarray(f["reconstruction_rss"][slice_id])
            print("found reconstruction_rss:", recon_rss_from_file.shape)

    print("selected kspace shape:", kspace.shape, kspace.dtype)

    print("\n========== Load sensitivity maps ==========")
    with h5py.File(args.maps, "r") as f:
        print("maps keys:", list(f.keys()))
        smaps_all = f["s_maps"]
        print("s_maps full shape:", smaps_all.shape, smaps_all.dtype)

        smaps = to_complex(smaps_all[slice_id])

    # 有些 BART 输出可能多一个 singleton 维度，这里兜一下
    smaps = np.squeeze(smaps)

    print("selected smaps shape:", smaps.shape, smaps.dtype)

    if kspace.ndim != 3:
        raise ValueError(f"kspace slice 应该是 [coil, H, W]，但现在是 {kspace.shape}")

    if smaps.ndim != 3:
        raise ValueError(f"s_maps slice 应该是 [coil, H, W]，但现在是 {smaps.shape}")

    if kspace.shape != smaps.shape:
        raise ValueError(
            f"kspace 和 s_maps shape 不一致：kspace={kspace.shape}, s_maps={smaps.shape}"
        )

    num_coils, h, w = kspace.shape
    num_show = min(args.num_coils, num_coils)

    print("\n========== Basic info ==========")
    print("slice_id:", slice_id)
    print("num_coils:", num_coils)
    print("raw image size:", h, w)
    print("show coils:", num_show)
    print("crop size:", args.crop)

    print("\n========== IFFT coil images ==========")
    coil_imgs = ifft2c(kspace)  # [coil, H, W]

    # 图像域中心裁剪：coil image 和 smaps 必须同步裁剪
    coil_imgs_crop = center_crop(coil_imgs, args.crop)
    smaps_crop = center_crop(smaps, args.crop)

    if recon_rss_from_file is not None:
        recon_rss_from_file = center_crop(recon_rss_from_file, args.crop)

    print("coil_imgs_crop:", coil_imgs_crop.shape)
    print("smaps_crop:", smaps_crop.shape)

    print("\n========== Combine coils ==========")
    rss_gt = np.sqrt(np.sum(np.abs(coil_imgs_crop) ** 2, axis=0))

    denom = np.sum(np.abs(smaps_crop) ** 2, axis=0) + 1e-8
    sense_img = np.sum(np.conj(smaps_crop) * coil_imgs_crop, axis=0) / denom

    map_norm = np.sqrt(np.sum(np.abs(smaps_crop) ** 2, axis=0))

    print("rss_gt:", rss_gt.shape, rss_gt.dtype)
    print("sense_img:", sense_img.shape, sense_img.dtype)
    print("map_norm:", map_norm.shape, map_norm.dtype)
    print("smaps nan:", np.isnan(smaps_crop).any(), "inf:", np.isinf(smaps_crop).any())
    print("smaps abs max:", np.abs(smaps_crop).max(), "mean:", np.abs(smaps_crop).mean())

    # 1. 展示 16 个 coil image
    coil_imgs_vis = []
    coil_titles = []
    for c in range(num_show):
        coil_imgs_vis.append(normalize_img(coil_imgs_crop[c]))
        coil_titles.append(f"coil image {c}")

    save_grid(
        coil_imgs_vis,
        coil_titles,
        out_dir / f"slice_{slice_id:03d}_coil_images_{num_show}.png",
        nrow=4,
        cmap="gray",
        suptitle=f"Coil images, slice {slice_id}",
    )

    # 2. 展示 16 个 sensitivity map magnitude
    smap_abs_vis = []
    smap_titles = []
    for c in range(num_show):
        smap_abs_vis.append(normalize_img(smaps_crop[c]))
        smap_titles.append(f"smap abs {c}")

    save_grid(
        smap_abs_vis,
        smap_titles,
        out_dir / f"slice_{slice_id:03d}_smap_abs_{num_show}.png",
        nrow=4,
        cmap="gray",
        suptitle=f"Sensitivity map magnitude, slice {slice_id}",
    )

    # 3. 展示 16 个 sensitivity map phase
    smap_phase_vis = []
    smap_phase_titles = []
    for c in range(num_show):
        phase = np.angle(smaps_crop[c])
        smap_phase_vis.append(phase)
        smap_phase_titles.append(f"smap phase {c}")

    save_grid(
        smap_phase_vis,
        smap_phase_titles,
        out_dir / f"slice_{slice_id:03d}_smap_phase_{num_show}.png",
        nrow=4,
        cmap="twilight",
        suptitle=f"Sensitivity map phase, slice {slice_id}",
    )

    # 4. 展示 GT / RSS / SENSE combine / map norm
    combined_imgs = [
        normalize_img(rss_gt),
        normalize_img(sense_img),
        normalize_img(map_norm),
    ]
    combined_titles = [
        "GT from full kspace: RSS",
        "SENSE combine with s_maps",
        "sqrt(sum |s_maps|^2)",
    ]

    if recon_rss_from_file is not None:
        combined_imgs.insert(0, normalize_img(recon_rss_from_file))
        combined_titles.insert(0, "reconstruction_rss in h5")

    save_grid(
        combined_imgs,
        combined_titles,
        out_dir / f"slice_{slice_id:03d}_combined_gt.png",
        nrow=2,
        cmap="gray",
        suptitle=f"Combined images, slice {slice_id}",
    )

    # 5. 单独保存每个 coil 的切片图
    single_dir = out_dir / f"slice_{slice_id:03d}_single_coils"
    single_dir.mkdir(parents=True, exist_ok=True)

    for c in range(num_show):
        plt.figure(figsize=(5, 5))
        plt.imshow(normalize_img(coil_imgs_crop[c]), cmap="gray")
        plt.title(f"Coil image {c}, slice {slice_id}")
        plt.axis("off")
        plt.savefig(single_dir / f"coil_image_{c:02d}.png", dpi=200, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(5, 5))
        plt.imshow(normalize_img(smaps_crop[c]), cmap="gray")
        plt.title(f"Sensitivity map abs {c}, slice {slice_id}")
        plt.axis("off")
        plt.savefig(single_dir / f"smap_abs_{c:02d}.png", dpi=200, bbox_inches="tight")
        plt.close()

    print("\n========== Saved ==========")
    print(out_dir)
    print("1) coil images grid")
    print("2) sensitivity magnitude grid")
    print("3) sensitivity phase grid")
    print("4) combined GT/RSS/SENSE/map_norm")
    print("5) individual coil images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--kspace", type=str, required=True, help="原始 fastMRI brain kspace h5")
    parser.add_argument("--maps", type=str, required=True, help="对应的 s_maps h5")
    parser.add_argument("--slice", type=int, default=-1, help="-1 表示取中间 slice")
    parser.add_argument("--crop", type=int, default=320, help="图像域中心裁剪尺寸；设为 0 表示不裁剪")
    parser.add_argument("--num_coils", type=int, default=16, help="展示前多少个 coil")
    parser.add_argument("--out_dir", type=str, default="./debug_brain_maps_vis")

    args = parser.parse_args()
    main(args)
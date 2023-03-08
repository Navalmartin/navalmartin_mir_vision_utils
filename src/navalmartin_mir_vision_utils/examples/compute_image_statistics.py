from pathlib import Path
from navalmartin_mir_vision_utils import load_img, ImageLoadersEnumType
from navalmartin_mir_vision_utils.statistics import compute_image_statistics, fit_gaussian_distribution_on_image

if __name__ == '__main__':

    image_path = Path("/home/alex/qi3/mir-engine/datasets/cracks_v_3_id_8/train/cracked/img_9_9.jpg")
    image = load_img(path=image_path, loader=ImageLoadersEnumType.PIL)

    print(f"Image size {image.size}")
    print(f"Image bands {image.getbands()}")

    image_stats = compute_image_statistics(image)
    print(f"Image channel mean {image_stats.mean}")
    print(f"Image channel var {image_stats.var}")
    print(f"Image channel median {image_stats.median}")

    channels_fit = fit_gaussian_distribution_on_image(image=image)
    print(f"Gaussian distribution channel fit: {channels_fit}")


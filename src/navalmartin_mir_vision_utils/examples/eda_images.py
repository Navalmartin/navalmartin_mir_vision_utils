"""example image_pipeline
simple example that showcases how to create
pipelines for images

"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from navalmartin_mir_vision_utils import ImageLoadersEnumType
from navalmartin_mir_vision_utils.mir_vision_io.file_utils import (rename_files_with_counter,
                                                                   get_all_files,
                                                                   copy_files_from_to_dir)
from navalmartin_mir_vision_utils.image_loaders import load_images

from navalmartin_mir_vision_utils.statistics import compute_image_statistics

# Apply the default theme
sns.set_theme()

def copy_images(images_input_path: Path,
                images_output_path: Path) -> None:

    files = get_all_files(dir_path=images_input_path,
                          file_formats=[".jpg", ".png", ".jpeg"])

    print(f"Number of files found in {images_input_path}, {len(files)}")

    copy_files_from_to_dir(sources=files,
                           dst_dir=images_output_path,
                           build_dst_dir=True)

    # rename the files in the curated dataset
    rename_files_with_counter(base_path_src=images_output_path,
                              base_path_dist=images_output_path,
                              common_filename="img_",
                              file_formats=[".jpg", ".png", ".jpeg"],
                              start_counter=0,
                              do_copy=False)


if __name__ == '__main__':
    images_input_path = Path("/home/alex/qi3/mir_datasets/vessels/trimaran")
    images_output_path = Path("/home/alex/qi3/mir_datasets/vessels/trimaran/curated")

    copy_images(images_input_path=images_input_path,
                images_output_path=images_output_path)

    #  get all the image files
    images = load_images(path=images_output_path,
                         loader=ImageLoadersEnumType.PIL)

    means = {"R": [], "G": [], "B": []}
    variances = {"R": [], "G": [], "B": []}

    for img in images:
        stats = compute_image_statistics(image=img)
        means["R"].append(stats.mean[0])
        means["G"].append(stats.mean[1])
        means["B"].append(stats.mean[2])

        variances["R"].append(stats.var[0])
        variances["G"].append(stats.var[1])
        variances["B"].append(stats.var[2])

    plt.hist(means["R"], bins=20, color="red")
    plt.hist(means["G"], bins=20, color="green")
    plt.hist(means["B"], bins=20, color="blue")
    plt.show()










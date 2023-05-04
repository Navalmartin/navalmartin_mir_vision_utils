from pathlib import Path

from navalmartin_mir_vision_utils import (plot_pil_image,
                                          plot_pytorch_image,
                                          is_valid_pil_image_from_bytes_string,
                                          load_img,
                                          ImageLoadersEnumType,
                                          pil_image_to_bytes_string,
                                          pil_to_torch_tensor,
                                          pils_to_torch_tensor)
from navalmartin_mir_vision_utils.mir_vision_config import WITH_TORCH


if __name__ == '__main__':

    image_path = Path("../../tests/test_data/img_18_3.jpg")
    image = load_img(path=image_path, loader=ImageLoadersEnumType.PIL)

    print("Showing image from normal loading...")
    plot_pil_image(image=image, title="Image from file")

    image_bytes = pil_image_to_bytes_string(image=image)
    image = is_valid_pil_image_from_bytes_string(image_byte_string=image_bytes)

    print("Showing image from bytes...")
    plot_pil_image(image=image, title="Image from byte string")

    if WITH_TORCH:
        # convert the image to a torch tensor
        torch_tensor = pil_to_torch_tensor(image=image)

        print(f"Torch tensor size={torch_tensor.size()}")
        plot_pytorch_image(image=torch_tensor, title="PyTorch tensor image")

        # load another image
        image_path = Path("../../tests/test_data/img_2_6.jpg")
        image_2 = load_img(path=image_path, loader=ImageLoadersEnumType.PIL)
        plot_pil_image(image=image_2, title="Second image")

        # make the two images torch tensors
        torch_tensors = pils_to_torch_tensor(images=[image, image_2])
        print(f"Torch tensor size={torch_tensors.size()}")






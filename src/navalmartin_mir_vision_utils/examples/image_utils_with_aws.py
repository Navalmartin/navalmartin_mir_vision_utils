import os

from pathlib import Path
from dotenv import load_dotenv
from navalmartin_mir_vision_utils import (is_valid_pil_image_file,
                                          get_pil_image_size,
                                          create_thumbnail_from_pil_image,
                                          get_image_metadata,
                                          get_image_info,
                                          remove_metadata_from_image,
                                          get_pil_image_as_bytes_io_value)

from navalmartin_mir_aws_utils.aws_credentials import AWSCredentials_S3
from navalmartin_mir_aws_utils.boto3_client import get_aws_s3_client

# where .env file is located
DOTENV_PATH = Path("/home/alex/qi3/mir_engine_rest_api/.env")

if __name__ == '__main__':

    # where the image is located
    image_file = Path("/home/alex/qi3/mir_vision_utils/test_data/P1030888.JPG")
    image = is_valid_pil_image_file(image=image_file)
    if image is not None:
        print("The provided image is OK")
        image_size = get_pil_image_size(image=image)
        print(f"Image size is {image_size}")
    else:
        print("The provided image is NOT OK")

    print(f"Print full image size: {image.size}")
    image_info = get_image_info(image=image)
    print(image_info)

    print(f"Image format {image.format}")
    image_metadata = get_image_metadata(image)
    print(image_metadata)

    new_image = remove_metadata_from_image(image, new_filename=None)

    # verify that removing the metadata removes format
    # and image.info is empty
    print(f"New image format {new_image.format}")
    print(f"Remove metadata image info {new_image.info}")

    # reset these as they are needed to
    # visualize the image
    new_image.info = image_info
    new_image.format = image_info['format']
    new_image.mode = image_info['mode']
    new_image.palette = image_info['palette']

    # check that these are set properly
    print(f"New image format {new_image.format}")
    print(f"Set image info {new_image.info}")
    print(f"Print image no-meta size: {new_image.size}")

    thumbnail_img = create_thumbnail_from_pil_image(image=new_image, max_size=(500, 500))
    print(f"Print thumbnail image no-meta size: {thumbnail_img.size}")

    # the filename of the file to upload
    img_filename = "my_test_img.jpg"

    new_image = get_pil_image_as_bytes_io_value(image=new_image)

    load_dotenv(DOTENV_PATH)
    aws_s3_bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
    aws_region = os.getenv("AWS_S3_BUCKET_REGION_NAME")
    credentials = AWSCredentials_S3(aws_s3_bucket_name=aws_s3_bucket_name,
                                    aws_region=aws_region)
    s3_client = get_aws_s3_client(credentials)

    response = s3_client.put_object(
        Body=new_image,
        Bucket=credentials.aws_s3_bucket_name, Key=img_filename
    )

    print(response)

from pathlib import Path
from typing import List, Any, Union
import json
import os
import csv
import pickle
import datetime
import shutil
import hashlib
import uuid


def has_suffix(filename: Union[Path, str], suffixes: List[str]) -> bool:
    """Assumes that the filename path givens has a structure
    of directories separated with '/'. The last name is assumed
    to be the filename that we look to verify the suffix

   Parameters
   ----------
   filename
   suffixes

   Returns
   -------

   """

    if isinstance(filename, Path):
        filename_copy = str(filename)
    else:
        filename_copy = filename

    filename, file_extension = os.path.splitext(filename_copy)

    if file_extension in suffixes:
        return True

    return False


def get_md5_checksum(file: Union[Path, str, bytes]):
    """Returns the MD5 checksum of the file
    in the given path. Implementation taken from:
    https://stackoverflow.com/questions/16874598/how-do-i-calculate-the-md5-checksum-of-a-file-in-python

    Parameters
    ----------
    filename

    Returns
    -------

    """

    if isinstance(file, Path):
        # Open,close, read file and calculate MD5 on its contents
        with open(file, 'rb') as file_to_check:
            # read contents of the file
            data = file_to_check.read()
            # pipe contents of the file through
            md5_returned = hashlib.md5(data).hexdigest()
            return md5_returned
    elif isinstance(file, bytes):
        md5_returned = hashlib.md5(file).hexdigest()
        return md5_returned
    elif isinstance(file, str):
        md5_returned = hashlib.md5(file).hexdigest()
        return md5_returned
    else:
        raise ValueError("Invalid 'file' type. 'file' must be either Path of bytes")


def read_json(filename: Path) -> dict:
    """Read the specified JSON file into
    a directory

    Parameters
    ----------
    filename: The path to the JSON file

    Returns
    -------

    An instance of dict with the properties
    described in the JSON file
    """

    with open(filename) as json_file:
        json_input = json.load(json_file)
        return json_input


def save_as_json(obj: Any, filename: Path) -> None:
    """Save the given object file in json

    Parameters
    ----------
    obj: The object to serialize
    filename: The path to the file

    Returns
    -------
    None
    """
    with open(filename, "w") as fp:
        json.dump(obj, fp)


def save_as_pickle(obj: Any, filename: Path,
                   protocol=pickle.HIGHEST_PROTOCOL) -> None:
    """Save the object in a pickle representation

    Parameters
    ----------
    obj: The object to save
    filename: The filename to use
    protocol: pickle protocol

    Returns
    -------

    None
    """

    with open(filename, 'wb') as outp:
        pickle.dump(obj, outp, protocol)


def create_dir_path(path: Path):
    if not os.path.exists(path=str(path)):
        os.makedirs(str(path))
    else:
        raise FileExistsError(f"File={path} already exists")


def create_dir(path: Path, dirname: str):
    """Create a directory with the given name in the specified path
    Throws a ValueError exception if the directory already exists

    Parameters
    ----------
    path: The path to create the directory
    dirname: The directory name

    Returns
    -------

    """

    dir_path = str(path) + "/" + dirname
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        raise ValueError("Could not create directory. The directory={0} exist".format(dir_path))


def save_list_as_csv(list_inst: List[Any], filename: Path, write_default_header: bool = False,
                     header: Any = None) -> None:
    """Save the given list in a csv file format

    Parameters
    ----------
    list_inst: The list to save
    filename: The filename to save at
    write_default_header: If true writes the default header which is simply
    the system date and time. In this case the row starts with a '#' mark
    header: Application defined header.
    Returns
    -------

    """
    with open(filename, "w") as fh:
        csv_writer = csv.writer(fh, delimiter=",")

        if header is not None:
            csv_writer.writerow(header)
        elif write_default_header:
            now = datetime.datetime.now()
            csv_writer.writerow("# " + str(now))

        for i, item in enumerate(list_inst):
            row = [i, item]
            csv_writer.writerow(row)


def load_list_from_csv(filename: Path) -> List[float]:
    """Loads a list form the given csv file

    Parameters
    ----------
    filename

    Returns
    -------

    A list of floats
    """
    with open(filename, 'r') as fh:
        reader = csv.reader(fh, delimiter=",")

        data: List[float] = []

        for item in reader:

            if not item:
                continue
            else:
                data.append(float((item[1])))

        return data


def copy_file_from_to(source: Path, dst: Path) -> None:
    """Copy the source image  to the dst image

    Parameters
    ----------
    source: The source image
    dst: The destination image

    Returns
    -------

    """
    shutil.copy(src=source, dst=dst)


def copy_files_from_to(sources: List[Path], dsts: List[Path]):
    for src, dist in zip(sources, dsts):
        copy_file_from_to(source=src, dst=dist)


def copy_files_from_to_dir(sources: List[Path], dst_dir: Path, build_dst_dir: bool = True):
    if not os.path.isdir(dst_dir) and not build_dst_dir:
        raise ValueError("Destination directory does not exist and build_dst_dir is False."
                         " Create destination directory or set build_dst_dir=True")

    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)

    for file in sources:
        filename = file.name
        new_dst = dst_dir / filename
        copy_file_from_to(source=file, dst=new_dst)


def move_file_from_to(source: Path, dst: Path):
    shutil.move(src=source, dst=dst)


def move_files_from_to(sources: List[Path], dsts: List[Path]):
    if len(sources) != len(dsts):
        raise ValueError("Invalid size of sources and destinations")

    for src, dist in zip(sources, dsts):
        move_file_from_to(source=src, dst=dist)


def rename_file(src: Path, dst: Path):
    os.rename(src=src, dst=dst)


def rename_files_with_counter(base_path_src: Path,
                              base_path_dist: Path,
                              common_filename: str,
                              file_formats: List[str] = [],
                              start_counter: int = 0,
                              do_copy: bool = False) -> None:

    def _do_name(filename: Path, counter: int, do_copy: bool) -> None:
        filename_, file_extension = os.path.splitext(filename)

        dst_name = common_filename + str(counter) + f"{file_extension}"
        dst = base_path_dist / dst_name

        if do_copy:
            copy_file_from_to(source=filename, dst=dst)
        else:
            move_file_from_to(source=filename, dst=dst)
        counter += 1

    filenames = get_all_files(dir_path=base_path_src,
                              file_formats=file_formats, skip_dirs=True)

    counter = start_counter
    if len(file_formats) != 0:
        for filename in filenames:

            if has_suffix(filename=filename, suffixes=file_formats):
                _do_name(filename=filename, counter=counter, do_copy=do_copy)
                counter += 1
    else:
        for filename in filenames:
            _do_name(filename=filename, counter=counter, do_copy=do_copy)
            counter += 1


def compare_dirs(dir_path_1: Path, dir_path_2: Path,
                 ignore_suffixes: bool,
                 sort_contents: bool = True) -> None:
    """Compare the contents of the two directories.
    Throws a ValueError exception is the directories
    do not have the same contents according to the specified
    criteria.

    Parameters
    ----------
    dir_path_1: The first directory
    dir_path_2: The second directory
    ignore_suffixes: Flag indicating whether file suffixes should be ignored
    sort_contents: Flag indicating whether the contents of the directories should be sorted
    before the comparison

    Returns
    -------

    None
    """

    files_1 = os.listdir(dir_path_1)
    files_2 = os.listdir(dir_path_2)

    if len(files_1) != len(files_2):
        raise ValueError(f"Directories don't have the same contents. Size {len(files_1)} != {len(files_2)}")

    if sort_contents:
        files_1.sort()
        files_2.sort()

    if ignore_suffixes:

        for file_1, file_2 in zip(files_1, files_2):
            name1 = file_1.split(".")[0]
            name2 = file_2.split(".")[0]

            if str(name1) != str(name2):
                raise ValueError(f"Names not equal: {name1} != {name2}")
    else:
        for file_1, file_2 in zip(files_1, files_2):
            if str(file_1) != str(file_2):
                raise ValueError(f"Names not equal: {file_1} != {file_2}")


def get_all_files(dir_path: Path, file_formats: List[str],
                  with_batch_structure: bool = False,
                  skip_dirs: bool = False) -> List[Path]:
    """Get the image files in the given image directory that have
    the specified image format.

    Parameters
    ----------
    dir_path
    with_batch_structure: Flag indicating that the given path is structured
    with subdirectories
    img_dir: The image directory
    file_formats: The image formats

    Returns
    -------
    An instance of List[Path]
    """

    if with_batch_structure and skip_dirs:
        raise ValueError("Data has batch structure but skip_dirs=True")

    # load the corrosion images
    img_files = os.listdir(dir_path)

    files: List[Path] = []

    for filename in img_files:
        if os.path.isfile(dir_path / filename):

            filename_, file_extension = os.path.splitext(filename)

            if file_extension in file_formats:
                files.append(dir_path / filename)
            else:
                continue

        elif with_batch_structure:
            if os.path.isdir(dir_path / filename):
                dir_files = os.listdir(dir_path / filename)

                for dir_filename in dir_files:
                    if os.path.isfile(dir_path / filename / dir_filename):
                        filename_, file_extension = os.path.splitext(dir_filename)
                        if file_extension in file_formats:
                            files.append(dir_path / filename / dir_filename)
        elif skip_dirs:
            continue
        else:
            raise ValueError(f"The given path {dir_path / filename} is a directory but no batches are assumed")

    return files

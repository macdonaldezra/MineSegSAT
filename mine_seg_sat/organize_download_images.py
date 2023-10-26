import csv
import os
import shutil


def fast_scandir(dirname: str) -> list:
    """
    Scan and return all subfolders of a directory.

    Parameters:
    dirname: root directory name to be scanned.

    Returns:
    A list of subfolders under the given directory.
    """
    subfolders = [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders


def get_folders_with_keyword(keyword: str, folders_list: list) -> list:
    """
    From a list of folders, cherry pick the ones with given keyword.

    Parameters:
    keyword: keyword that folders directory must contain.
    folders_list: list of folders to be searched.

    Returns:
    A list of paths to folders that contain the given keyword.
    """
    folders_with_keyword_list = []

    for folder in folders_list:
        if keyword in folder:
            folders_with_keyword_list.append(folder)

    return folders_with_keyword_list


def get_list_of_files_in_directory(directory_name: str, keyword: str = ".tif") -> list:
    """
    Given a folder directory, get a list of files with certain keywords.
    For the usecase, we search for ".tif" files.

    Parameters:
    directory_name: name of the folder directory.
    keyword: keyword the files inside the directory must contain.

    Returns:
    A list of files whose names contain the specified keyword, under the specified directory.
    """
    return [f"{directory_name}/{f}" for f in os.listdir(directory_name) if f.endswith(keyword)]


def move_file(source_path: str, label: str, id: int) -> None:
    """
    Move one file from source to destination. The destination folder is separated
    based on label, date, and id. The moved file will retain its original name.

    Parameters:
    source_path: path to the file to be moved.
    label: either "image" or "mask".
    id: suffix to the destination folder name.
    """
    file_name = source_path.split("/")[-1]
    file_date = source_path.split("/")[-4]
    destination_folder = os.path.join(os.path.dirname(
        __file__), f"../prepare_dataset/{label}_directory_{file_date}_{id}")
    destination_path = os.path.join(destination_folder, file_name)

    if not os.path.isdir(destination_folder):
        os.makedirs(destination_folder)

    if os.path.isfile(destination_path):
        print("File exists.")
        return

    shutil.move(source_path, destination_path)
    print(f"File copied to destination: {destination_path}.")


def batch_move_files(source_path_list: list) -> None:
    """
    Move files from a list of source paths. In this use case, images under `tiles` folder from `download_file`
    are moved to `prepare_dataset` folder. Images that are originally inside the same directory are grouped
    using the same id. Band geotif's are put under `image_directory{id}` while mask tif's are put under
    `mask_directory{id}`.

    Parameters:
    source_path_list: list of source paths.
    """
    path_dict = {}

    for i in range(len(source_path_list)):
        current_path = source_path_list[i]
        current_folder = current_path.rsplit(
            "/", 1)[0]  # split on the last occurrence

        if current_folder not in path_dict:
            path_dict[current_folder] = len(path_dict)

        current_id = path_dict[current_folder]

        if "mask" in current_path:
            move_file(current_path, "mask", current_id)
        else:
            move_file(current_path, "image", current_id)


def update_dataset_splits_csv(prepare_dataset_path: str) -> None:
    """
    Update the dataset_splits.csv file based on the image and mask directories
    present in the prepare_dataset folder.

    Parameters:
    prepare_dataset_path: path to the prepare_dataset folder.
    """
    image_dirs = sorted([d for d in os.listdir(
        prepare_dataset_path) if "image_directory" in d])
    mask_dirs = sorted([d for d in os.listdir(
        prepare_dataset_path) if "mask_directory" in d])

    # Assuming the number of image directories and mask directories are the same
    assert len(image_dirs) == len(
        mask_dirs), "Mismatch between image and mask directories."

    # Define the splits (you can adjust this as needed)
    num_train = int(0.7 * len(image_dirs))
    num_val = int(0.15 * len(image_dirs))
    # Remaining will be test

    with open(os.path.join(prepare_dataset_path, "dataset_splits.csv"), "w", newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["split", "data_path", "mask_path"])

        for i, (img_dir, mask_dir) in enumerate(zip(image_dirs, mask_dirs)):
            if i < num_train:
                split = "train"
            elif i < num_train + num_val:
                split = "val"
            else:
                split = "test"
            csvwriter.writerow([split, img_dir, mask_dir])


def main():
    # Get all folders from download_file folder
    source_path = os.path.join(os.path.dirname(__file__), "../download_file/")
    subfolders_list = fast_scandir(source_path)

    # Get folders that are under tiles folder
    # note that we need the / to get folders
    subfolders_with_keyword_list = get_folders_with_keyword(
        "tiles/", subfolders_list)

    all_files = []

    for subfolder in subfolders_with_keyword_list:
        current_list = get_list_of_files_in_directory(subfolder)
        all_files.extend(current_list)

    batch_move_files(all_files)


if __name__ == "__main__":
    main()
    prepare_dataset_path = os.path.join(
        os.path.dirname(__file__), "../prepare_dataset/")
    update_dataset_splits_csv(prepare_dataset_path)

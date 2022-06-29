import multiprocessing
from pathlib import Path
from typing import Tuple

from PIL import ImageDraw, ImageChops, ImageFilter, ImageOps, Image
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import definitions
from src.data.utils import load_geometry_matrix, create_grid_transformation_matrices


def generate_dataset(n_images: int, shot_no: int, subset: str, output_grid_shape: Tuple[int, int] = (256, 256)):
    data_path = (
                definitions.DATA_DIR / "processed" / "MANTIS" / str(shot_no)
                ) / subset

    if not data_path.exists():
        Path.mkdir(data_path, parents=True)

    start_id = 0

    if any(data_path.iterdir()):
        print("The data directory is not empty, will append to existing data")
        start_id = 1 + max(
            int("".join([s for s in f.name if s.isdigit()]))
            for f in data_path.glob("*")
        )

    images_i_list = tqdm(range(start_id, n_images + start_id))
    geometry_matrix = load_geometry_matrix(shot_no)

    tri_to_square_grid_mat, square_to_tri_grid_mat = create_grid_transformation_matrices(output_grid_shape)

    # maximum of 16 cores because of a small bottleneck on the marconi cluster
    num_cores = min(multiprocessing.cpu_count(), 16)
    Parallel(n_jobs=num_cores)(
        delayed(create_and_save_data_pair)(
            i,
            data_path,
            geometry_matrix,
            output_grid_shape,
            tri_to_square_grid_mat,
            square_to_tri_grid_mat,
        )
        for i in images_i_list
    )


def create_and_save_data_pair(
    i,
    data_path,
    geometry_matrix,
    output_grid_shape,
    tri_to_square_grid_mat,
    square_to_tri_grid_mat,
):
    Path.mkdir(data_path / str(i))

    random_lines = generate_random_lines()
    poloidal_crosssection = square_to_tri_grid_mat @ random_lines.reshape(-1)
    forward_modelled = geometry_matrix @ poloidal_crosssection

    # because in an image we only have 8 bits of information, we would like to use it all
    scaling_fw = 255 / max(forward_modelled.max(), 1e-2)

    fw_modelled_scaled = (forward_modelled * scaling_fw).astype(np.uint8)
    fw_modelled_reshaped = fw_modelled_scaled.reshape((1032, -1))

    fw_image = Image.fromarray(fw_modelled_reshaped)
    fw_image.save(data_path / str(i) / "cam_img.png")

    with open(data_path / str(i) / "scaling_cam.txt", "w") as f:
        f.write(f"{scaling_fw}")

    scaling_inv = 255 / max(poloidal_crosssection.max(), 1e-2)
    inversion_square_grid = tri_to_square_grid_mat @ poloidal_crosssection
    inversion_square_grid_scaled = (inversion_square_grid * scaling_inv).astype(
        np.uint8
    )
    inversion_image = Image.fromarray(
        inversion_square_grid_scaled.reshape(output_grid_shape)
    )
    inversion_image.save(data_path / str(i) / "inversion.png")

    with open(data_path / str(i) / "scaling_inv.txt", "w") as f:
        f.write(f"{scaling_inv}")


def generate_all_datasets(shot_no: int, n_images_train: int):
    generate_dataset(n_images_train, shot_no, "train")
    generate_dataset(int(0.2 * n_images_train), shot_no, "test")
    generate_dataset(int(0.2 * n_images_train), shot_no, "validation")


def generate_random_lines(n_lines: int = 20, output_shape: Tuple[int, int] = (256, 256), crop_padding: int = 80,
                          line_type_probs: Tuple[float, float, float] = (1/3, 1/3, 1/3)):
    (img_height, img_width) = output_shape
    img_shape_extended = (img_height + crop_padding, img_width + crop_padding)
    img = Image.new("L", (img_width + crop_padding, img_height + crop_padding))

    for i in range(n_lines):
        line_img = Image.new("L", (img_width + crop_padding, img_height + crop_padding))
        draw = ImageDraw.Draw(line_img)
        intensity = np.random.randint(0, 255)
        width = np.random.randint(1, 10)
        filter_radius = np.random.randint(1, 4)

        y1, x1 = np.random.randint(0, img_shape_extended, (2,))
        y2, x2 = np.random.randint(0, img_shape_extended, (2,))

        line_type = np.random.choice(["uniform", "curved", "gradient"], p=line_type_probs)
        if line_type == "gradient":
            draw.line((x1, y1, x2, y2), fill=intensity, width=width)
            with np.errstate(divide="ignore", invalid="ignore"):
                direction = (np.arctan2((x2 - x1), (y2 - y1))) * 180 / np.pi + 90
            gradient_img = create_gradient_img(
                img_shape_extended,
                np.random.randint(255),
                np.random.randint(200, 255),
                direction,
            )
            line_img = ImageChops.multiply(line_img, gradient_img)
        if line_type == "curved":
            arc_start = np.random.randint(0, 360)
            arc_length = np.random.randint(70, 160)
            draw.arc(
                (x1, y1, x2, y2),
                arc_start,
                (arc_start + arc_length),
                fill=intensity,
                width=width,
            )
        if line_type == "uniform":
            point0, point1 = create_random_points_uniform(output_shape)

            width = np.random.randint(1, 10)
            intensity = np.random.randint(0, 255)
            draw.line((point0, point1), fill=intensity, width=width)

        line_img = line_img.filter(ImageFilter.GaussianBlur(filter_radius))
        img = ImageChops.add(img, line_img)
    img = ImageOps.crop(img, border=crop_padding // 2)

    filter_radius = np.random.randint(1, 7)
    img = img.filter(ImageFilter.GaussianBlur(filter_radius))

    return np.array(img)


def create_gradient_img(
    img_shape: Tuple[int, int],
    start_intensity: int,
    end_intensity: int,
    direction: float,
):
    img = Image.new("L", img_shape)
    draw = ImageDraw.Draw(img)
    n_lines = 80
    step_size = img_shape[0] / n_lines
    start_point = np.random.randint(n_lines)

    min_length = min(30, n_lines - start_point - 1)
    length = max(1, np.random.randint(min_length, n_lines - start_point))
    for i in range(n_lines):
        if i < start_point:
            intensity = start_intensity
        elif i > start_point + length:
            intensity = end_intensity
        else:
            intensity = (
                start_intensity
                + (end_intensity - start_intensity) * (i - start_point) / length
            )
        draw.line(
            (i * step_size, 0, i * step_size, img_shape[1]),
            fill=int(intensity),
            width=10,
        )
    img = img.rotate(direction)
    return img


def create_random_points_uniform(output_shape: Tuple[int, int]):
    img_height, img_width = output_shape
    max_shape = max(img_height, img_width)

    x1, x2, x3 = np.random.uniform(0, 1, size=3)
    x3 = np.random.randint(-max_shape, max_shape)
    m = (x1 - 0.5) / (x2 - 0.5)
    b = (1 - m) * x3 if (m < 0) else (-1 - m) * x3 + 1

    point0 = (0, b)
    point1 = (img_width, m * img_width + b)

    return point0, point1

import numpy as np
from scipy.sparse import csc_matrix

import definitions


def load_geometry_matrix(geometry_id: str):
    filename = f"{geometry_id}.npz"
    geometry_matrix_file = (definitions.DATA_DIR / "utils" / "geometry_matrices" / filename)

    if not geometry_matrix_file.exists():
        raise IOError(f"Geometry matrix file not found for {filename=}")

    gm_data = np.load(geometry_matrix_file)
    geometry_matrix = csc_matrix((gm_data['mat_data'],
                                  (gm_data['mat_row_inds'], gm_data['mat_col_inds'])),
                                 shape=gm_data['mat_shape'])

    geometry_matrix = geometry_matrix.astype(np.float32)
    return geometry_matrix

import os
import numpy as np
import torch
import plyfile

import marching_cubes_cpp


def save_mesh(verts, colors, indices, output_file):
    ext = os.path.splitext(output_file)[1]
    if ext == '.obj':
        with open(output_file, 'w') as f:
            for k in range(len(verts)):
                v = verts[k]
                c = colors[k]
                f.write('v %f %f %f %d %d %d\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
            f.write('g foo\n')
            for ind in indices:
                f.write('f %d %d %d\n' % (ind[0] + 1, ind[1] + 1, ind[2] + 1))
            f.write('g\n')
    else:
        verts = np.array([tuple(v) for v in verts], dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
        face = np.array([(f.tolist(),220,220,220) for f in indices], dtype=[('vertex_indices', 'i4', (3,)), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        el_vert = plyfile.PlyElement.describe(verts,'vertex')
        el_face = plyfile.PlyElement.describe(face,'face')
        plyfile.PlyData([el_vert, el_face]).write(output_file)

def marching_cubes(tsdf, colors, isovalue, truncation, thresh, output_filename):
    if colors is None:
        colors = torch.ones(tsdf.shape[0], tsdf.shape[1], tsdf.shape[2], 3, dtype=torch.uint8) * 220
    if os.path.splitext(output_filename)[1] == '.ply':
        marching_cubes_cpp.export_marching_cubes(tsdf, colors, isovalue, truncation, thresh, output_filename)
    else: # can't figure out how to save vert colors with plyfile...
        vertices, vertcolors, faces = marching_cubes_cpp.run_marching_cubes(tsdf, colors, isovalue, truncation, thresh)
        save_mesh(vertices, vertcolors, faces, output_filename)

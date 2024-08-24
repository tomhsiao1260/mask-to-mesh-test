import nrrd
import tifffile
import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay
from skimage.morphology import skeletonize

center = 'center.json'
maskPath = './01744_02256_02768/01744_02256_02768_mask.nrrd'
volumePath = './01744_02256_02768/01744_02256_02768_volume.nrrd'

mask, header = nrrd.read(maskPath)
volume, header = nrrd.read(volumePath)

arr = np.zeros((256, 256, 256))

mask = np.where(mask == 2, 255, 0).astype(np.uint8)

tifffile.imwrite('output-0.tif', mask)

maskSkeleton = np.zeros_like(mask)

z, y, x = np.indices((256, 256, 256))
maskSkeleton[(x*x+y*y+z*z > 190*190) & (x*x+y*y+z*z < 200*200)] = True

for i in range(mask.shape[0]):
    maskSkeleton[i, :, :] = skeletonize(maskSkeleton[i, :, :])
    # maskSkeleton[i, :, :] = skeletonize(mask[i, :, :])

points = np.argwhere(maskSkeleton)
points = points[np.random.choice(len(points), int(len(points) * 0.1), replace=False)]

zmin, zmax = np.min(points[:, 0]), np.max(points[:, 0])
ymin, ymax = np.min(points[:, 1]), np.max(points[:, 1])

v = (points[:, 0] - zmin) / (zmax - zmin)
u = (points[:, 1] - ymin) / (ymax - ymin)

uvs = np.column_stack((u, v))

colors = np.zeros((len(uvs), 3)).astype(np.float16)

colors[:, 0] = uvs[:, 0]
colors[:, 1] = uvs[:, 1]
colors[:, 2] = 1

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
point_cloud.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud("points.ply", point_cloud)

tri = Delaunay(uvs)
faces = tri.simplices

mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(points)
mesh.triangles = o3d.utility.Vector3iVector(faces)
o3d.io.write_triangle_mesh('output.obj', mesh)





import os
import json
import numpy as np
import open3d as o3d
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import shutil

# Diretórios
base_dir = Path("~/Fernando-Braga/Automatic Calibration/Dataset/ds0").expanduser()
output_dir = base_dir.parent / "ds0_aug"
pcd_dir = base_dir / "pointcloud"
ann_dir = base_dir / "ann"
output_pcd_dir = output_dir / "pointcloud"
output_ann_dir = output_dir / "ann"

# Criar diretórios de saída
output_pcd_dir.mkdir(parents=True, exist_ok=True)
output_ann_dir.mkdir(parents=True, exist_ok=True)

# Funções auxiliares
def load_point_cloud(filepath):
    return o3d.io.read_point_cloud(str(filepath))

def save_point_cloud(pcd, filepath):
    o3d.io.write_point_cloud(str(filepath), pcd)

def load_annotation(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def save_annotation(annotation, filepath):
    with open(filepath, 'w') as f:
        json.dump(annotation, f, indent=2)

def apply_transform(pcd_np, transform):
    ones = np.ones((pcd_np.shape[0], 1))
    homo_coords = np.hstack((pcd_np, ones))
    transformed = (transform @ homo_coords.T).T[:, :3]
    return transformed

def get_transform_matrix(rotation_deg=(0, 0, 0), translation=(0, 0, 0), scale=1.0):
    rot = R.from_euler('xyz', rotation_deg, degrees=True).as_matrix()
    rot *= scale
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = translation
    return transform

def augment_sample(file_id, transform_id, rotation=(0, 0, 0), translation=(0, 0, 0), scale=1.0):
    # Arquivos
    pcd_file = pcd_dir / f"{file_id}.pcd"
    ann_file = ann_dir / f"{file_id}.pcd.json"
    out_pcd_file = output_pcd_dir / f"{file_id}_{transform_id}.pcd"
    out_ann_file = output_ann_dir / f"{file_id}_{transform_id}.pcd.json"

    # Carrega ponto e anotação
    pcd = load_point_cloud(pcd_file)
    ann = load_annotation(ann_file)

    # Aplica transformação na nuvem
    pts = np.asarray(pcd.points)
    transform = get_transform_matrix(rotation, translation, scale)
    pts_transformed = apply_transform(pts, transform)
    pcd.points = o3d.utility.Vector3dVector(pts_transformed)
    save_point_cloud(pcd, out_pcd_file)

    # Aplica transformação na anotação (posição + rotação)
    for fig in ann.get("figures", []):
        pos = fig["geometry"]["position"]
        rot = fig["geometry"]["rotation"]

        # Aplica rotação e translação ao centro
        center = np.array([pos['x'], pos['y'], pos['z']])
        center_h = np.append(center, 1.0)
        new_center = transform @ center_h
        fig["geometry"]["position"] = {
            "x": float(new_center[0]),
            "y": float(new_center[1]),
            "z": float(new_center[2]),
        }

        # Atualiza rotação z (considerando só yaw para simplicidade)
        fig["geometry"]["rotation"]["z"] += np.deg2rad(rotation[1])

        # Aplica escala à dimensão da bbox
        for dim_key in ['x', 'y', 'z']:
            fig["geometry"]["dimensions"][dim_key] *= scale

    save_annotation(ann, out_ann_file)
    print(f"[OK] {file_id}_{transform_id} salvo.")

# Coletar IDs dos arquivos
file_ids = [f.stem for f in pcd_dir.glob("*.pcd")]

if not file_ids:
    print("[ERRO] Nenhum arquivo .pcd encontrado em:", pcd_dir)
else:
    for file_id in file_ids:
        augment_sample(file_id, "rot1", rotation=(0, 20, 0))
        augment_sample(file_id, "scale1", scale=1.05)
        augment_sample(file_id, "trans1", translation=(0.05, 0, 0))
        augment_sample(file_id, "rot2", rotation=(0, -20, 0))
        augment_sample(file_id, "scale2", scale=0.95)
        augment_sample(file_id, "trans2", translation=(-0.05, 0.02, 0.01))

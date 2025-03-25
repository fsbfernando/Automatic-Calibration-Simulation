import open3d as o3d
import numpy as np
import glob
import os
from scipy.spatial.transform import Rotation as R
from scipy.linalg import svd

# =============================================================================
# Funções auxiliares
# =============================================================================
def load_and_downsample(file_path, voxel_size):
    """
    Carrega a nuvem de pontos do arquivo e aplica voxel downsampling.
    """
    pcd = o3d.io.read_point_cloud(file_path)
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    return pcd_down

def align_with_icp(source, target, distance_threshold):
    """
    Alinha a nuvem 'source' à 'target' usando ICP com um threshold definido.
    Retorna a matriz de transformação (4x4).
    """
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )
    return reg_p2p.transformation

def simulate_robot_transforms(n):
    """
    Para exemplificação, simula uma sequência de n transformações do robô.
    Aqui, cada pose tem uma rotação incremental de 1° em torno do eixo Z e 
    uma pequena translação no eixo X. Em prática, essas transformações 
    devem vir dos dados reais do robô.
    """
    robot_transforms = []
    for i in range(n):
        angle = np.deg2rad(1.0 * i)  # incremento de 1° por medição
        t = np.array([0.005 * i, 0, 0])  # translação incremental
        R_mat = R.from_euler('z', angle).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = t
        robot_transforms.append(T)
    return robot_transforms

def compute_relative_transforms(transforms):
    """
    Calcula as transformações relativas entre medições consecutivas.
    Ou seja, para cada i, retorna: T_rel = inv(T_i) * T_{i+1}.
    """
    rel_transforms = []
    for i in range(len(transforms)-1):
        rel_transforms.append(np.linalg.inv(transforms[i]) @ transforms[i+1])
    return rel_transforms

def solve_hand_eye(A_list, B_list):
    """
    Resolve o problema hand-eye: A_i * X = X * B_i para i=1...n,
    onde cada A_i e B_i são transformações 4x4.
    
    A abordagem é feita em duas etapas:
      1. Resolver para a parte de rotação:
         - Para cada par, temos: R_Ai * R_X = R_X * R_Bi.
         - Vetorizamos essa equação e empilhamos as equações lineares para resolver
           um sistema homogêneo do tipo M vec(R_X) = 0.
         - A solução é projetada em SO(3) via SVD.
      2. Resolver para a translação:
         - A equação é: (R_Ai - I) t_X = R_X * t_Bi - t_Ai, empilhada para todos os i,
           resolvendo em least-squares para t_X.
    
    Retorna a matriz X (4x4) que representa a transformação entre o sensor e o robô.
    """
    n_pairs = len(A_list)
    M = []
    for i in range(n_pairs):
        R_A = A_list[i][:3, :3]
        R_B = B_list[i][:3, :3]
        # Constrói a equação: vec(R_A * R_X - R_X * R_B) = 0
        M_i = np.kron(np.eye(3), R_A) - np.kron(R_B.T, np.eye(3))
        M.append(M_i)
    M = np.vstack(M)
    
    # Resolver M vec(R_X) = 0 usando SVD
    U, s, Vt = np.linalg.svd(M)
    vec_RX = Vt[-1, :]
    R_X_est = vec_RX.reshape(3, 3)
    
    # Projeta R_X_est para SO(3)
    U_r, _, Vt_r = np.linalg.svd(R_X_est)
    R_X = U_r @ Vt_r
    if np.linalg.det(R_X) < 0:
        R_X = -R_X

    # Resolver a parte de translação:
    # Para cada par: (R_A - I)*t_X = R_X*t_B - t_A
    M_t = []
    b_t = []
    for i in range(n_pairs):
        R_A = A_list[i][:3, :3]
        t_A = A_list[i][:3, 3]
        t_B = B_list[i][:3, 3]
        M_t.append(R_A - np.eye(3))
        b_t.append(R_X @ t_B - t_A)
    M_t = np.vstack(M_t)
    b_t = np.hstack(b_t)
    
    # Resolver em least-squares para t_X
    t_X, _, _, _ = np.linalg.lstsq(M_t, b_t, rcond=None)
    
    # Monta a matriz X
    X = np.eye(4)
    X[:3, :3] = R_X
    X[:3, 3] = t_X
    return X

# =============================================================================
# Pipeline principal
# =============================================================================
def main():
    # Definição dos diretórios
    base_path = r"C:\Automatic Calibration\Base do robo\robo-base-final-ply.ply"
    run_path = r"C:\Automatic Calibration\Run_1"
    voxel_size = 0.005
    distance_threshold = 0.02
    
    # Carregar o modelo da base ideal
    print("Carregando modelo base ideal...")
    base = load_and_downsample(base_path, voxel_size)
    
    # Carregar e processar cada nuvem de pontos do diretório Run_1
    ply_files = glob.glob(os.path.join(run_path, "*.ply"))
    ply_files.sort()  # para garantir ordem
    sensor_transforms = []
    sensor_clouds = []
    
    print("Processando nuvens de pontos do sensor (Run_1)...")
    for file in ply_files:
        print("Processando:", file)
        cloud = load_and_downsample(file, voxel_size)
        sensor_clouds.append(cloud)
        # Alinha a nuvem ao modelo base usando ICP
        T_sensor = align_with_icp(cloud, base, distance_threshold)
        sensor_transforms.append(T_sensor)
    
    n_measurements = len(sensor_transforms)
    print(f"Foram processadas {n_measurements} medições de sensor.")
    
    # Simular as transformações do robô (estas devem vir dos dados reais do robô)
    robot_transforms = simulate_robot_transforms(n_measurements)
    
    # Calcular as transformações relativas para o robô (A_i) e para o sensor (B_i)
    A_list = compute_relative_transforms(robot_transforms)
    B_list = compute_relative_transforms(sensor_transforms)
    
    # Resolver o problema hand-eye: A_i * X = X * B_i
    X = solve_hand_eye(A_list, B_list)
    
    print("\nMatriz de calibração hand-eye (X):")
    print(X)
    
    # Salvar a matriz em um arquivo de texto
    output_file = r"C:\Users\fsbfe\Desktop\hand_eye_calibration_matrix.txt"
    np.savetxt(output_file, X, fmt="%.6f")
    print(f"\nMatriz de calibração salva em: {output_file}")

if __name__ == "__main__":
    main()

import time
import numpy as np
import open3d as o3d
import os
import sys

import sim

# ==============================
# CONFIGURAÇÃO DE POSIÇÕES DAS JUNTAS
# ==============================
joint_positions = [
    [0, 65, 0, -90, 0, 120, 0],
    [0, 40, 0, -70, 0, 120, 0],
    [0, 20, 0, -100, 0, 120, 0],
    [0, 50, 0, -80, 0, 120, 0]
]

# ==============================
# CONFIGURAÇÃO DE CAMINHO PARA SALVAR AS NUVENS
# ==============================
BASE_DIR = os.path.join('C:\\', 'Automatic Calibration')
os.makedirs(BASE_DIR, exist_ok=True)

# Conta o número de runs já existentes para criar a próxima
run_count = len([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]) + 1
RUN_DIR = os.path.join(BASE_DIR, f'Run_{run_count}')
os.makedirs(RUN_DIR, exist_ok=True)

print(f"📂 Salvando nuvens de pontos em: {RUN_DIR}")

# ==============================
# CONEXÃO COM O COPPELIASIM
# ==============================
print("Conectando ao CoppeliaSim...")
sim.simxFinish(-1)  # Fecha qualquer conexão anterior
clientID = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)

if clientID != -1:
    print("✅ Conexão bem-sucedida com a Remote API!")
else:
    print("❌ Falha na conexão. Verifique se a simulação está rodando.")
    sys.exit()

# ==============================
# PEGANDO HANDLES DAS JUNTAS E DA CÂMERA
# ==============================
joint_handles = []
for i in range(1, 8):
    res, joint_handle = sim.simxGetObjectHandle(clientID, f'LBR_iiwa_14_R820_joint{i}', sim.simx_opmode_blocking)
    if res == sim.simx_return_ok:
        joint_handles.append(joint_handle)
    else:
        print(f"❌ Falha ao obter handle da LBR_iiwa_14_R820_joint{i}")

# Pega o handle do sensor de visão
res, vision_sensor_handle = sim.simxGetObjectHandle(clientID, 'Realsense', sim.simx_opmode_blocking)
if res != sim.simx_return_ok:
    print("❌ Falha ao obter o handle do sensor de visão.")
    sys.exit()

# Inicia o stream da câmera
sim.simxGetVisionSensorImage(clientID, vision_sensor_handle, 0, sim.simx_opmode_streaming)
time.sleep(1)  # Dá tempo para iniciar o stream

# ==============================
# MOVIMENTAÇÃO E EXTRAÇÃO DA NUVEM DE PONTOS
# ==============================
for i, position in enumerate(joint_positions):
    print(f"🔄 Movendo para posição {i + 1}: {position}")

    # Move cada junta para o valor especificado
    for j in range(7):
        sim.simxSetJointTargetPosition(clientID, joint_handles[j], np.deg2rad(position[j]), sim.simx_opmode_blocking)
    time.sleep(2)  # Dá tempo para o robô se mover completamente

    # Captura a imagem do sensor de profundidade
    res, resolution, image = sim.simxGetVisionSensorImage(clientID, vision_sensor_handle, 0, sim.simx_opmode_buffer)
    
    if res == sim.simx_return_ok:
        print(f"📸 Resolução da câmera: {resolution}")

        # ==========================
        # TRATAMENTO DE IMAGEM RGB E PROFUNDIDADE
        # ==========================
        if len(image) == resolution[0] * resolution[1] * 3:  # RGB (256 x 256 x 3)
            depth_data = np.array(image[0::3], dtype=np.float32).reshape((resolution[1], resolution[0])) * 0.001
        else:
            depth_data = np.array(image, dtype=np.float32).reshape((resolution[1], resolution[0])) * 0.001

        # ==========================
        # CONVERSÃO PARA NUVEM DE PONTOS
        # ==========================
        fx = resolution[0] / (2 * np.tan(np.deg2rad(60) / 2))
        fy = fx
        cx = resolution[0] / 2
        cy = resolution[1] / 2

        points = []
        for v in range(resolution[1]):
            for u in range(resolution[0]):
                depth = depth_data[v][u]
                if depth > 0:
                    x = (u - cx) * depth / fx
                    y = (v - cy) * depth / fy
                    z = depth
                    points.append([x, y, z])

        if points:
            # Cria a nuvem de pontos usando Open3D
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # Nome do arquivo com caminho completo
            filename = os.path.join(RUN_DIR, f'PointCloud_{i + 1}.ply')

            # Força o Open3D a salvar sem rply
            success = o3d.io.write_point_cloud(filename, pcd, write_ascii=True)
            if success:
                print(f"✅ Nuvem de pontos salva em: {filename}")
            else:
                print(f"❌ Falha ao salvar a nuvem de pontos em: {filename}")
        else:
            print("❌ Falha ao criar nuvem de pontos: Nenhum dado foi capturado.")

# ==============================
# FINALIZA A CONEXÃO
# ==============================
sim.simxFinish(clientID)
print("✅ Conexão encerrada com sucesso!")

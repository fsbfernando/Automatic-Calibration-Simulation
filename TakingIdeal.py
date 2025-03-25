import time
import os
import numpy as np
import open3d as o3d
import sim

# ==============================
# CONFIGURAÇÃO DE CAMINHO PARA SALVAR AS NUVENS
# ==============================
BASE_DIR = os.path.join(os.path.expanduser('~'), 'Desktop', 'Ref_PointClouds')
os.makedirs(BASE_DIR, exist_ok=True)

# Cria uma subpasta para cada execução (Run 1, Run 2, ...)
run_id = 1
while os.path.exists(os.path.join(BASE_DIR, f'Run {run_id}')):
    run_id += 1
run_dir = os.path.join(BASE_DIR, f'Run {run_id}')
os.makedirs(run_dir, exist_ok=True)

print(f"📂 Salvando nuvens de pontos em: {run_dir}")

# ==============================
# CONECTAR AO COPPELIASIM
# ==============================
print("Conectando ao CoppeliaSim...")
sim.simxFinish(-1)  # Fecha conexões antigas
clientID = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)

if clientID != -1:
    print("✅ Conexão bem-sucedida com a Remote API!")
else:
    print("❌ Falha na conexão. Verifique se a simulação está rodando.")
    exit()

# ==============================
# HANDLE DO SENSOR DE VISÃO
# ==============================
vision_sensor_handle = sim.simxGetObjectHandle(clientID, 'Vision_sensor', sim.simx_opmode_blocking)[1]

# ==============================
# CAPTURA DE NUVEM DE PONTOS (DURANTE A ROTAÇÃO)
# ==============================
samples_per_rotation = 65
angle_step = 360 / samples_per_rotation
current_sample = 0

start_time = time.time()

for i in range(3 * samples_per_rotation):  # 3 voltas
    print(f"🔄 Capturando posição {current_sample + 1} de {3 * samples_per_rotation}...")

    # Captura a imagem de profundidade
    result, resolution, image = sim.simxGetVisionSensorDepthBuffer(clientID, vision_sensor_handle, sim.simx_opmode_blocking)
    
    if result == sim.simx_return_ok:
        print(f"📸 Resolução da câmera: {resolution}")
        
        # Converte os dados para array numpy
        depth_data = np.array(image, dtype=np.float32).reshape((resolution[1], resolution[0])) * 0.001

        fx = resolution[0] / (2 * np.tan(np.radians(86) / 2))
        fy = resolution[1] / (2 * np.tan(np.radians(86) / 2))
        cx = resolution[0] / 2
        cy = resolution[1] / 2

        points = []
        for v in range(resolution[1]):
            for u in range(resolution[0]):
                z = depth_data[v][u]
                if z > 0 and z < 10:
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    points.append([x, y, z])

        # ==============================
        # OBTER A POSIÇÃO E ORIENTAÇÃO DO SENSOR NO SISTEMA GLOBAL
        # ==============================
        result, sensor_position = sim.simxGetObjectPosition(clientID, vision_sensor_handle, -1, sim.simx_opmode_blocking)
        result, sensor_orientation = sim.simxGetObjectOrientation(clientID, vision_sensor_handle, -1, sim.simx_opmode_blocking)

        if result != sim.simx_return_ok:
            print("❌ Erro ao obter a posição ou orientação global do sensor.")
            exit()

        print(f"📍 Posição global do sensor: {sensor_position}")
        print(f"🔄 Orientação global do sensor: {sensor_orientation}")

        # ==============================
        # APLICAR A TRANSFORMAÇÃO PARA COORDENADAS GLOBAIS
        # ==============================
        # Criar a matriz de rotação baseada na orientação do sensor
        angle_rad = sensor_orientation[2]  # Usando a rotação em torno do eixo Z (azimutal)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])

        # Aplique a rotação e a translação
        transformed_points = []
        for point in points:
            rotated_point = np.dot(rotation_matrix, point)  # Rotaciona o ponto
            global_point = rotated_point + np.array(sensor_position)  # Translação pela posição do sensor

            # Se necessário, ajuste a escala (caso as distâncias ainda estejam muito grandes)
            scale_factor = 0.1  # Ajuste esse fator de escala conforme necessário
            global_point *= scale_factor

            transformed_points.append(global_point)

        # ==============================
        # CRIAR E SALVAR A NUVEM DE PONTOS TRANSFORMADA
        # ==============================
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.array(transformed_points))

        # Salva o arquivo .PLY para cada amostra
        file_path = os.path.join(run_dir, f'PointCloud_{current_sample + 1}.ply')
        o3d.io.write_point_cloud(file_path, point_cloud)
        print(f"✅ Nuvem de pontos salva em: {file_path}")

        current_sample += 1

    else:
        print("❌ Erro ao capturar a imagem de profundidade")

    # Aguarda o próximo passo para capturar (permite rotação contínua)
    time.sleep(0.2)

# ==============================
# FINALIZA CONEXÃO
# ==============================
sim.simxFinish(clientID)
print("✅ Conexão encerrada com sucesso!")
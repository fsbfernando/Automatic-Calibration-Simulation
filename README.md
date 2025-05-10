## CoppeliaSim:

CenaBaseIdeal.ttt - Robô avulso com a câmera orbitando ao redor da base (Coppelia) - Sensor configurado de maneira ideal

CenaCalibracao.ttt - Cena do robô sobre a base fixa, com a câmera fixa (Coppelia) - Sensor configurado de modo a simular uma Realsense D435

## Scripts Desenvolvidos em Python em uso:

CloudPointExtractor.py - Responsável por mover o robô para 4 posições pré definidas (que posicionam o flange em direção a base) e registrar a "visão de Realsense" ao estar apontada para base.

TakingIdeal.py - Responsável por capturar as diversas nuvens de pontos que servem de matéria prima para a reconstrução da base a partir de diversas nuvens de pontos da base.

ScriptComando.py - Responsável por movimentar o robô da simulação para quantas posições forem desejadas. Utilizado para testar quais posições de juntas apontam o flange para a base.

3DDataAugmentation.py - Responsável por aumentar a quantidade de dados rotulados da base do robô, de modo a formar um dataset adequado para o treino da rede neural PV RCNN.

PVRCNN-Training.py - Responsável pelo treinamento da rede neural PV RCNN a partir dos dados 3d previamente rotulados.

## Scripts Desenvolvidos em Python que não estão em uso no momento:

MatrixCalculator.py - Primeira tentativa de extrair a matriz de transformação desejada a partir do alinhamentos das nuvens de pontos

ExtrinsicMatrix.py - Segunda tentativa de extrair a matriz de transformação desejada a partir do alinhamento das nuvens de pontos

IdealConstructor.py - Responsável por reconstruir o modelo 3D a partir das diversas nuvens de pontos obtidas pelo TakingIdeal.py

## Scripts Python para Integração Python + Coppelia:

sim.py

simConst.py

RemoteAPI.dll também é responsável pela integração do Python com o Coppelia

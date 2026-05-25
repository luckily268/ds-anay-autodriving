"""全局配置文件"""

import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据目录
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
IMAGES_DIR = os.path.join(RESULTS_DIR, "images")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")

# GTSRB 数据集
GTSRB_NUM_CLASSES = 43

# GTSRB 43类交通标志名称
GTSRB_LABELS = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)",
    "Speed limit (120km/h)", "No passing",
    "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection", "Priority road", "Yield",
    "Stop", "No vehicles", "Vehicles over 3.5 metric tons prohibited",
    "No entry", "General caution", "Dangerous curve to the left",
    "Dangerous curve to the right", "Double curve", "Bumpy road",
    "Slippery road", "Road narrows on the right", "Road work",
    "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead",
    "Turn left ahead", "Ahead only", "Go straight or right",
    "Go straight or left", "Keep right", "Keep left",
    "Roundabout mandatory", "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons",
]

# 重点关注的交通标志索引（用于攻击实验）
# 14=Stop, 2=限速50, 7=限速100, 13=Yield, 17=No entry
TARGET_INDICES = [14, 2, 7, 13, 17]

# 训练参数
BATCH_SIZE = 64
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
IMAGE_SIZE = 64

# 攻击参数
FGSM_EPSILONS = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]
PGD_EPSILON = 0.1
PGD_ALPHA = 0.01
PGD_STEPS_LIST = [5, 10, 20, 50]
CW_CONFIDENCE = 0
CW_LEARNING_RATE = 0.01
CW_ITERATIONS = 500
PATCH_SIZE = 50
PATCH_LEARNING_RATE = 0.1
PATCH_EPOCHS = 1000

# GPT API 配置
GPT_API_KEY = os.environ.get("GPT_API_KEY", "")
GPT_BASE_URL = os.environ.get("GPT_BASE_URL", "https://api.deepseek.com/v1")
GPT_MODEL = os.environ.get("GPT_MODEL", "deepseek-v4-flash")

# 确保目录存在
for d in [DATA_DIR, MODEL_DIR, RESULTS_DIR, IMAGES_DIR, FIGURES_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

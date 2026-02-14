"""
客戶端程式碼 - 使用 Flower 進行聯邦學習
"""
import flwr as fl
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
import os
import seaborn as sns
from collections import Counter, defaultdict
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import socket
import json
import time
import resource



csv_file_train= "NSLKDD_clientsplit50_No.1_train.csv"  
csv_file_test= "NSLKDD_clientsplit50_No.1_test.csv"
total_clients = 50  
total_rounds = 100  
sampling_ratio = 1  
client_id = "1"  
     

local_epochs = 5  
server_IP = "0.0.0.0:8080"  
TCP_IP = "0.0.0.0" 
TCP_Port = 8888  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置：{device}")

try:
    raw_data = pd.read_csv(csv_file_train)
    last_column_name = raw_data.columns[-1]
    value_counts = raw_data[last_column_name].value_counts()
    value_percentages = raw_data[last_column_name].value_counts(normalize=True) * 100
    results_df = pd.DataFrame({
        'Count': value_counts.astype(int),  
        'Percentage (%)': value_percentages.round(3).astype(str) + '%' 
    })
    total_count = value_counts.sum()
    results_df.loc['Total'] = [total_count, '100.0%']
    print(f"最後一列的標籤名稱為：{last_column_name}")
    print(f"{results_df}\n")
except Exception as e:
    print(f"讀取合併後的 CSV 文件時發生錯誤: {e}\n")

try:
    raw_data = pd.read_csv(csv_file_test)
    last_column_name = raw_data.columns[-1]
    value_counts = raw_data[last_column_name].value_counts()
    value_percentages = raw_data[last_column_name].value_counts(normalize=True) * 100
    results_df = pd.DataFrame({
        'Count': value_counts.astype(int),  
        'Percentage (%)': value_percentages.round(3).astype(str) + '%' 
    })
    total_count = value_counts.sum()
    results_df.loc['Total'] = [total_count, '100.0%']
    print(f"最後一列的標籤名稱為：{last_column_name}")
    print(f"{results_df}\n")
except Exception as e:
    print(f"讀取合併後的 CSV 文件時發生錯誤: {e}\n")    

train_df = pd.read_csv(csv_file_train)
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = pd.read_csv(csv_file_test)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
X_train = train_df.drop(columns=['label'])
y_train = train_df['label']
X_test = test_df.drop(columns=['label'])
y_test = test_df['label']
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

class LoadData(Dataset):
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = torch.tensor(self.X[index], dtype=torch.float32)
        y = torch.tensor(self.y[index], dtype=torch.long)
        return X, y
train_data = LoadData(X_train, y_train)
val_data   = LoadData(X_val, y_val)
test_data = LoadData(X_test, y_test)

X_dimension = len(X_train.columns)  
y_dimension = len(y_train.value_counts()) 
print(f"X的維度：{X_dimension}")  
print(f"y的維度：{y_dimension}")
print(f"訓練集樣本數：{len(X_train)}")
print(f"測試集樣本數：{len(X_test)}\n")

batch_size = 128
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader   = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)  

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_image_paperCNN = "/home/root/output_image/"
image_client_root = os.path.join(output_image_paperCNN, "image_client")
client_folder = os.path.join(
    image_client_root,
    f"NSLKDD",
    f"SAFE",
    f"total_clients_{total_clients}",
    f"rounds_{total_rounds}",
    f"sample_{int(sampling_ratio * 100)}%",
    f"clients_id_{client_id}"
)
os.makedirs(client_folder, exist_ok=True)
print(f"當前儲存資料夾為：{client_folder}")
image = os.path.join(client_folder, "client_training")
os.makedirs(image, exist_ok=True)
image_file = os.path.join(image, f"client_training_plots_{timestamp}.png")
roc_curve_file = os.path.join(image, f"client_roc_curve_plots_{timestamp}.png")
history_log_file = os.path.join(image, f"history_log_{timestamp}.csv")

train_confusion_matrix_image = os.path.join(client_folder, "client_train_confusion_matrix")
os.makedirs(train_confusion_matrix_image, exist_ok=True)
train_confusion_matrix = os.path.join(train_confusion_matrix_image, f"client_train_confusion_matrix_{timestamp}.png")
train_confusion_matrix_Percentage_path = os.path.join(train_confusion_matrix_image, f"client_train_confusion_matrix_Percentage_{timestamp}.png")

test_confusion_matrix_image = os.path.join(client_folder, "client_test_confusion_matrix")
os.makedirs(test_confusion_matrix_image, exist_ok=True)
test_confusion_matrix = os.path.join(test_confusion_matrix_image, f"client_test_confusion_matrix_{timestamp}.png")
test_confusion_matrix_Percentage_path = os.path.join(test_confusion_matrix_image, f"client_test_confusion_matrix_Percentage_{timestamp}.png")

train_matric_image = os.path.join(client_folder, "client__train_matrics")
os.makedirs(train_matric_image, exist_ok=True)
train_metrics = os.path.join(train_matric_image, f"client_train_matrics_{timestamp}.csv")

test_matric_image = os.path.join(client_folder, "client__test_matrics")
os.makedirs(test_matric_image, exist_ok=True)
test_metrics = os.path.join(test_matric_image, f"client_test_matrics_{timestamp}.csv")

confusion_matrix_csv = os.path.join(client_folder, "client_confusion_matrix_csv")
os.makedirs(confusion_matrix_csv, exist_ok=True)
train_confusion_matrix_csv_path = os.path.join(confusion_matrix_csv, f"train_confusion_matrix_{timestamp}.csv")
test_confusion_matrix_csv_path = os.path.join(confusion_matrix_csv, f"test_confusion_matrix_{timestamp}.csv")

class_labels = ["dos", "probe", "u2r", "r2l", "normal"]
label_to_index = {label: idx for idx, label in enumerate(class_labels)}
index_to_label = {idx: label for label, idx in label_to_index.items()}
fixed_label_indices = list(range(len(class_labels)))

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)  
        self.pool1 = nn.MaxPool1d(kernel_size=2) 

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1) 
        self.pool2 = nn.MaxPool1d(kernel_size=2) 

        self.flatten = nn.Flatten()  
        
        self.fc1 = nn.Linear(1280, 32)  
        self.dropout = nn.Dropout(p=0.5)  
        self.fc2 = nn.Linear(32, 5)  
        
    def extract_features(self, x):
        x = x.unsqueeze(1)
        x = nn.functional.relu(self.conv1(x))
        x = self.pool1(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))  
        x = self.dropout(x)
        return x    

    def forward(self, x):
        x = x.unsqueeze(1)
        x = nn.functional.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = nn.functional.relu(self.conv2(x))
        x = self.pool2(x)
      
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = self.fc2(x)  
        return x

model = CNNModel().to(device)

def agg_func(protos):  
    for label, vectors in protos.items():
        protos[label] = sum(vectors) / len(vectors)
    return protos

def TRM(x, y, kernel="rbf", device='cpu'):
    if x.numel() == 0 or y.numel() == 0:
        print("[警告] TRM: 有空特徵輸入，返回 0")
        return torch.tensor(0.0, device=device)
    x = x.to(device)
    y = y.to(device)
    
    xx = torch.matmul(x, x.t())
    yy = torch.matmul(y, y.t())
    zz = torch.matmul(x, y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2 * xx
    dyy = ry.t() + ry - 2 * yy
    dxy = rx.t() + ry - 2 * zz

    XX, YY, XY = torch.zeros_like(xx), torch.zeros_like(yy), torch.zeros_like(zz)

    if kernel == "multiscale":
        bandwidths = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidths:
            XX += a**2 / (a**2 + dxx)
            YY += a**2 / (a**2 + dyy)
            XY += a**2 / (a**2 + dxy)
    elif kernel == "rbf":
        bandwidths = [10, 15, 20, 50]
        for a in bandwidths:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)
    
    return torch.mean(XX + YY - 2 * XY)

def cm_from_preds_labels(preds_list, labels_list, num_classes=None):
    preds = torch.cat([p.reshape(-1).detach().cpu().long() for p in preds_list], dim=0)
    labels = torch.cat([y.reshape(-1).detach().cpu().long() for y in labels_list], dim=0)
    if num_classes is None:
        num_classes = int(max(preds.max(), labels.max()).item()) + 1
    idx = labels * num_classes + preds
    cm = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm.numpy(), num_classes

def macro_f1_from_cm(cm: np.ndarray):
    cm = np.asarray(cm, dtype=np.int64)
    TP = np.diag(cm).astype(np.float64)
    FP = cm.sum(0) - TP
    FN = cm.sum(1) - TP
    denom = 2*TP + FP + FN
    per_class_f1 = np.divide(2*TP, denom, out=np.zeros_like(denom, dtype=np.float64), where=denom>0)
    support = cm.sum(1)
    valid = support > 0
    return float(per_class_f1[valid].mean()) if valid.any() else 0.0

def train(encoder_model, classifier_head, train_dataloader, val_dataloader, epochs, device):
    encoder_model.to(device)
    classifier_head.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(
        list(encoder_model.parameters()) + list(classifier_head.parameters()), 
        lr=0.001
    )

    for epoch in range(epochs):
        encoder_model.train()
        classifier_head.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_preds, train_labels = [], []
        for X_batch, y_batch in train_dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            features = encoder_model.extract_features(X_batch)
            outputs = classifier_head(features)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)
            train_preds.append(predictions)
            train_labels.append(y_batch)
        avg_train_loss = running_loss / max(1, len(train_dataloader))
        train_accuracy = correct / max(1, total)
        cm_train, _ = cm_from_preds_labels(train_preds, train_labels)
        macro_f1_train = macro_f1_from_cm(cm_train)

        encoder_model.eval()
        classifier_head.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X_val, y_val in val_dataloader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                features = encoder_model.extract_features(X_val)
                val_outputs = classifier_head(features)
                vloss = criterion(val_outputs, y_val)
                val_running_loss += vloss.item()
                val_predictions = torch.argmax(val_outputs, dim=1)
                val_correct += (val_predictions == y_val).sum().item()
                val_total += y_val.size(0)
                val_preds.append(val_predictions)
                val_labels.append(y_val)
        avg_val_loss = val_running_loss / max(1, len(val_dataloader))
        val_accuracy = val_correct / max(1, val_total)
        cm_val, _ = cm_from_preds_labels(val_preds, val_labels)
        macro_f1_val = macro_f1_from_cm(cm_val)
                
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} Train Accuracy: {train_accuracy:.4f} Train macro f1-score: {macro_f1_train:.4f} | "
            f"Validation Loss: {avg_val_loss:.4f} Validation Accuracy: {val_accuracy:.4f} Validation macro f1-score: {macro_f1_val:.4f}"
        )
    return float(avg_train_loss), float(train_accuracy), float(avg_val_loss), float(val_accuracy), float(macro_f1_train)

def test(encoder_model, global_head, test_dataloader, device):
    encoder_model.to(device).eval()
    global_head.to(device).eval()
    criterion = torch.nn.CrossEntropyLoss().to(device)
    running_loss = 0.0
    correct = 0
    total = 0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            features = encoder_model.extract_features(X_batch)
            outputs = global_head(features)
            loss = criterion(outputs, y_batch)
            running_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)
            preds_all.append(predictions)
            labels_all.append(y_batch)
    avg_test_loss = running_loss / max(1, len(test_dataloader))
    test_accuracy = correct / max(1, total)
    cm_test, _ = cm_from_preds_labels(preds_all, labels_all)
    macro_f1_test = macro_f1_from_cm(cm_test)
    
    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test macro f1-score: {macro_f1_test:.4f}")
    return float(avg_test_loss), float(test_accuracy), float(macro_f1_test)

def convert_numpy_floats(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_floats(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_floats(i) for i in obj)
    elif hasattr(obj, 'item'):  
        return obj.item()
    else:
        return obj

client_id = client_id
print(f"客戶端成功啟動，CLIENT_ID = {client_id}")

def connect_to_server():
    server_ip = TCP_IP  
    server_port = TCP_Port  

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((server_ip, server_port))

        message = f"CLIENT_ID={client_id}"
        sock.sendall(message.encode())

        print(f"客戶端 {client_id} 已通報伺服器")
    except Exception as e:
        print(f"連接 TCP 伺服器失敗: {e}")
    finally:
        sock.close()
connect_to_server()

def calculate_data_heterogeneity(dataloader):
    labels = []
    for _, y in dataloader:
        labels.extend(y.numpy())
    
    label_counts = Counter(labels)
    total_samples = sum(label_counts.values())
    probabilities = np.array(list(label_counts.values())) / total_samples
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  
    return round(float(entropy), 4)

def calculate_data_skewness(dataloader):
    labels = []
    for _, y in dataloader:
        labels.extend(y.numpy())
    
    label_counts = np.array(list(Counter(labels).values()))
    mean_count = np.mean(label_counts)
    std_dev = np.std(label_counts)

    if mean_count == 0:
        return 0.0  
    skewness = round(float(std_dev / mean_count), 4) 
    return skewness

def get_peak_memory_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def read_cpu_times():
    with open("/proc/stat", "r") as f:
        parts = f.readline().split()[1:]
        nums = list(map(int, parts))
    idle = nums[3]           
    total = sum(nums)         
    return total, idle

def calc_cpu_percent(t0, t1):
    tot0, idle0 = t0
    tot1, idle1 = t1
    delta_tot = tot1 - tot0
    delta_idle = idle1 - idle0
    if delta_tot == 0:
        return 0.0
    return (1.0 - delta_idle / delta_tot) * 100.0

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_dataloader, val_dataloader, test_dataloader, local_epochs, device, client_id):
        self.model = model
        self.round = 1  
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.local_epochs = local_epochs
        self.device = device  
        self.client_id = client_id 
        self.global_classifier_head = nn.Linear(32, 5).to(device)
        self.prev_features = None 
        self.phi_k_t_minus_1_params_numpy = None
        
    def get_encoder_params_numpy(self):
        params_to_return = []
        # print("[客戶端 DEBUG get_encoder_params_numpy] 提取 encoder 參數:") 
        for name, val in self.model.state_dict().items():
            if 'fc2' not in name: 
                print(f"  - 包括: {name} (形狀: {val.shape})")
                params_to_return.append(val.cpu().numpy())
            else:
                print(f"  - 排除: {name} (形狀: {val.shape})")
        print(f"  get_encoder_params_numpy 返回的參數總數: {len(params_to_return)}")
        return params_to_return    
    
    def calculate_raw_deltas(self, all_phi_params, L_self):
        theta = self.global_classifier_head
        criterion = torch.nn.CrossEntropyLoss()

        raw = {}
        for peer_cid, params_list in all_phi_params.items():
            if str(self.client_id) == peer_cid:
                continue
            peer_model = CNNModel().to(self.device)
            state = peer_model.state_dict()
            for (k, _), arr in zip(state.items(), params_list):
                if 'fc2' not in k:
                    state[k] = torch.from_numpy(np.array(arr)).to(self.device)
            peer_model.load_state_dict(state)
            peer_model.eval()
            loss_peer_sum = loss_peer_count = 0
            with torch.no_grad():
                for X, y in self.val_dataloader:
                    X, y = X.to(self.device), y.to(self.device)
                    feats = peer_model.extract_features(X)
                    out = theta(feats)
                    loss_peer_sum += criterion(out, y).item() * X.size(0)
                    loss_peer_count += X.size(0)
            L_peer = loss_peer_sum / loss_peer_count if loss_peer_count else float('inf')
            denom = 0.0
            self_params = self.get_encoder_params_numpy()
            for pk, pj in zip(self_params, params_list):
                denom += np.sum((pk - np.array(pj))**2)
            denom = np.sqrt(denom) if denom > 1e-9 else 1e-9

            raw[peer_cid] = (L_self - L_peer) / denom

        return raw

    def calculate_affinity_with_peer_protos(self, L_self):

        print(f"[客戶端 {self.client_id}] calculate_affinity_with_peer_protos: 開始計算親和度。")

        if not self.similar_client_phis_params_deserialized: 
            print("  - 沒有收到相似 peer 的 encoder 參數，無法計算親和度。返回空字典。")
            return {}
        if not self.val_dataloader: 
            print("  - 沒有驗證數據加載器 (self.val_dataloader)，無法計算損失。返回空字典。")
            return {}
        if self.global_classifier_head is None: 
            print("  - 全局分類頭 (self.global_classifier_head) 未設置。返回空字典。")
            return {}

        theta_t_minus_1_head = self.global_classifier_head
        theta_t_minus_1_head.eval()

        phi_k_t_minus_1_encoder = self.model 
        phi_k_t_minus_1_encoder.eval()

        criterion_affinity = torch.nn.CrossEntropyLoss()

        raw_delta_kj_scores = {}
        peer_cids_in_order_for_delta = list(self.similar_client_phis_params_deserialized.keys())

        for peer_cid in peer_cids_in_order_for_delta:
            phi_j_t_minus_1_numpy_params = self.similar_client_phis_params_deserialized[peer_cid]

            temp_peer_encoder = CNNModel().to(self.device) 
            encoder_keys = [name for name, _ in temp_peer_encoder.state_dict().items() if 'fc2' not in name]

            print(f"    Peer {peer_cid}: 期望 encoder 參數層數: {len(encoder_keys)}, 收到參數層數: {len(phi_j_t_minus_1_numpy_params)}")
            if len(encoder_keys) == len(phi_j_t_minus_1_numpy_params):
                try:
                    params_dict_for_peer_phi = zip(encoder_keys, phi_j_t_minus_1_numpy_params)
                    state_dict_to_load_for_peer_phi = {
                        k: torch.from_numpy(np.copy(v)).to(self.device) 
                        for k, v in params_dict_for_peer_phi
                    }
                    current_temp_state_dict = temp_peer_encoder.state_dict()
                    current_temp_state_dict.update(state_dict_to_load_for_peer_phi)
                    temp_peer_encoder.load_state_dict(current_temp_state_dict)
                    temp_peer_encoder.eval()
                except Exception as e:
                    print(f"    [錯誤] 加載 Peer {peer_cid} 的 encoder 參數失敗: {e}")
                    raw_delta_kj_scores[peer_cid] = -1e9 
                    continue
            else:
                print(f"    [錯誤] Peer {peer_cid} 的 encoder 參數數量不匹配！給予極低分。")
                raw_delta_kj_scores[peer_cid] = -1e9 
                continue

            loss_k_omega_kj_sum = 0
            loss_k_omega_kj_count = 0
            with torch.no_grad():
                for X_b, y_b in self.val_dataloader:
                    X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                    try:
                        features_peer = temp_peer_encoder.extract_features(X_b)
                        outputs_peer = theta_t_minus_1_head(features_peer)
                        loss_k_omega_kj_sum += criterion_affinity(outputs_peer, y_b).item() * X_b.size(0)
                        loss_k_omega_kj_count += X_b.size(0)
                    except Exception as e:
                        print(f"    [錯誤] 計算 Lk(ωk,j) for Peer {peer_cid} 內部出錯: {e}")
                        loss_k_omega_kj_count = 0 
                        break 
        
            L_k_omega_k_j = loss_k_omega_kj_sum / loss_k_omega_kj_count if loss_k_omega_kj_count > 0 else float('inf')
            if L_k_omega_k_j == float('inf'):
                print(f"    [警告] Peer {peer_cid}: L_k(ωk,j) 計算失敗。給予極低分。")
                raw_delta_kj_scores[peer_cid] = -1e9
                continue
            l2_norm_difference_sq = 0.0
            if len(self.phi_k_t_minus_1_params_numpy) == len(phi_j_t_minus_1_numpy_params):
                for param_k, param_j in zip(self.phi_k_t_minus_1_params_numpy, phi_j_t_minus_1_numpy_params):
                    if param_k.shape == param_j.shape:
                        l2_norm_difference_sq += np.sum(np.square(param_k - param_j))
                    else:
                        print(f"    [警告] Eq.8分母: Peer {peer_cid} 與本地歷史encoder參數形狀不匹配，無法計算L2距離。")
                        l2_norm_difference_sq = float('inf') 
                        break
                if l2_norm_difference_sq == float('inf'):
                    denominator = 1e-9 
                elif l2_norm_difference_sq < 1e-9 :
                    denominator = 1e-9
                else:
                    denominator = np.sqrt(l2_norm_difference_sq)
            else:
                print(f"    [警告] Eq.8分母: Peer {peer_cid} 與本地歷史encoder參數列表長度不匹配，無法計算L2距離。")
                denominator = 1e-9

            print(f"      Peer {peer_cid}: L2_norm_diff_phi = {denominator:.4f}")

            loss_difference = L_self - L_k_omega_k_j
                
            if denominator < 1e-8 :
                raw_delta = loss_difference 
                print(f"      Peer {peer_cid}: 分母過小，raw_delta 約等於損失差 = {raw_delta:.4f}")
            else:
                raw_delta = loss_difference / denominator
            raw_delta_kj_scores[peer_cid] = raw_delta
            print(f"    Peer {peer_cid}: Lk(ωk,j)={L_k_omega_k_j:.4f}, raw_delta_kj={raw_delta:.4f}")

        if not raw_delta_kj_scores:
            print(f"  沒有計算出任何有效的 raw_delta_kj_scores。返回空親和度向量。")
            return {}

        positive_deltas = {cid: max(0, score) for cid, score in raw_delta_kj_scores.items()}
        sum_positive_deltas = sum(positive_deltas.values())
    
        affinity_vector_to_send = {}
        if sum_positive_deltas > 1e-9:
            affinity_vector_to_send = {cid: score / sum_positive_deltas for cid, score in positive_deltas.items()}
        else: 
            affinity_vector_to_send = {cid: 0.0 for cid in positive_deltas.keys()} 
    
        print(f"  計算得到的 affinity_vector: {affinity_vector_to_send}")
        return affinity_vector_to_send
    
    def get_average_representations(self, encoder_model, dataloader, device):
        encoder_model.eval()
        protos_raw = defaultdict(list)
        with torch.no_grad():
            for X_batch, y_batch in dataloader: 
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                features = encoder_model.extract_features(X_batch)
                for feat, label in zip(features, y_batch):
                    protos_raw[int(label.item())].append(feat.cpu())

        return agg_func(protos_raw.copy())

    def fit(self, parameters, config):
        
        t_fit_start = time.perf_counter()
        self.round = int(config.get("round", self.round))
        print(f"[客戶端 {self.client_id}] === 開始第 {self.round} 輪訓練 ===")
        
        current_encoder_params = self.get_encoder_params_numpy()
        if current_encoder_params:
            self.phi_k_t_minus_1_params_numpy = current_encoder_params
        
        is_final_round = (self.round == total_rounds)
        
        criterion = torch.nn.CrossEntropyLoss()
        self.model.eval()
        self.global_classifier_head.eval()
        loss_self_sum = 0.0
        loss_self_count = 0
        with torch.no_grad():
            for X_val, y_val in self.val_dataloader:
                X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                feats = self.model.extract_features(X_val)
                logits = self.global_classifier_head(feats)
                loss_self_sum += criterion(logits, y_val).item() * X_val.size(0)
                loss_self_count += X_val.size(0)
        L_self = loss_self_sum / loss_self_count if loss_self_count else float('inf')
        print(f"[客戶端 {self.client_id}] 本地 L_self = {L_self:.4f}")
        
        all_phi = json.loads(config["all_client_phi_params"])
        raw_deltas = self.calculate_raw_deltas(all_phi, L_self)
        raw_deltas.pop(str(self.client_id), None)
        n = int(config.get("num_similar_clients"))
        top_n = sorted(raw_deltas.items(), key=lambda x: x[1], reverse=True)[:n]
        self.similar_client_phis_params_deserialized = {
            cid: [np.array(p_list, dtype=np.float32) for p_list in all_phi[cid]]
            for cid, _ in top_n
        }

        print(f"[客戶端 {self.client_id}] 本地選出 Top-{n} peers: {list(self.similar_client_phis_params_deserialized)}")
        global_classifier_state_str = config.get("global_classifier_state", "{}")
        if self.round > 1: 
            if global_classifier_state_str and global_classifier_state_str != "{}":
                try:
                    server_state_dict_list_form = json.loads(global_classifier_state_str)
                    server_state_tensor_form = {
                        key: torch.tensor(value, dtype=torch.float32).to(self.device) 
                        for key, value in server_state_dict_list_form.items()
                    }
                    self.global_classifier_head.load_state_dict(server_state_tensor_form)
                    print(f"[客戶端 {self.client_id}] 已更新 self.global_classifier_head 權重")
                except Exception as e:
                    print(f"[客戶端 {self.client_id}] 載入 global_classifier_head 失敗: {e}. 將使用現有 head。")
            else:
                print(f"[客戶端 {self.client_id}] 第 {self.round} 輪，未收到有效 global_classifier_state，將使用現有 head。")
        elif self.round == 1: 
            print(f"[客戶端 {self.client_id}] 第 {self.round} 輪 (伺服器第一輪指令)，不從伺服器加載 classifier head (尚未訓練)。")
        print(f"[客戶端 {self.client_id}] 計算親和度向量 δk (基於 φk^(t-1) 和 φj^(t-1))。")
        affinity_vector_calculated = {} 
        if self.round > 1 and self.similar_client_phis_params_deserialized and \
            self.val_dataloader and len(self.val_dataloader) > 0 and \
            self.phi_k_t_minus_1_params_numpy is not None:
            affinity_vector_calculated = self.calculate_affinity_with_peer_protos(L_self)
        else:
            if self.round == 1:
                print(f"  Fit: 第 {self.round} 輪 (客戶端首次 fit)，親和度計算跳過。")
            elif not self.similar_client_phis_params_deserialized:
                print(f"  Fit: 第 {self.round} 輪，但沒有收到相似 peer 的 encoder 參數，親和度計算跳過。")
            elif not (self.val_dataloader and len(self.val_dataloader) > 0):
                print(f"  Fit: 第 {self.round} 輪，val_dataloader 未提供或為空，親和度計算跳過。")
            elif self.phi_k_t_minus_1_params_numpy is None:
                print(f"  Fit: 第 {self.round} 輪，self.phi_k_t_minus_1_params_numpy 為空，親和度計算跳過。")
        print(f"  Fit: 計算得到的 affinity_vector: {affinity_vector_calculated}")

        if self.round > 1 and affinity_vector_calculated and self.phi_k_t_minus_1_params_numpy is not None:
            peers_with_positive_affinity = {
                p_cid: score for p_cid, score in affinity_vector_calculated.items() 
                if score > 1e-6 and p_cid in self.similar_client_phis_params_deserialized 
            }

            if peers_with_positive_affinity:
                print(f"[客戶端 {self.client_id}] 使用 {len(peers_with_positive_affinity)} 個 peers 的親和度聚合更新本地 encoder。")
                phi_k_prev_params_np = self.phi_k_t_minus_1_params_numpy
                updated_encoder_params_np = [np.copy(p) for p in phi_k_prev_params_np]
                for peer_cid, normalized_delta in peers_with_positive_affinity.items():
                    phi_j_prev_params_np = self.similar_client_phis_params_deserialized[peer_cid]
                
                    if len(updated_encoder_params_np) == len(phi_j_prev_params_np) and \
                        len(updated_encoder_params_np) == len(phi_k_prev_params_np):
                    
                        print(f"  聚合來自 Peer {self.client_id} 的 encoder，正規化權重 δk,j = {normalized_delta:.4f}")
                        for i in range(len(updated_encoder_params_np)):
                            difference = phi_j_prev_params_np[i] - phi_k_prev_params_np[i]
                            updated_encoder_params_np[i] += normalized_delta * difference
                            
                if self.set_encoder_params_from_numpy(updated_encoder_params_np, self.model): 
                    print(f"  成功聚合更新了本地 encoder (self.model)。")
                else: print(f"  [錯誤] 更新本地 encoder 失敗。")

            else:
                print(f"  [警告] 客戶端{self.client_id} 無有效正親和度 peer，跳過 encoder 聚合。")
            
        elif self.round > 1 : print(f"[客戶端 {self.client_id}] 跳過 encoder 聚合 (affinity_empty={not affinity_vector_calculated}, phi_k_prev_empty={self.phi_k_t_minus_1_params_numpy is None})")
        else: print(f"[客戶端 {self.client_id}] 第 {self.round} 輪，不執行 encoder 聚合。")
        mem_before = get_peak_memory_mb()
        cpu_before = read_cpu_times()
        avg_train_loss_for_round = 0.0
        train_accuracy_for_round = 0.0
        avg_val_loss_for_round = 0.0
        val_accuracy_for_round = 0.0 
        macro_f1_train_for_round = 0.0

        print(f"[客戶端 {self.client_id}] 第 {self.round} 輪：訓練本地完整模型 (φk 和 ωk)。")
        avg_train_loss_for_round, train_accuracy_for_round, avg_val_loss_for_round, val_accuracy_for_round, macro_f1_train_for_round = train(
            self.model,                    
            self.global_classifier_head,    
            self.train_dataloader, 
            self.val_dataloader, 
            self.local_epochs, 
            self.device
        )
        
        mem_after = get_peak_memory_mb()
        peak_mem_mb = max(mem_before, mem_after)
        print(f"[客戶端 {self.client_id}] 本輪 RSS 峰值：{peak_mem_mb:.1f} MB")
        
        cpu_after = read_cpu_times()
        client_cpu = calc_cpu_percent(cpu_before, cpu_after)
        print(f"[客戶端 {self.client_id}] 提取當前輪次原型")
        R_k_t_s = self.get_average_representations(self.model, self.train_dataloader, self.device)
        print(f"[客戶端 {self.client_id}] 已提取 R_k_t_s，包含 {len(R_k_t_s)} 個類別。")

        R_k_t_minus_1_s = {} 
        if self.round > 1 and self.phi_k_t_minus_1_params_numpy is not None:
            print(f"[客戶端 {self.client_id}] 提取上一輪原型")
            temp_old_encoder = CNNModel().to(self.device)
            self.set_encoder_params_from_numpy(self.phi_k_t_minus_1_params_numpy, temp_old_encoder)
            R_k_t_minus_1_s = self.get_average_representations(temp_old_encoder, self.train_dataloader, self.device)
            del temp_old_encoder 
            print(f"[客戶端 {self.client_id}] 已提取，包含 {len(R_k_t_minus_1_s)} 個類別。")
        else:
            print(f"[客戶端 {self.client_id}] 第 {self.round} 輪，無舊特徵可提取。")

        print(f"[客戶端 {self.client_id}] 計算")

        if not R_k_t_minus_1_s: 
            R_k_agg_to_send = {k: v.cpu().clone() for k, v in R_k_t_s.items()}
        else:
            TRM_dict = {}
            for label, current_feat in R_k_t_s.items():
                if label in R_k_t_minus_1_s:
                    old_feat = R_k_t_minus_1_s[label].cpu()
                    TRM_val = TRM(
                        old_feat.unsqueeze(0).to(self.device),
                        current_feat.unsqueeze(0).to(self.device),
                        kernel="rbf", device=self.device
                    ).item()
                    TRM_dict[label] = TRM_val

            if not TRM_dict:
                R_k_agg_to_send = {k: v.cpu().clone() for k, v in R_k_t_s.items()}
            else:
                TRM_min = min(TRM_dict.values())
                TRM_max = max(TRM_dict.values())

                R_k_agg_to_send = {}
                for label, current_feat in R_k_t_s.items():
                    curr_cpu = current_feat.cpu()
                    hist_cpu = R_k_t_minus_1_s.get(label, curr_cpu).cpu() 
                    if label in TRM_dict and TRM_max > TRM_min:
                        W = (TRM_dict[label] - TRM_min) / (TRM_max - TRM_min) 
                    else:
                        W = 0.5 
                    R_k_agg_to_send[label] = W * curr_cpu + (1 - W) * hist_cpu

        data_heterogeneity = calculate_data_heterogeneity(self.train_dataloader)
        data_skewness = calculate_data_skewness(self.train_dataloader)
        r = self.round
        history["train_loss"][r] = round(float(avg_train_loss_for_round), 4)
        history["train_accuracy"][r] = round(float(train_accuracy_for_round), 4)
        history["val_loss"][r] = round(float(avg_val_loss_for_round), 4)
        history["val_accuracy"][r] = round(float(val_accuracy_for_round), 4)
        history["train_f1"][r] = round(float(macro_f1_train_for_round), 4)
        history["train_data_heterogeneity"].append(data_heterogeneity)
        history["train_data_skewness"].append(data_skewness)

        print(f"[客戶端 fit 完成] 本地輪次 {self.round} - Train Loss: {avg_train_loss_for_round:.4f}, Train Accuracy: {train_accuracy_for_round:.4f} - Val Loss: {avg_val_loss_for_round:.4f}, Val Accuracy: {val_accuracy_for_round:.4f} - Train F1-score: {macro_f1_train_for_round}")
        print(f"  數據異質性: {data_heterogeneity}, 數據偏斜: {data_skewness}")
        
        metrics_to_send = {
            "train_loss": round(float(avg_train_loss_for_round),4), 
            "train_accuracy": round(float(train_accuracy_for_round),4),
            "val_loss": round(float(avg_val_loss_for_round),4), 
            "val_accuracy": round(float(val_accuracy_for_round),4),
            "train_f1": round(float(macro_f1_train_for_round),4),
            "train_data_heterogeneity": round(float(data_heterogeneity), 4),
            "train_data_skewness": round(float(data_skewness), 4),
            "R_k_agg": json.dumps({str(k): v.numpy().tolist() for k, v in R_k_agg_to_send.items()}),
            "affinity_vector": json.dumps(convert_numpy_floats(affinity_vector_calculated)),
            "client_id": self.client_id,
            "memory_peak_mb": round(peak_mem_mb, 2), 
            "client_cpu_percent": round(client_cpu, 2),
        }

        if is_final_round:
            print(f"[客戶端 {self.client_id}] 最後一輪，計算訓練集混淆矩陣。")
            train_preds_final, train_labels_final = [], []
            self.model.eval()
            self.global_classifier_head.eval()
            with torch.no_grad():
                for X_batch, y_batch in self.train_dataloader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    features = self.model.extract_features(X_batch)
                    outputs = self.global_classifier_head(features)
                    predictions = torch.argmax(outputs, dim=1)
                    train_preds_final.append(predictions)
                    train_labels_final.append(y_batch)
            
            num_classes = len(class_labels) 
            cm_train_np, _ = cm_from_preds_labels(train_preds_final, train_labels_final, num_classes=num_classes)
            metrics_to_send["train_cm"] = json.dumps(cm_train_np.tolist())
            print(f"[客戶端 {self.client_id}] 訓練集混淆矩陣已加入回傳指標。")

        else:
            print(f" fit ({self.round}) ")

        encoder_params_to_send = self.get_encoder_params_numpy()
        metrics_to_send["client_latency"] = time.perf_counter() - t_fit_start
        return encoder_params_to_send, len(self.train_dataloader.dataset), metrics_to_send
    
    def set_encoder_params_from_numpy(self, params_list_numpy, target_model):
        print(f"  [set_encoder_params] 嘗試設置 {len(params_list_numpy)} 組 encoder 參數。")
        encoder_keys = [name for name, _ in target_model.state_dict().items() if 'fc2' not in name]

        if len(encoder_keys) != len(params_list_numpy):
            print(f"    [錯誤] 參數數量不匹配！期望 {len(encoder_keys)} 組，收到 {len(params_list_numpy)} 組。")
            return False 

        params_dict_to_load = {}
        try:
            for i in range(len(encoder_keys)):
                key = encoder_keys[i]
                param_numpy = params_list_numpy[i]
                expected_shape = self.model.state_dict()[key].shape
                if expected_shape != torch.from_numpy(param_numpy).shape: 
                    print(f"    [錯誤] 參數 '{key}' 形狀不匹配！期望 {expected_shape}，收到 {param_numpy.shape}。")
                    return False 
                params_dict_to_load[key] = torch.from_numpy(np.copy(param_numpy)).to(self.device)
            
            current_model_state_dict = target_model.state_dict()
            current_model_state_dict.update(params_dict_to_load) 
            target_model.load_state_dict(current_model_state_dict)
            print(f"  成功設置了 {len(params_dict_to_load)} 組 encoder 參數到 {type(target_model).__name__}。")
        except Exception as e:
            print(f"    [錯誤] set_encoder_params_from_numpy 加載參數時出錯: {e}")
            return False
        return True

    def evaluate(self, parameters, config):
        
        t_eval_start = time.perf_counter()
        self.round = int(config.get("round", self.round))
        print(f"[客戶端 {self.client_id}] === 開始第 {self.round} 輪評估 ===")
        
        is_final_round = (self.round == total_rounds)
        global_classifier_state_str = config.get("global_classifier_state", "{}")
        if global_classifier_state_str and global_classifier_state_str != "{}":
            try:
                server_state_dict_list_form = json.loads(global_classifier_state_str)
                server_state_tensor_form = {
                    key: torch.tensor(value, dtype=torch.float32).to(self.device) 
                    for key, value in server_state_dict_list_form.items()
                }
                self.global_classifier_head.load_state_dict(server_state_tensor_form)
                print(f"[客戶端 {self.client_id}] 評估前已更新 self.global_classifier_head 權重")
            except Exception as e:
                print(f"[客戶端 {self.client_id}] 評估時載入 global_classifier_head 失敗: {e}. 將使用現有 head。")
        else:
            print(f"[客戶端 {self.client_id}] 評估時未收到 global_classifier_state。將使用現有 global_classifier_head。")

        mem_before = get_peak_memory_mb()    
        start_time = time.perf_counter()    
        cpu_before = read_cpu_times()
        loss, accuracy, f1= test(self.model, self.global_classifier_head, self.test_dataloader, self.device)
        end_time = time.perf_counter()

        inference_time = end_time - start_time 
        
        mem_after = get_peak_memory_mb()
        peak_mem_mb = max(mem_before, mem_after)
        print(f"[客戶端 {self.client_id}] 推論 RSS 峰值：{peak_mem_mb:.1f} MB")
        
        cpu_after = read_cpu_times()
        client_cpu = calc_cpu_percent(cpu_before, cpu_after)
        data_heterogeneity = calculate_data_heterogeneity(self.test_dataloader)
        data_skewness = calculate_data_skewness(self.test_dataloader)
        history["test_data_heterogeneity"].append(data_heterogeneity)
        history["test_data_skewness"].append(data_skewness)
        
        print(f"[客戶端測試] 數據異質性程度: {data_heterogeneity}")
        print(f"[客戶端測試] 數據分佈變異係數 (CV): {data_skewness}")
        r = self.round
        history["test_loss"][r] = round(float(loss), 4)
        history["test_accuracy"][r] = round(float(accuracy), 4)
        history["test_f1"][r] = round(float(f1), 4)
        metrics = {"test_loss": round(float(loss),4), 
                   "test_accuracy": round(float(accuracy),4),
                   "test_f1": round(float(f1),4),
                   "test_data_heterogeneity": round(float(data_heterogeneity), 4),
                   "test_data_skewness": round(float(data_skewness), 4),
                   "inference_time": inference_time,
                   "memory_peak_mb": round(peak_mem_mb, 2),
                   "client_cpu_percent": round(client_cpu, 2),
                   }
        if is_final_round:
            print(f"[客戶端 {self.client_id}] 最後一輪，計算測試集混淆矩陣。")
            test_preds_final, test_labels_final = [], []
            self.model.eval()
            self.global_classifier_head.eval()
            with torch.no_grad():
                for X_batch, y_batch in self.test_dataloader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    features = self.model.extract_features(X_batch)
                    outputs = self.global_classifier_head(features)
                    predictions = torch.argmax(outputs, dim=1)
                    test_preds_final.append(predictions)
                    test_labels_final.append(y_batch)
        
            num_classes = len(class_labels) 
            cm_test_np, _ = cm_from_preds_labels(test_preds_final, test_labels_final, num_classes=num_classes)
            metrics["test_cm"] = json.dumps(cm_test_np.tolist())
            print(f"[客戶端 {self.client_id}] 測試集混淆矩陣已加入回傳指標。")
        
        print(f"[客戶端] 返回的測試指標: {metrics}")
        metrics["client_latency"] = time.perf_counter() - t_eval_start
        return float(loss), len(self.test_dataloader.dataset), metrics
    
    def save_training_log_csv(self):
        all_rounds = sorted(set(
            history["train_loss"].keys() |
            history["val_loss"].keys() |
            history["test_loss"].keys() |
            history["train_accuracy"].keys() |
            history["val_accuracy"].keys() |
            history["test_accuracy"].keys() |
            history["train_f1"].keys() |
            history["test_f1"].keys()
        ))

        df = pd.DataFrame({
            "Round": all_rounds,
            "Train Loss": [history["train_loss"].get(r, "") for r in all_rounds],
            "Validation Loss": [history["val_loss"].get(r, "") for r in all_rounds],
            "Test Loss": [history["test_loss"].get(r, "") for r in all_rounds],
            "Train Accuracy": [history["train_accuracy"].get(r, "") for r in all_rounds],
            "Validation Accuracy": [history["val_accuracy"].get(r, "") for r in all_rounds],
            "Test Accuracy": [history["test_accuracy"].get(r, "") for r in all_rounds],
            "Train F1-score": [history["train_f1"].get(r, "") for r in all_rounds],
            "Test F1-score": [history["test_f1"].get(r, "") for r in all_rounds],
        })

        df.to_csv(history_log_file, index=False)
        print(f"訓練測試歷史紀錄結果已儲存：{history_log_file}")

    def visualize_progress(self):
        print("[客戶端] 訓練損失：", dict(sorted(history["train_loss"].items())))
        print("[客戶端] 測試損失：", dict(sorted(history["test_loss"].items())))
        print("[客戶端] 訓練準確：", dict(sorted(history["train_accuracy"].items())))
        print("[客戶端] 測試準確：", dict(sorted(history["test_accuracy"].items())))

        if not history["train_loss"] or not history["test_loss"] or not history["train_accuracy"] or not history["test_accuracy"]:
            print("[警告] 訓練和測試過程中沒有數據被儲存，無法繪製圖表！")
            return

        try:
            plt.figure(figsize=(14, 5))
            x_epochs = range(1, len(history["train_loss"]) + 1)
            plt.subplot(1, 2, 1)
            plt.plot(x_epochs, history["train_loss"], marker='x', label='Train Loss')
            plt.plot(x_epochs, history["test_loss"], marker='o', label='Local Test Loss')
            plt.title("Model Loss")
            plt.xlabel("Rounds")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.subplot(1, 2, 2)
            plt.plot(x_epochs, history["train_accuracy"], marker='x', label='Train Accuracy')
            plt.plot(x_epochs, history["test_accuracy"], marker='o', label='Local Test Accuracy')
            plt.title("Model Accuracy")
            plt.xlabel("Rounds")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
        
            plt.savefig(image_file)
            print(f"訓練曲線圖已儲存至 '{image_file}'\n")
            
        except Exception as e:
            print(f"[警告] 繪圖時發生錯誤：{e}。跳過畫圖並繼續。")

print("[客戶端] 初始化...")
local_epochs = local_epochs
history = {
    "train_loss": {}, 
    "val_loss": {},
    "test_loss": {},
    "train_accuracy": {},
    "val_accuracy": {},
    "test_accuracy": {},
    "train_f1": {},
    "test_f1": {},
    "train_data_heterogeneity": [],
    "test_data_heterogeneity": [],
    "train_data_skewness": [],
    "test_data_skewness": [],
    "TRM_value": [],
}

client = FlowerClient(model, train_dataloader, val_dataloader, test_dataloader, local_epochs, device, client_id)
fl.client.start_client(
    server_address=server_IP, 
    client=client.to_client(),
    grpc_max_message_length=1024 * 1024 * 100,
)

client.save_training_log_csv()
client.visualize_progress()

train_labels = []
train_predictions = []
test_labels = []
test_predictions = []

final_encoder = client.model
final_global_head = client.global_classifier_head
final_encoder.to(device).eval()
final_global_head.to(device).eval()

with torch.no_grad():
    for X_batch, y_batch in train_dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        features = final_encoder.extract_features(X_batch)
        outputs = final_global_head(features)
        _, predicted = torch.max(outputs, 1)
        train_labels.extend(y_batch.cpu().numpy())
        train_predictions.extend(predicted.cpu().numpy())

with torch.no_grad():
    for X_batch, y_batch in test_dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        features = final_encoder.extract_features(X_batch)
        outputs = final_global_head(features)
        _, predicted = torch.max(outputs, 1)
        test_labels.extend(y_batch.cpu().numpy())
        test_predictions.extend(predicted.cpu().numpy())

def report_missing(labels_list, label_type):
    present = set(labels_list)
    missing = set(fixed_label_indices) - present
    if missing:
        missing_names = [index_to_label[i] for i in sorted(missing)]
        print(f"[警告] {label_type} 中缺少類別：{missing_names}，混淆矩陣將有全為 0 的列/欄。")

report_missing(train_labels, "訓練資料")
report_missing(test_labels, "測試資料")

train_cm = confusion_matrix(train_labels, train_predictions, labels=fixed_label_indices)
test_cm = confusion_matrix(test_labels, test_predictions, labels=fixed_label_indices)

train_cm_df = pd.DataFrame(train_cm, index=class_labels, columns=class_labels)
test_cm_df = pd.DataFrame(test_cm, index=class_labels, columns=class_labels)

train_cm_df.to_csv(train_confusion_matrix_csv_path)
print(f"訓練集混淆矩陣數值已儲存至 '{train_confusion_matrix_csv_path}'")
test_cm_df.to_csv(test_confusion_matrix_csv_path)
print(f"測試集混淆矩陣數值已儲存至 '{test_confusion_matrix_csv_path}'")

plt.figure(figsize=(12, 10))
sns.heatmap(train_cm_df, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Training Data")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(train_confusion_matrix)
print(f"訓練集混淆矩陣圖已儲存至 '{train_confusion_matrix}'")

plt.figure(figsize=(12, 10))
sns.heatmap(test_cm_df, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Testing Data")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(test_confusion_matrix)
print(f"測試集混淆矩陣圖已儲存至 '{test_confusion_matrix}'")

def safe_normalize(cm):
    with np.errstate(divide='ignore', invalid='ignore'):
        row_sums = cm.sum(axis=1, keepdims=True)
        normalized = np.divide(cm, row_sums, where=row_sums != 0)
        normalized[np.isnan(normalized)] = 0.0
    return normalized

train_cm_percent = safe_normalize(train_cm)
test_cm_percent = safe_normalize(test_cm)

plt.figure(figsize=(12, 10))
sns.heatmap(train_cm_percent, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix (Percentage) - Training Data")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(train_confusion_matrix_Percentage_path)
print(f"訓練集混淆矩陣(百分比)圖已儲存至 '{train_confusion_matrix_Percentage_path}'")

plt.figure(figsize=(12, 10))
sns.heatmap(test_cm_percent, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix (Percentage) - Testing Data")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(test_confusion_matrix_Percentage_path)
print(f"測試集混淆矩陣(百分比)圖已儲存至 '{test_confusion_matrix_Percentage_path}'")

def calculate_metrics(cm, index_to_label):
    num_classes = cm.shape[0]

    metrics = {
        "Class Index": [],
        "Class Label": [],
        "TP": [],
        "FP": [],
        "TN": [],
        "FN": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1-Score": []
    }
    for i in range(num_classes):
        TP = cm[i, i] 
        FP = cm[:, i].sum() - TP  
        FN = cm[i, :].sum() - TP 
        TN = cm.sum() - (TP + FP + FN) 
        Accuracy = round((TP + TN) / cm.sum(),4) if cm.sum() != 0 else 0
        Precision = round(TP / (TP + FP),4) if (TP + FP) != 0 else 0
        Recall = round(TP / (TP + FN), 4) if (TP + FN) != 0 else 0
        F1_Score = round(2 * Precision * Recall / (Precision + Recall), 4) if (Precision + Recall) != 0 else 0
        metrics["Class Index"].append(i)
        metrics["Class Label"].append(index_to_label.get(i, f"Class {i}"))
        metrics["TP"].append(TP)
        metrics["FP"].append(FP)
        metrics["TN"].append(TN)
        metrics["FN"].append(FN)
        metrics["Accuracy"].append(Accuracy)
        metrics["Precision"].append(Precision)
        metrics["Recall"].append(Recall)
        metrics["F1-Score"].append(F1_Score)
        
    return pd.DataFrame(metrics)
train_metrics_df = calculate_metrics(train_cm, index_to_label)
test_metrics_df = calculate_metrics(test_cm, index_to_label)

train_metrics_df.to_csv(train_metrics, index=False)
test_metrics_df.to_csv(test_metrics, index=False)

print(f"訓練集每個類別的指標已儲存至 '{train_metrics}'")
print(train_metrics_df)

print(f"\n測試集每個類別的指標已儲存至 '{test_metrics}'")
print(test_metrics_df)

y_true_for_roc = []
y_scores_for_roc = [] 

with torch.no_grad():
    for X_batch, y_batch in test_dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        features = final_encoder.extract_features(X_batch)
        outputs = final_global_head(features)        
        y_true_for_roc.extend(y_batch.cpu().numpy())
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        y_scores_for_roc.extend(probabilities)

y_true = np.array(y_true_for_roc)
y_scores = np.array(y_scores_for_roc)
num_classes = y_scores.shape[1]
y_true_bin = label_binarize(y_true, classes=range(num_classes)) if len(y_true) > 0 else np.array([])

fpr, tpr, roc_auc = {}, {}, {}

plt.figure(figsize=(10, 7))
for i in range(num_classes):
    if len(np.unique(y_true_bin[:, i])) > 1:
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        label = index_to_label.get(i, f"Class {i}")
        plt.plot(fpr[i], tpr[i], lw=2, label=f'{label} (AUC = {roc_auc[i]:.2f})')
    else:
        print(f" 類別 {i} 無法計算 ROC（僅單一類樣本）。")
        roc_auc[i] = np.nan

if y_scores.ndim == 2 and y_scores.shape[1] == num_classes and len(np.unique(y_true)) > 1:
    try:
        macro_auc = roc_auc_score(y_true_bin, y_scores, average="macro", multi_class="ovr")
        print(f" Macro AUC: {macro_auc:.4f}")
    except Exception as e:
        print(f" 無法計算 Macro AUC: {e}")
        macro_auc = np.nan
else:
    print(" 資料不足，跳過 Macro AUC 計算。")
    macro_auc = np.nan

if plt.gca().has_data():
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("Multi-Class ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize="small")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(roc_curve_file)
    print(f"ROC 曲線圖已儲存至 '{roc_curve_file}'")
else:
    print("[警告] 無可用 ROC 資料，未繪圖。")
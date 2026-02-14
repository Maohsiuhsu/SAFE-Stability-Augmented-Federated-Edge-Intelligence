"""
伺服器程式碼 - 使用 Flower 進行聯邦學習
"""
import os
import flwr as fl
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
import socket
import threading
import json
import numpy as np
from flwr.server.strategy import Strategy
from flwr.common import FitIns, EvaluateIns, parameters_to_ndarrays
from collections import defaultdict
import math
import pandas as pd
import time
import seaborn as sns



dataset_name = "NSLKDD"  
strategy_name = "SAFE"  
ip = "0.0.0.0:8080"  
TCP_Port = 8888  
num_expected_clients = 50  
num_rounds = 100  
Ratio = 1  



output_image = "/home/root/server/"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
image_server = os.path.join(
    output_image,
    "image_server",
    dataset_name,
    strategy_name,
    f"total_clients_{num_expected_clients}",
    f"rounds_{num_rounds}",
    f"sample_{int(Ratio * 100)}%"
)
os.makedirs(image_server, exist_ok=True)
server_training_plot = os.path.join(image_server, f"server_training_plots_{timestamp}.png")
server_history_log = os.path.join(image_server, f"server_history_log_{timestamp}.csv")
aggregated_cm_plot = os.path.join(image_server, f"aggregated_confusion_matrix_{timestamp}.png")
aggregated_cm_csv = os.path.join(image_server, "aggregated_confusion_matrix_csv")
os.makedirs(aggregated_cm_csv, exist_ok=True)
aggregated_train_cm_csv_path = os.path.join(aggregated_cm_csv, f"aggregated_train_cm_{timestamp}.csv")
aggregated_test_cm_csv_path = os.path.join(aggregated_cm_csv, f"aggregated_test_cm_{timestamp}.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置：{device}")

class CustomStrategy(Strategy):
    def __init__(self, num_expected_clients):
        super().__init__()
        self.global_classifier = None  # θ^t
        self.affinity_matrix = {}      # M_N
        self.client_feature_extractors_params = {}
        self.round = 1
        self.num_expected_clients = num_expected_clients 
        self.known_client_cids = set()

    def initialize_parameters(self, client_manager):
        print("[伺服器] 初始化全域分類器")

        feature_dim = 32  
        num_classes = 5   
        self.global_classifier = torch.nn.Linear(feature_dim, num_classes).to("cpu")
        
        print("[伺服器] 全局分類器初始化完成。親和度矩陣 MN 將動態構建。")

        return None

    def configure_fit(self, server_round, parameters, client_manager):
        
        t0 = time.perf_counter()
        
        self.round = server_round 

        print(f"[伺服器] configure_fit (第 {server_round} 輪) 開始。")

        if self.global_classifier is None: 
            print("[伺服器][嚴重錯誤] configure_fit: self.global_classifier 未被初始化！")
            return []

        if parameters is not None:
            print(f"  收到 Flower 框架傳遞的 parameters (可能來自上一輪聚合或客戶端初始參數)。")
            try:
                ndarray_list = parameters_to_ndarrays(parameters)
                expected_num_params = len(list(self.global_classifier.parameters()))
                if len(ndarray_list) == expected_num_params:
                    params_dict = {
                        name: torch.from_numpy(np.copy(param_array))
                        for name, param_array in zip(self.global_classifier.state_dict().keys(), ndarray_list)
                    }
                    self.global_classifier.load_state_dict(params_dict)
                    print("    已成功將 Flower 傳遞的參數加載到 self.global_classifier。")
                elif server_round == 1 and len(ndarray_list) > expected_num_params:
                    print(f"    [警告] 第 1 輪: Flower 傳遞的參數數量 ({len(ndarray_list)}) 多於 global_classifier ({expected_num_params})，可能來自客戶端完整模型。將使用 self.global_classifier 的初始隨機權重。")
                else: 
                    print(f"    [警告] 傳入的 parameters 數量 ({len(ndarray_list)}) 與 self.global_classifier ({expected_num_params}) 不匹配。將使用 self.global_classifier 現有權重。")
            except Exception as e:
                print(f"    [錯誤] 加載 Flower 傳遞的 parameters 到 self.global_classifier 失敗: {e}。將使用現有權重。")
        else: 
            if server_round == 1:
                print(f"  第 1 輪: Flower 未提供初始 parameters (可能獲取客戶端初始參數失敗)。將使用 self.global_classifier 的初始隨機權重。")
            else: 
                print(f"  [警告] 第 {server_round} 輪:你完全正確！我之前的修改建議中")
                
        global_classifier_state_serialized = "{}"
        try:
            gc_state_dict = self.global_classifier.state_dict()
            gc_state_list_form = {k: v.cpu().numpy().tolist() for k, v in gc_state_dict.items()}
            global_classifier_state_serialized = json.dumps(gc_state_list_form)
            print(f"  準備發送的 global_classifier_state (len {len(global_classifier_state_serialized)}, 前100): {global_classifier_state_serialized[:100]}...")
        except Exception as e:
            print(f"  [錯誤] 序列化 self.global_classifier 失敗: {e}。將發送空狀態。")
            global_classifier_state_serialized = "{}"

        if parameters is not None:
            ndarray_list = parameters_to_ndarrays(parameters)
            model_bytes = sum(arr.nbytes for arr in ndarray_list)
        else:
            model_bytes = 0    

        k = max(1, math.ceil(Ratio * self.num_expected_clients))
        print(f"  總共設定 {self.num_expected_clients} 個客戶端，隨機選 {Ratio*100}% → {k} 個參與訓練。")

        comm_cost = 2 * model_bytes * k
        history["comm_bytes_per_round"].append(comm_cost)
        print(f"[伺服器] 第 {server_round} 輪 通訊成本：{comm_cost} bytes")
        
        if parameters is not None:
            ndarray_list = parameters_to_ndarrays(parameters)
        else:
            ndarray_list = [val.cpu().numpy() for val in self.global_classifier.state_dict().values()]

        model_size = sum(arr.nbytes for arr in ndarray_list)
        history["model_size_bytes"].append(model_size)
        print(f"[伺服器] 第 {server_round} 輪 模型大小：{model_size} bytes ({model_size/1024/1024:.2f} MB)")
        
        clients = client_manager.sample(
            num_clients=k,
            min_num_clients=self.num_expected_clients,  
            )
        if not clients:
            print(f"[伺服器][警告] configure_fit: 抽樣不到 {k} 個客戶端，取消本輪 fit。")
            return []
        print(f"  成功抽樣 {len(clients)} 個客戶端進行 fit。")

        msg_cnt = 2 * k
        history["messages_per_round"].append(msg_cnt)
        prev = history["cumulative_messages"][-1] if history["cumulative_messages"] else 0
        history["cumulative_messages"].append(prev + msg_cnt)
        print(f"[通訊次數] 第 {server_round} 輪：本輪 {msg_cnt} 次，累計 {prev + msg_cnt} 次")

        all_phi_serialized = "{}"
        if self.client_feature_extractors_params:
            all_phi_serialized = json.dumps({
                cid: [p.tolist() for p in params]
                for cid, params in self.client_feature_extractors_params.items()
            })

        current_global_params = fl.common.ndarrays_to_parameters(
            [v.cpu().numpy() for v in self.global_classifier.state_dict().values()]
        )
        fit_ins_list = []
        for client_proxy in clients:
            client_config = {
                "global_classifier_state": global_classifier_state_serialized,
                "round": str(server_round),
                "num_similar_clients": str(5),
                "all_client_phi_params": all_phi_serialized,
                "num_total_rounds_from_server": num_rounds,
            }
            fit_ins_list.append((client_proxy, FitIns(current_global_params, client_config)))
            
        t1 = time.perf_counter()
        latency = t1 - t0
        history["server_sched_fit_latency"].append(latency)
        print(f"[伺服器] 第 {server_round} 輪 configure_fit 排程耗時：{latency:.4f} 秒")
            
        return fit_ins_list
    
    def aggregate_fit(self, server_round, results, failures):
        
        t0 = time.perf_counter()
    
        if not results:
            print(f"[伺服器] aggregate_fit (第 {server_round} 輪): 沒有收到客戶端結果。")
            if self.global_classifier:
                current_global_classifier_params_ndarrays = [p.cpu().numpy() for _, p in self.global_classifier.state_dict().items()]
                return fl.common.ndarrays_to_parameters(current_global_classifier_params_ndarrays), {}
            return None, {}

        print(f"[伺服器] aggregate_fit (第 {server_round} 輪): 收到 {len(results)} 個客戶端 fit 結果。")
    
        all_R_k_agg_from_clients_parsed = [] 
        metrics_tuples_for_calculation = [] 

        is_final_round = (server_round == num_rounds)
        aggregated_cm_train = None

        for client_proxy, fit_res in results: 
            client_id_str = client_proxy.cid

            metrics_from_client = fit_res.metrics 
            num_examples_fit = fit_res.num_examples

            metrics_tuples_for_calculation.append((num_examples_fit, metrics_from_client))

            if is_final_round:
                train_cm_str = metrics_from_client.get("train_cm")
                if train_cm_str:
                    try:
                        client_cm = np.array(json.loads(train_cm_str))
                        if aggregated_cm_train is None:
                            aggregated_cm_train = client_cm
                        else:
                            if aggregated_cm_train.shape == client_cm.shape:
                                aggregated_cm_train += client_cm
                            else:
                                print(f"[警告] Client {client_id_str} 的訓練混淆矩陣形狀不匹配，已跳過。")
                        print(f"  已聚合 Client {client_id_str} 的訓練混淆矩陣。")
                    except Exception as e:
                        print(f"[警告] 解析 Client {client_id_str} 的訓練混淆矩陣失敗: {e}")
            if client_id_str not in self.known_client_cids:
                self.known_client_cids.add(client_id_str)
                self.affinity_matrix[client_id_str] = {cid: 0.0 for cid in self.known_client_cids} 
                self.affinity_matrix[client_id_str][client_id_str] = 1.0 
                for known_cid in self.known_client_cids:
                    if known_cid != client_id_str:
                        self.affinity_matrix[known_cid][client_id_str] = 0.0 
                print(f"  新 Client {client_id_str} 加入，已更新 affinity_matrix 結構。")

            affinity_vector_str = metrics_from_client.get("affinity_vector", "{}") 
            if affinity_vector_str and affinity_vector_str != "{}" and server_round > 0 : 
                try:
                    affinity_updates_from_client_k = json.loads(affinity_vector_str) 
                    if isinstance(affinity_updates_from_client_k, dict):
                        print(f"  收到 Client {client_id_str} 的 affinity_vector: {affinity_updates_from_client_k}")
                        for peer_j_cid, delta_kj_score in affinity_updates_from_client_k.items():
                            if peer_j_cid in self.affinity_matrix.get(client_id_str, {}):
                                self.affinity_matrix[client_id_str][peer_j_cid] = delta_kj_score
                        print(f"  已使用 Client {client_id_str} 的 affinity_vector 更新其在 MN 中的行。")
                    else:
                        print(f"[伺服器][警告] aggregate_fit: Client {client_id_str} 的 affinity_vector 解析後不是字典。")
                except Exception as e:
                    print(f"[伺服器][警告] aggregate_fit: 解析 Client {client_id_str} 的 affinity_vector 失敗: {e}")
            elif server_round == 1: 
                print(f"  Client {client_id_str}: 第 {server_round} 輪 (客戶端初次fit)，不處理空的/無效的 affinity_vector。")
            else:
                print(f"[伺服器] aggregate_fit: Client {client_id_str} 未發送有效的 (非空) affinity_vector。")
            client_encoder_params_ndarrays = fl.common.parameters_to_ndarrays(fit_res.parameters)
            if client_encoder_params_ndarrays: 
                self.client_feature_extractors_params[client_id_str] = client_encoder_params_ndarrays
                print(f"  已存儲 Client {client_id_str} 更新後的 encoder 參數 (共 {len(client_encoder_params_ndarrays)} 層)。")

            r_k_agg_str = metrics_from_client.get("R_k_agg", "{}")
            if not r_k_agg_str or r_k_agg_str == "{}":
                print(f"[伺服器][警告] aggregate_fit: Client {client_id_str} 未發送有效的 R_k_agg。")
            else:
                try:
                    r_k_agg_dict_list_form = json.loads(r_k_agg_str) 
                    parsed_r_k_agg_for_client = {
                        int(label_str): torch.tensor(feat_vec_list, dtype=torch.float32).cpu()
                        for label_str, feat_vec_list in r_k_agg_dict_list_form.items()
                    }
                    if parsed_r_k_agg_for_client:
                        all_R_k_agg_from_clients_parsed.append(parsed_r_k_agg_for_client)
                        print(f"  已解析 Client {client_id_str} 的 R_k_agg，包含 {len(parsed_r_k_agg_for_client)} 個類別。")
                except Exception as e:
                    print(f"[伺服器][錯誤] aggregate_fit: 解析 Client {client_id_str} 的 R_k_agg 失敗: {e}")

        if is_final_round and aggregated_cm_train is not None:
            history["aggregated_train_cm"] = aggregated_cm_train
            print("[伺服器] 已成功匯總所有客戶端的訓練混淆矩陣。")
        
        global_protos_for_classifier_training = defaultdict(list)
        for r_k_agg_single_client_dict in all_R_k_agg_from_clients_parsed: 
            for label_int, feat_tensor in r_k_agg_single_client_dict.items():
                global_protos_for_classifier_training[label_int].append(feat_tensor.cpu())
        final_global_protos_to_train_on = {}
        if global_protos_for_classifier_training:
            for label_int, tensors_for_label in global_protos_for_classifier_training.items():
                if tensors_for_label: 
                    final_global_protos_to_train_on[label_int] = torch.stack(tensors_for_label).mean(dim=0).cpu()
            print(f"  已聚合所有客戶端的 R_k_agg，準備訓練全局分類器。共 {len(final_global_protos_to_train_on)} 個類別的原型。")
        if final_global_protos_to_train_on and self.global_classifier:
            print("[伺服器] aggregate_fit: 使用聚合後的 R_k_agg 訓練全局分類器 Θ^t")
            self.train_global_classifier(final_global_protos_to_train_on) 
        elif not self.global_classifier:
            print("[伺服器][錯誤] aggregate_fit: global_classifier 未初始化，無法訓練！")
        else:
            print("[伺服器] aggregate_fit: 沒有有效的聚合 R_k_agg 可用於訓練全局分類器。")

        aggregated_fit_metrics_std = {} 
        if metrics_tuples_for_calculation: 
            aggregated_fit_metrics_std = fit_metrics_calculate(metrics_tuples_for_calculation) 
            print(f"[伺服器] fit_metrics_calculate 返回: {aggregated_fit_metrics_std}")

        parameters_to_return_for_flower = None
        if self.global_classifier:
            updated_global_classifier_params_ndarrays = [val.cpu().numpy() for _, val in self.global_classifier.state_dict().items()]
            parameters_to_return_for_flower = fl.common.ndarrays_to_parameters(updated_global_classifier_params_ndarrays)
        else:
            print("[伺服器][警告] aggregate_fit: global_classifier 不可用，返回 None 作為參數。")
        
        final_metrics_for_flower = aggregated_fit_metrics_std.copy() 
        
        t1 = time.perf_counter()
        latency = t1 - t0
        history["server_agg_fit_latency"].append(latency)
        print(f"[伺服器] 第 {server_round} 輪 aggregate_fit 聚合耗時：{latency:.4f} 秒")
        
        return parameters_to_return_for_flower, final_metrics_for_flower

    def configure_evaluate(self, server_round, parameters, client_manager):
        t0 = time.perf_counter()
        num_available_clients = client_manager.num_available()
        if num_available_clients == 0:
            print("[伺服器] configure_evaluate: 沒有可用的客戶端進行評估。")
            return []

        clients_to_evaluate = client_manager.sample(
            num_clients=num_available_clients,
            min_num_clients=num_available_clients,
        )
        
        if not clients_to_evaluate:
            print("[伺服器] configure_evaluate: 未能成功要求全部客戶端進行評估。")
            return []

        print(f"[伺服器] configure_evaluate: 為 {len(clients_to_evaluate)} 個客戶端配置第 {server_round} 輪評估。")
        
        classifier_state_str = "{}"
        if self.global_classifier:
            try:
                classifier_state = {
                    k: v.cpu().numpy().tolist()
                    for k, v in self.global_classifier.state_dict().items()
                }
                classifier_state_str = json.dumps(classifier_state)
                print(f"[伺服器] configure_evaluate: 傳送序列化 global_classifier ")
            except Exception as e:
                print(f"[伺服器][錯誤] configure_evaluate: 序列化 global_classifier 失敗: {e}")
        else:
            print("[伺服器][警告] configure_evaluate: global_classifier 尚未初始化！將發送空分類器狀態。")

        evaluate_instructions = []
        for client_proxy in clients_to_evaluate:
            client_config = {
                "global_classifier_state": classifier_state_str,
                "round": str(server_round) 
            }
            evaluate_instructions.append(
                (client_proxy, EvaluateIns(parameters=parameters, config=client_config))
            )
            
        t1 = time.perf_counter()
        history["server_sched_eval_latency"].append(t1 - t0)
        print(f"[伺服器] 第 {server_round} 輪 configure_evaluate 排程耗時：{t1-t0:.4f} 秒")    
            
        return evaluate_instructions

    def aggregate_evaluate(self, server_round: int, results, failures):        
        t0 = time.perf_counter()
        
        if not results:
            print(f"[伺服器] aggregate_evaluate (第 {server_round} 輪): 沒有收到評估結果。")
            return None, {} 
        
        print(f"[伺服器] 第 {server_round} 輪收到 {len(results)} 個客戶端評估結果。")
 
        metrics_for_calculation = []
        total_loss_weighted_sum = 0
        total_examples_sum = 0

        is_final_round = (server_round == num_rounds)
        aggregated_cm_test = None

        for client_proxy, eval_res in results:
            num_examples = eval_res.num_examples
            loss_from_client = eval_res.loss
            metrics_from_client = eval_res.metrics

            if is_final_round:
                test_cm_str = metrics_from_client.get("test_cm")
                if test_cm_str:
                    try:
                        client_cm = np.array(json.loads(test_cm_str))
                        if aggregated_cm_test is None:
                            aggregated_cm_test = client_cm
                        else:
                            if aggregated_cm_test.shape == client_cm.shape:
                                aggregated_cm_test += client_cm
                            else:
                                print(f"[警告] Client {client_proxy.cid} 的測試混淆矩陣形狀不匹配，已跳過。")
                        print(f"  已聚合 Client {client_proxy.cid} 的測試混淆矩陣。")
                    except Exception as e:
                        print(f"[警告] 解析 Client {client_proxy.cid} 的測試混淆矩陣失敗: {e}")

            current_client_metrics_dict = {
                "test_loss": loss_from_client,
                **metrics_from_client
            }
            metrics_for_calculation.append((num_examples, current_client_metrics_dict))
            
            total_loss_weighted_sum += loss_from_client * num_examples
            total_examples_sum += num_examples

        if is_final_round and aggregated_cm_test is not None:
            history["aggregated_test_cm"] = aggregated_cm_test
            print("[伺服器] 已成功匯總所有客戶端的測試混淆矩陣。")    

        if not metrics_for_calculation:
            print(f"[伺服器] aggregate_evaluate (第 {server_round} 輪): 沒有有效的客戶端評估指標可以聚合。")
            return None, {}

        aggregated_eval_metrics_dict = evaluate_metrics_calculate(metrics_for_calculation)
 
        aggregated_loss = total_loss_weighted_sum / total_examples_sum if total_examples_sum > 0 else None
        
        print(f"[伺服器評估聚合完成] 第 {server_round} 輪 - 聚合損失（直接計算）: {aggregated_loss if aggregated_loss is not None else 'N/A'}")

        final_metrics_for_flower = {}
        if "test_accuracy" in aggregated_eval_metrics_dict:
            final_metrics_for_flower["accuracy"] = aggregated_eval_metrics_dict["test_accuracy"]

        for key, value in aggregated_eval_metrics_dict.items():
            if isinstance(value, (int, float)): 

                if key not in final_metrics_for_flower:
                    final_metrics_for_flower[key] = value
                    
        t1 = time.perf_counter()
        history["server_agg_eval_latency"].append(t1 - t0)
        print(f"[伺服器] 第 {server_round} 輪 aggregate_evaluate 聚合耗時：{t1-t0:.4f} 秒")            

        return aggregated_loss, final_metrics_for_flower

    def evaluate(self, server_round, parameters):
        return None, {}

    def train_global_classifier(self, protos):
        X = []
        y = []
        for label, vector in protos.items():
            X.append(vector)
            y.append(label)
        X = torch.stack(X)
        y = torch.tensor(y)

        self.global_classifier.train()
        optimizer = torch.optim.Adam(self.global_classifier.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        for _ in range(10): 
            optimizer.zero_grad()
            logits = self.global_classifier(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        print("[伺服器] 分類器訓練完成，loss =", loss.item())

history = {
    "train_loss": [],
    "train_accuracy": [],
    "test_loss": [],
    "test_accuracy": [], 
    "best_train_loss": [],
    "worst_train_loss": [],
    "best_test_loss": [],
    "worst_test_loss": [],
    "best_train_accuracy": [],
    "worst_train_accuracy": [],
    "best_test_accuracy": [],
    "worst_test_accuracy": [],
    "train_max_data_heterogeneity": [],  
    "train_min_data_heterogeneity": [], 
    "test_max_data_heterogeneity": [],  
    "test_min_data_heterogeneity": [], 
    "train_max_data_skewness": [], 
    "train_min_data_skewness": [],  
    "test_max_data_skewness": [], 
    "test_min_data_skewness": [],
    "avg_inference_time": [],
    "comm_bytes_per_round": [], 
    "messages_per_round": [],
    "cumulative_messages": [], 
    "model_size_bytes": [],
    "train_memory_peak_mb": [],
    "test_memory_peak_mb": [],
    "train_client_latency": [], 
    "test_client_latency": [],
    "server_sched_fit_latency": [],     
    "server_agg_fit_latency": [],      
    "server_sched_eval_latency": [],    
    "server_agg_eval_latency": [],    
    "train_avg_client_cpu": [],
    "test_avg_client_cpu": [],
    "total_time_seconds": [],
    "train_f1": [],
    "test_f1": [],
    "aggregated_train_cm": None, 
    "aggregated_test_cm": None, 
}

def fit_metrics_calculate(metrics):
    
    train_losses = [num_examples * m["train_loss"] for num_examples, m in metrics]  
    train_accuracies = [num_examples * m["train_accuracy"] for num_examples, m in metrics] 
    train_F1_all = [num_examples * m["train_f1"] for num_examples, m in metrics] 
    each_train_loss = [m["train_loss"] for _, m in metrics]  
    each_train_accuracy = [m["train_accuracy"] for _, m in metrics]  
    train_data_heterogeneities = [m["train_data_heterogeneity"] for _, m in metrics] 
    train_data_skewnesses = [m["train_data_skewness"] for _, m in metrics]  
    memory_peaks = [m["memory_peak_mb"] for _, m in metrics if "memory_peak_mb" in m]  
    latencies = [m["client_latency"] for _, m in metrics if "client_latency" in m]  
    cpu_list = [m["client_cpu_percent"] for _, m in metrics if "client_cpu_percent" in m]  
    examples = [num_examples for num_examples, _ in metrics]

    weighted_loss = sum(train_losses) / sum(examples)  
    weighted_accuracy = sum(train_accuracies) / sum(examples)  
    weighted_f1 = sum(train_F1_all) / sum(examples)  
    best_train_loss= min(each_train_loss) 
    worst_train_loss= max(each_train_loss)  
    best_train_accuracy= max(each_train_accuracy)  
    worst_train_accuracy= min(each_train_accuracy)  
    max_data_heterogeneity = max(train_data_heterogeneities)  
    min_data_heterogeneity = min(train_data_heterogeneities)  
    max_data_skewness = max(train_data_skewnesses)  
    min_data_skewness = min(train_data_skewnesses)  
    peak_memory = max(memory_peaks) if memory_peaks else None 
    avg_train_lat = sum(latencies) / len(latencies) if latencies else None  
    avg_cpu = sum(cpu_list) / len(cpu_list) if cpu_list else None  
    
    history["train_loss"].append(weighted_loss)
    history["train_accuracy"].append(weighted_accuracy)
    history["train_f1"].append(weighted_f1)
    history["best_train_loss"].append(best_train_loss)
    history["worst_train_loss"].append(worst_train_loss)
    history["best_train_accuracy"].append(best_train_accuracy)
    history["worst_train_accuracy"].append(worst_train_accuracy)
    history["train_max_data_heterogeneity"].append(max_data_heterogeneity)
    history["train_min_data_heterogeneity"].append(min_data_heterogeneity)
    history["train_max_data_skewness"].append(max_data_skewness)  
    history["train_min_data_skewness"].append(min_data_skewness)
    history["train_memory_peak_mb"].append(peak_memory)
    history["train_client_latency"].append(avg_train_lat)
    history["train_avg_client_cpu"].append(avg_cpu)
    print(f"[伺服器訓練] 加權平均訓練損失: {weighted_loss:.4f}, 加權平均訓練準確: {weighted_accuracy:.4f}, 加權平均訓練F1-score: {weighted_f1:.4f}") 
    print(f"[伺服器訓練] 最佳訓練損失: {best_train_loss}, 最差訓練損失: {worst_train_loss}, 最佳訓練準確: {best_train_accuracy}, 最差訓練準確: {worst_train_accuracy}")
    print(f"[伺服器訓練] 客户端 RSS 峰值 (MB)：{peak_memory}")
    print(f"[伺服器訓練] 客戶端平均 CPU 使用率：{avg_cpu:.2f}%")
    print(f"[伺服器訓練] 平均客戶端延遲: {avg_train_lat:.4f} 秒")
    print(f"[伺服器訓練] 最大數據異質性程度: {max_data_heterogeneity}, 最小數據異質性程度: {min_data_heterogeneity}, 最大數據分佈變異係數 (CV): {max_data_skewness}, 最小數據分佈變異係數 (CV): {min_data_skewness}")

    return {"train_loss": weighted_loss,
            "train_accuracy": weighted_accuracy, 
            "train_f1": weighted_f1, 
            "best_train_loss": best_train_loss,
            "worst_train_loss": worst_train_loss,
            "best_train_accuracy": best_train_accuracy,
            "worst_train_accuracy": worst_train_accuracy,
            "train_max_data_heterogeneity": max_data_heterogeneity,
            "train_min_data_heterogeneity": min_data_heterogeneity,
            "train_max_data_skewness": max_data_skewness,
            "train_min_data_skewness": min_data_skewness,
            "train_memory_peak_mb": peak_memory,
            "train_client_latency": avg_train_lat,
            "train_avg_client_cpu": avg_cpu,
            }
    
def evaluate_metrics_calculate(metrics):

    test_losses = [num_examples * m["test_loss"] for num_examples, m in metrics] 
    test_accuracies = [num_examples * m["test_accuracy"] for num_examples, m in metrics]  
    test_F1_all = [num_examples * m["test_f1"] for num_examples, m in metrics]  
    each_test_loss = [m["test_loss"] for _, m in metrics]  
    each_test_accuracy = [m["test_accuracy"] for _, m in metrics]  
    test_data_heterogeneities = [m["test_data_heterogeneity"] for _, m in metrics]  
    test_data_skewnesses = [m["test_data_skewness"] for _, m in metrics]  
    inf_times = [m["inference_time"] for _, m in metrics]  
    memory_peaks = [m["memory_peak_mb"] for _, m in metrics if "memory_peak_mb" in m]  
    latencies = [m["client_latency"] for _, m in metrics if "client_latency" in m]  
    cpu_list = [m["client_cpu_percent"] for _, m in metrics if "client_cpu_percent" in m]  
    examples = [num_examples for num_examples, _ in metrics]
    
    weighted_loss = sum(test_losses) / sum(examples)  
    weighted_accuracy = sum(test_accuracies) / sum(examples)  
    weighted_f1 = sum(test_F1_all) / sum(examples)  
    best_test_loss=min(each_test_loss)  
    worst_test_loss=max(each_test_loss)  
    best_test_accuracy=max(each_test_accuracy)  
    worst_test_accuracy=min(each_test_accuracy)  
    max_data_heterogeneity = max(test_data_heterogeneities)  
    min_data_heterogeneity = min(test_data_heterogeneities)  
    max_data_skewness = max(test_data_skewnesses)  
    min_data_skewness = min(test_data_skewnesses)  
    avg_inf_time = sum(inf_times) / len(inf_times)  
    peak_memory = max(memory_peaks) if memory_peaks else None  
    avg_test_lat = sum(latencies) / len(latencies) if latencies else None 
    avg_cpu = sum(cpu_list) / len(cpu_list) if cpu_list else None  
    
    history["test_loss"].append(weighted_loss)
    history["test_accuracy"].append(weighted_accuracy)
    history["test_f1"].append(weighted_f1)
    history["best_test_loss"].append(best_test_loss)
    history["worst_test_loss"].append(worst_test_loss)
    history["best_test_accuracy"].append(best_test_accuracy)
    history["worst_test_accuracy"].append(worst_test_accuracy)
    history["test_max_data_heterogeneity"].append(max_data_heterogeneity)
    history["test_min_data_heterogeneity"].append(min_data_heterogeneity)
    history["test_max_data_skewness"].append(max_data_skewness)  
    history["test_min_data_skewness"].append(min_data_skewness)
    history["avg_inference_time"].append(avg_inf_time)
    history["test_memory_peak_mb"].append(peak_memory)
    history["test_client_latency"].append(avg_test_lat)
    history["test_avg_client_cpu"].append(avg_cpu)
    print(f"[伺服器測試] 加權平均測試損失: {weighted_loss:.4f}, 加權平均測試準確: {weighted_accuracy:.4f}, 加權平均測試F1-score:: {weighted_f1:.4f}")
    print(f"[伺服器測試] 最佳測試損失: {best_test_loss}, 最差測試損失: {worst_test_loss}, 最佳測試準確: {best_test_accuracy}, 最差測試準確: {worst_test_accuracy}")
    print("[伺服器測試] 各客戶端推論時間 (秒)：", [f"{t:.4f}" for t in inf_times])
    print(f"[伺服器測試] 平均推論時間：{avg_inf_time:.4f} 秒 ({avg_inf_time*1000:.1f} 毫秒)")
    print(f"[伺服器測試] 客户端 RSS 峰值 (MB)：{peak_memory}")
    print(f"[伺服器測試] 客戶端平均 CPU 使用率：{avg_cpu:.2f}%")
    print(f"[伺服器測試] 平均客戶端延遲: {avg_test_lat:.4f} 秒")
    print(f"[伺服器測試] 最大數據異質性程度: {max_data_heterogeneity}, 最小數據異質性程度: {min_data_heterogeneity}, 最大數據分佈變異係數 (CV): {max_data_skewness}, 最小數據分佈變異係數 (CV): {min_data_skewness}")

    return {"test_loss": weighted_loss,
            "test_accuracy": weighted_accuracy,
            "test_f1": weighted_f1,
            "best_test_loss": best_test_loss,
            "worst_test_loss": worst_test_loss,
            "best_test_accuracy": best_test_accuracy,
            "worst_test_accuracy": worst_test_accuracy,
            "test_max_data_heterogeneity": max_data_heterogeneity,
            "test_min_data_heterogeneity": min_data_heterogeneity,
            "test_max_data_skewness": max_data_skewness,
            "test_min_data_skewness": min_data_skewness,
            "avg_inference_time": avg_inf_time,
            "test_memory_peak_mb": peak_memory,
            "test_client_latency": avg_test_lat,
            "test_avg_client_cpu": avg_cpu,
            }

def save_history_to_csv():
    col_map = {
        "round":                        "輪次",
        "train_loss":                   "訓練損失",
        "best_train_loss":              "最佳訓練損失",
        "worst_train_loss":             "最差訓練損失",
        "train_accuracy":               "訓練準確",
        "best_train_accuracy":          "最佳訓練準確",
        "worst_train_accuracy":         "最差訓練準確",
        "train_max_data_heterogeneity": "訓練最大資料異質性",
        "train_min_data_heterogeneity": "訓練最小資料異質性",
        "train_max_data_skewness":      "訓練最大資料偏斜",
        "train_min_data_skewness":      "訓練最小資料偏斜",
        "test_loss":                    "測試損失",
        "best_test_loss":               "最佳測試損失",
        "worst_test_loss":              "最差測試損失",
        "test_accuracy":                "測試準確",
        "best_test_accuracy":           "最佳測試準確",
        "worst_test_accuracy":          "最差測試準確",
        "test_max_data_heterogeneity":  "測試最大資料異質性",
        "test_min_data_heterogeneity":  "測試最小資料異質性",
        "test_max_data_skewness":       "測試最大資料偏斜",
        "test_min_data_skewness":       "測試最小資料偏斜",
        "avg_inference_time":           "平均推論時間(秒)",
        "comm_bytes_per_round":         "通訊成本(bytes)",
        "messages_per_round":           "本輪訊息次數",
        "cumulative_messages":          "累計訊息次數",
        "model_size_bytes":             "模型大小(bytes)",
        "train_memory_peak_mb":         "客戶端訓練記憶體峰值(MB)",
        "test_memory_peak_mb":          "客戶端測試記憶體峰值(MB)",
        "train_client_latency":         "訓練平均延遲(秒)",
        "test_client_latency":          "測試平均延遲(秒)",
        "server_sched_fit_latency":     "伺服器排程訓練延遲(秒)",
        "server_agg_fit_latency":       "伺服器聚合訓練延遲(秒)",
        "server_sched_eval_latency":    "伺服器排程測試延遲(秒)",
        "server_agg_eval_latency":      "伺服器聚合測試延遲(秒)",
        "train_avg_client_cpu":         "客戶端訓練平均cpu使用率",
        "test_avg_client_cpu":          "客戶端測試平均cpu使用率",
        "total_time_seconds":           "聯邦學習總耗時",
        "train_f1":                     "訓練F1-score",
        "test_f1":                      "測試F1-score",
    }

    keys = ["round"] + [k for k in col_map.keys() if k != "round"] + ["sep"]
    rounds = list(range(1, len(history.get("train_loss", [])) + 1))
    data = {"round": rounds}
    for k in keys:
        if k == "round":
            continue
        elif k == "sep":
            data["sep"] = [""] * len(rounds)
        else:
            data[k] = history.get(k, [None] * len(rounds))
    df = pd.DataFrame(data)
    col_map_with_sep = dict(col_map)
    col_map_with_sep["sep"] = ""
    df = df.rename(columns=col_map_with_sep)

    desired_order = [
        "聯邦學習總耗時",
        "輪次",
        "訓練損失",
        "訓練準確",
        "測試損失",
        "測試準確",
        "平均推論時間(秒)",
        "訓練F1-score",
        "測試F1-score",
        "",
        "通訊成本(bytes)",
        "本輪訊息次數",
        "累計訊息次數",
        "模型大小(bytes)",
        "客戶端訓練記憶體峰值(MB)",
        "客戶端測試記憶體峰值(MB)",
        "客戶端訓練平均cpu使用率",
        "客戶端測試平均cpu使用率",
        "訓練平均延遲(秒)",
        "測試平均延遲(秒)",
        "",  
        "最佳訓練損失",
        "最差訓練損失",        
        "最佳訓練準確",
        "最差訓練準確",            
        "",                   
        "最佳測試損失",
        "最差測試損失",       
        "最佳測試準確",
        "最差測試準確",
        "",
        "伺服器排程訓練延遲(秒)",
        "伺服器聚合訓練延遲(秒)",
        "伺服器排程測試延遲(秒)",
        "伺服器聚合測試延遲(秒)",
        "",
        "訓練最大資料異質性",
        "訓練最小資料異質性",
        "訓練最大資料偏斜",        
        "訓練最小資料偏斜",
        "",
        "測試最大資料異質性",
        "測試最小資料異質性",
        "測試最大資料偏斜",
        "測試最小資料偏斜",
    ]
    cols_to_save = [c for c in desired_order if c in df.columns]
    df = df[cols_to_save]
    df.to_csv(server_history_log, index=False, encoding="utf-8-sig")
    print(f"歷史訊息已儲存至 {server_history_log}")

def visualize_training_progress():
    if not history["train_loss"] or not history["test_loss"] or not history["train_accuracy"] or not history["test_accuracy"]:
        print("[警告] 訓練和測試計算中沒有數據被儲存，無法繪製圖表！")
        return

    plt.figure(figsize=(14, 5))
    x_rounds = range(1, len(history["train_loss"]) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(x_rounds, history["train_loss"], marker='x', label='Local Train Loss')
    plt.plot(x_rounds, history["test_loss"], marker='o', label='Local Test Loss')
    plt.title(f"{dataset_name} {strategy_name} clients {num_expected_clients} Ratio {Ratio} - Average Local Loss")
    plt.xlabel("Rounds")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(x_rounds, history["train_accuracy"], marker='x', label='Local Train Accuracy')
    plt.plot(x_rounds, history["test_accuracy"], marker='o', label='Local Test Accuracy')
    plt.title(f"{dataset_name} {strategy_name} clients {num_expected_clients} Ratio {Ratio} - Average Local Accuracy")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.ylim(None, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(server_training_plot)
    print(f"訓練圖已儲存至 {server_training_plot}")
    
def visualize_confusion_matrices():
    class_labels = ["dos", "probe", "u2r", "r2l", "normal"]

    train_cm = history.get("aggregated_train_cm")
    if train_cm is not None:
        train_cm_df = pd.DataFrame(train_cm, index=class_labels, columns=class_labels)
        train_cm_df.to_csv(aggregated_train_cm_csv_path)
        print(f"匯總的訓練混淆矩陣數值已儲存至 {aggregated_train_cm_csv_path}")

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            train_cm_df, 
            annot=True, 
            fmt="d", 
            cmap="Blues"
        )
        plt.title("Aggregated Confusion Matrix - Training Data (Counts)")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        train_cm_path = aggregated_cm_plot.replace(".png", "_train_counts.png")
        plt.savefig(train_cm_path)
        print(f"匯總的訓練混淆矩陣圖(數值)已儲存至 {train_cm_path}")
        plt.close()

        with np.errstate(divide='ignore', invalid='ignore'):
            cm_percent = train_cm.astype('float') / train_cm.sum(axis=1)[:, np.newaxis]
            cm_percent = np.nan_to_num(cm_percent)

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm_percent,
            annot=True,
            fmt=".2%", 
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels
        )
        plt.title("Aggregated Confusion Matrix - Training Data (Percentage)")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        train_cm_percent_path = aggregated_cm_plot.replace(".png", "_train_percentage.png")
        plt.savefig(train_cm_percent_path)
        print(f"匯總的訓練混淆矩陣圖(百分比)已儲存至 {train_cm_percent_path}")
        plt.close()

    test_cm = history.get("aggregated_test_cm")
    if test_cm is not None:
        test_cm_df = pd.DataFrame(test_cm, index=class_labels, columns=class_labels)
        test_cm_df.to_csv(aggregated_test_cm_csv_path)
        print(f"匯總的測試混淆矩陣數值已儲存至 {aggregated_test_cm_csv_path}")

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            test_cm_df,
            annot=True,
            fmt="d",
            cmap="Blues"
        )
        plt.title("Aggregated Confusion Matrix - Testing Data (Counts)")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        test_cm_path = aggregated_cm_plot.replace(".png", "_test_counts.png")
        plt.savefig(test_cm_path)
        print(f"匯總的測試混淆矩陣圖(數值)已儲存至 {test_cm_path}")
        plt.close()

        with np.errstate(divide='ignore', invalid='ignore'):
            cm_percent = test_cm.astype('float') / test_cm.sum(axis=1)[:, np.newaxis]
            cm_percent = np.nan_to_num(cm_percent)

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm_percent,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels
        )
        plt.title("Aggregated Confusion Matrix - Testing Data (Percentage)")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        test_cm_percent_path = aggregated_cm_plot.replace(".png", "_test_percentage.png")
        plt.savefig(test_cm_percent_path)
        print(f"匯總的測試混淆矩陣圖(百分比)已儲存至 {test_cm_percent_path}")
        plt.close()

connected_clients = set()

def handle_client_connection(client_socket, client_address):
    try:
        data = client_socket.recv(1024).decode()
        if data.startswith("CLIENT_ID="):
            client_id = data.split("=")[1]
            connected_clients.add(client_id)
            print(f"客戶端 {client_id} 來自 {client_address} 已連接")
    except Exception as e:
        print(f"伺服器接收錯誤: {e}")
    finally:
        client_socket.close()

def start_client_listener():
    server_address = ("0.0.0.0", TCP_Port)  
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(server_address)
    server_socket.listen(50)

    print("客戶端監聽伺服器已啟動，等待客戶端連接...")
    
    while True:
        client_socket, client_address = server_socket.accept()
        client_thread = threading.Thread(target=handle_client_connection, args=(client_socket, client_address))
        client_thread.daemon = True
        client_thread.start()

threading.Thread(target=start_client_listener, daemon=True).start()
    

t_global_start = time.perf_counter()
fl.server.start_server(
    server_address=ip,
    config=fl.server.ServerConfig(num_rounds),
    grpc_max_message_length=1024 * 1024 * 100, 
    strategy=CustomStrategy(num_expected_clients)
)
t_global_end = time.perf_counter()
total_training_time = t_global_end - t_global_start
print(f"[伺服器] 整個聯邦學習總耗時：{total_training_time:.2f} 秒")
rounds_len = len(history["train_loss"])
history["total_time_seconds"] = [total_training_time] + [None] * (rounds_len - 1)
save_history_to_csv()
visualize_training_progress()
visualize_confusion_matrices()
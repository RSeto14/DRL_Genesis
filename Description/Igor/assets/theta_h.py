import numpy as np

def solve_y(A, B, x):
    """
    方程式 A*sin(x) = B*sin(y) を y について解く。
    一般解として y = arcsin(...) + 2nπ または y = π - arcsin(...) + 2nπ を返す。
    
    Parameters:
        A (float): 定数 A
        B (float): 定数 B
        x (float): 変数 x (ラジアン単位)
    
    Returns:
        tuple: y の解のリスト（一般解の形式で返す）
    """
    # k = (A / B) * sin(x)
    k = (A / B) * np.sin(x)
    
    # k の値が -1 から 1 の範囲内にあるか確認
    if abs(k) > 1:
        raise ValueError("解が存在しません：|A/B * sin(x)| が 1 を超えています。")
    
    # 主値解
    y1 = np.arcsin(k)
    y2 = np.pi - np.arcsin(k)
    
    # 一般解をリストで返す
    general_solution = {
        "y1": f"y = {y1} + 2nπ (n ∈ Z)",
        "y2": f"y = {y2} + 2nπ (n ∈ Z)"
    }
    return general_solution, y1, y2

# パラメータの例
L1 = 0.37433
L2 = 0.3292
# x = np.pi / 6  # x = 30度 (ラジアン)
th1 = 0.5

# 方程式を解く
try:
    print(f"L1: {L1}, L2: {L2}, th1: {th1}")
    solutions, y1, y2 = solve_y(L1, L2, th1)
    for key, sol in solutions.items():
        print(f"{key}: {sol}")

    th2 = y1 + th1
    print(f"th2: {-th2}")

    h = 0.06 + L1 * np.cos(th1) + L2 * np.cos(th2-th1) + 0.1016
    print(f"h: {h}")
except ValueError as e:
    print(e)

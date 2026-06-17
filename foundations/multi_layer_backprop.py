import numpy as np
from typing import List

class Solution:
    def forward_and_backward(self,
                             x: List[float],
                             W1: List[List[float]], b1: List[float],
                             W2: List[List[float]], b2: List[float],
                             y_true: List[float]) -> dict:
        
        # ==========================================
        # 1. SETUP (Chuyển list thành numpy array & reshape)
        # ==========================================
        X = np.array(x).reshape(-1, 1)
        W1 = np.array(W1)
        b1 = np.array(b1).reshape(-1, 1)
        W2 = np.array(W2)
        b2 = np.array(b2).reshape(-1, 1)
        Y_true = np.array(y_true).reshape(-1, 1)

        # ==========================================
        # 2. KHUNG INNER FUNCTIONS (Mày tự điền ruột)
        # ==========================================
        
        def linear_forward(A_prev, W, b):
            # TODO: Tính Z = W*A + b. Đừng quên tạo cache (A_prev, W, b).
            return W @ A_prev+b, [A_prev, W, b]

        def linear_backward(dZ, cache):
            # TODO: Lôi A_prev, W, b từ cache ra.
            # TODO: Tính dA_prev, dW, db dựa vào quy tắc Outer, Transpose tao đã chỉ.
            A_prev, W, b=cache

            dA_prev= W.T@dZ
            dW=dZ@A_prev.T
            db=np.sum(dZ,axis=1,keepdims=True)

            return dA_prev,dW,db

        def relu_forward(Z):
            # TODO: Tính A = max(0, Z) và lưu cache Z.
            return np.maximum(0,Z),Z

        def relu_backward(dA, cache):
            # TODO: Lôi Z từ cache ra. Chặn gradient (0) ở những chỗ Z <= 0.
            return dA*(cache>0) 
            

        def mse_forward_backward(Y_pred, Y_true):
            # TODO: Tính MSE loss (scalar).
            # TODO: Tính gradient khởi nguồn dY_pred.
            N=len(Y_pred)
            MSE=np.mean((Y_pred-Y_true)**2)
            dY_pred=2/N*(Y_pred-Y_true)
            return MSE,dY_pred

        # ==========================================
        # 3. DÂY CHUYỀN THỰC THI (Đã nối cáp sẵn cho mày)
        # ==========================================
        
        # -- Forward Pass --
        Z1, cache_L1 = linear_forward(X, W1, b1)
        A1, cache_relu = relu_forward(Z1)
        Z2, cache_L2 = linear_forward(A1, W2, b2)

        # -- Loss --
        loss, dZ2 = mse_forward_backward(Z2, Y_true)

        # -- Backward Pass --
        dA1, dW2, db2 = linear_backward(dZ2, cache_L2)
        dZ1 = relu_backward(dA1, cache_relu)
        _, dW1, db1 = linear_backward(dZ1, cache_L1)

        # ==========================================
        # 4. CHỐT SỔ & RETURN
        # ==========================================
        def fmt(arr, is_1d=False):
            arr_rounded = np.round(arr, 4)
            return arr_rounded.flatten().tolist() if is_1d else arr_rounded.tolist()

        return {
            'loss': round(float(loss), 4),
            'dW1': fmt(dW1),
            'db1': fmt(db1, is_1d=True),
            'dW2': fmt(dW2),
            'db2': fmt(db2, is_1d=True)
        }
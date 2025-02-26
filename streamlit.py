import streamlit as st
import numpy as np
import random
import time
import torch
from botorch.test_functions import Ackley
from botorch.models import SingleTaskGP

from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

@st.fragment
def col1_func():

    if ("run_button","col_1_executed") not in st.session_state:
        st.session_state.run_button = False
        st.session_state.col_1_executed = False

    if st.session_state.col_1_executed:
        st.stop()
        
    if st.button("実行"):
        st.session_state.run_button = True
        
        # 100サンプルを2次元[-1,1]領域から生成
        X = 2 * torch.rand(20, 2) - 1  # shape: (100, 2)
        # Ackley 関数の初期化（2次元用）
        ackley = Ackley(dim=2,negate=True)
        # 関数評価
        Y = ackley(X)
        #st.write("Ackley 関数の評価値:", Y)

        # 累積最大値を計算
        cummax = torch.empty_like(Y)
        cummax[0] = Y[0]
        for i in range(1, Y.shape[0]):
            cummax[i] = max(cummax[i - 1], Y[i])
        
        # 左右に分けてプロットを表示
        left_plot, right_plot = st.columns(2)
        
        with left_plot:

            grid_size = 100
            x = np.linspace(-1, 1, grid_size)
            y = np.linspace(-1, 1, grid_size)
            X_grid, Y_grid = np.meshgrid(x, y)
            grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
            grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
            Z = ackley(grid_tensor).detach().numpy().reshape(grid_size, grid_size)
            plt.contourf(X_grid, Y_grid, Z, levels=15, cmap='viridis', alpha=0.6)

            scatter = plt.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c='blue')
            plt.colorbar(scatter)
            plt.title("Ackley Function Response")
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            st.pyplot(plt)
            plt.close()
            
        with right_plot:
            plt.figure()
            plt.plot(cummax.numpy(), marker='o')
            plt.title("Ackley Function - Cumulative Maximum")
            plt.xlabel("Sample Index")
            plt.ylabel("Cumulative Maximum Value")
            plt.ylim(-10, 2)
            plt.axhline(y=0, color='red', linestyle='--', label='optimal')
            plt.legend(loc='best')
            st.pyplot(plt)
            plt.close()

        
        # グラフ（現在のFigure）をsession_stateに保存する
        if ("figures","col_2_executed") not in st.session_state:
            st.session_state["figures"] = []
        st.session_state["figures"]=plt
        best_value = Y.max().item()
        st.markdown(f"<h1><b>best value: {best_value:.2f}</b></h1>", unsafe_allow_html=True)

        st.session_state.col_1_executed = True

@st.fragment
def col2_func():
        if ("bayes_button_pressed","col_1_executed") not in st.session_state:
            st.session_state.bayes_button_pressed = False
            st.session_state.col_2_executed = False

        if st.session_state.col_2_executed:
            st.stop()

        if st.button("ベイズ最適化実行"):
            st.session_state.bayes_button_pressed = True

            left_plot, right_plot = st.columns(2)

            # Ackley関数の初期化（2次元用）
            ackley = Ackley(dim=2, negate=True)

            # 初期データ：5個のランダムな点を生成（評価値は列ベクトル形式）
            train_X = 2 * torch.rand(1, 2) - 1
            train_Y = ackley(train_X).unsqueeze(-1)

            # 探索領域の設定：各次元[-1, 1]
            bounds = torch.stack([torch.full((2,), -1.0), torch.full((2,), 1.0)])

            # Streamlit上でグラフを逐次更新するためのコンテナ
            chart_slot = st.empty()

            # 10回のベイズ最適化ループ
            for i in range(20):
                if i == 0:
                    left_chart = left_plot.empty()
                    right_chart = right_plot.empty()

                # GPモデルの構築と学習
                gp = SingleTaskGP(train_X, train_Y)
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)
                best_f = (train_Y).max().item()
                acq_func = LogExpectedImprovement(model=gp, best_f=best_f, maximize=True)
                candidate, _ = optimize_acqf(
                    acq_function=acq_func,
                    bounds=bounds,
                    q=1,
                    num_restarts=5,
                    raw_samples=20,
                )
                new_X = candidate.detach()
                new_Y = ackley(new_X).unsqueeze(-1)
                train_X = torch.cat([train_X, new_X], dim=0)
                train_Y = torch.cat([train_Y, new_Y], dim=0)

                # 左側のグラフを更新
                fig_left = plt.figure()


                grid_size = 100
                x = np.linspace(-1, 1, grid_size)
                y = np.linspace(-1, 1, grid_size)
                X_grid, Y_grid = np.meshgrid(x, y)
                grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
                grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
                Z = ackley(grid_tensor).detach().numpy().reshape(grid_size, grid_size)
                plt.contourf(X_grid, Y_grid, Z, levels=15, cmap='viridis', alpha=0.6)

                
                sc = plt.scatter(train_X[:, 0].numpy(), train_X[:, 1].numpy(), 
                                 c='blue')
                plt.colorbar(sc)
                plt.scatter(new_X[:, 0].numpy(), new_X[:, 1].numpy(), color='red', edgecolor='black', s=100, label='New Candidate')
                plt.title(f"Ackley Function - Iteration {i+1}")
                plt.xlim(-1, 1)
                plt.ylim(-1, 1)

                left_chart.pyplot(fig_left)
                plt.close(fig_left)

                # 右側のグラフを更新
                fig_right = plt.figure()
                cummax = np.maximum.accumulate(train_Y.numpy())
                plt.plot(cummax, marker="o")
                plt.title("Ackley Function - Cumulative Maximum")
                plt.xlabel("Sample Index")
                plt.ylabel("Cumulative Maximum Value")
                plt.ylim(-10, 2)
                plt.axhline(y=0, color='red', linestyle='--', label='optimal')
                plt.legend(loc='best')
                right_chart.pyplot(fig_right)
                plt.close(fig_right)

                # 事後分布の等高線を描画
                fig_posterior = plt.figure()
                with torch.no_grad():
                    posterior = gp.posterior(grid_tensor)
                    posterior_mean = posterior.mean.detach().numpy().reshape(grid_size, grid_size)
                plt.contourf(X_grid, Y_grid, posterior_mean, levels=15, cmap='coolwarm', alpha=0.6)
                plt.colorbar()
                plt.title(f"GP Posterior Mean - Iteration {i+1}")
                chart_slot.pyplot(fig_posterior)
                plt.close(fig_posterior)

                time.sleep(0.1)

            best_value = train_Y.max().item()
            st.markdown(f"<h1><b>best value: {best_value:.2f}</b></h1>", unsafe_allow_html=True)
            st.session_state.col_2_executed = True

with st.sidebar:
    st.title("サイドバー")
    st.write("ここにサイドバーの内容を追加可能です。")
# Streamlitスライドバー
dropdown_option = st.selectbox("画像表示オプション", options=["隠す", "表示する"], index=0)
if dropdown_option == "表示する":
    img = plt.imread("dial.jpg")  # 画像パスを指定
    st.image(img, caption="選択した画像", use_column_width=True)

# 2つのカラムを生成
col1, col2 = st.columns(2)


with col1:
    with st.container(height=600):
        st.title("HUMAN")
        col1_func()

with col2:
    with st.container(height=600):
        st.title("AI")
        col2_func()  # ベイズ最適化の実行

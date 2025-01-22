# Genesis使ってみた

## 参考

- [Genesisの公式ドキュメント](https://genesis-world.readthedocs.io/en/latest/index.html)
- [GenesisのGitHub](https://github.com/Genesis-Embodied-AI/Genesis)
- [誰かのQiita](https://qiita.com/hEnka/items/cc5fd872eb0bf7cd3abc)

## 0. 注意事項

- GPUを使うことが前提
- Linux環境でないと描画ができない(Windowsでは学習のみできる)
- Anacondaはインストール済みであること

## 1. 環境構築

```bash
conda create -n genesis-env python=3.11
conda activate genesis-env
```

## 2. ライブラリインストール

### 2.1. PyTorch

公式ページからインストール

<url>https://pytorch.org/</url>
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2.2. その他のライブラリ

```bash
pip install genesis-world
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl && git checkout v1.0.2 && pip install -e .
pip install tensorboard
pip install pytz
pip install PyYAML
```

## 3. Igor

### [Env.py](Scripts/Igor-RL/Env.py)

- 強化学習の環境を定義しているファイル
- 実行すると↓のように環境のテストを行い，動画を生成する

```bash
python Scripts/Igor-RL/Env.py
```

<video src="./Scripts/Igor-RL/env_video.mp4" controls></video>


### [Train.py](Scripts/Igor-RL/Train.py)

- 学習を行うファイル
- 実行するとLogsフォルダに学習ログが保存される

実行例
```bash
python Scripts/Igor-RL/Train.py
```
```bash
python Scripts/Igor-RL/Train.py --exp_name Igor-Turn --num_envs 4096 --max_iterations 200 --device 0
```

2000万ステップの学習を2分で終わらせることができる！

```bash
                       Learning iteration 199/200                       

                       Computation: 230036 steps/s (collection: 0.273s, learning 0.154s)
               Value function loss: 0.0004
                    Surrogate loss: 0.0030
             Mean action noise std: 0.29
                       Mean reward: 108.37
               Mean episode length: 1001.00
 Mean episode rew_tracking_lin_vel: 2.9442
 Mean episode rew_tracking_ang_vel: 2.8328
        Mean episode rew_lin_vel_z: -0.0123
      Mean episode rew_action_rate: -0.0419
Mean episode rew_similar_to_default: -0.2022
       Mean episode rew_leg_energy: -0.0101
--------------------------------------------------------------------------------
                   Total timesteps: 19660800
                    Iteration time: 0.43s
                        Total time: 94.80s
                               ETA: 0.5s
```

### [Eval.py](Scripts/Igor-RL/Eval.py)

- 学習したモデルを評価するファイル
- 実行するとLogsフォルダに動画が保存される
- 引数を指定しないと最新の学習ログからニューラルネットワークを読み込んで実行

実行例
```bash
python Scripts/Igor-RL/Eval.py
```

```bash
python Scripts/Igor-RL/Eval.py --log_dir Scripts/Igor-RL/Logs/250122_113438
```

<video src="./Scripts/Igor-RL/Logs/250122_113438/Test/250122_115822/video.mp4" controls></video>

# What is this?
[Matcha TTS](https://github.com/shivammehta25/Matcha-TTS)を日本語に特化させたものです。  

# 使い方
1. このリポジトリをクローンします。  
2. audio以下に音声ファイルを入れます。  
3. 以下の形式で`train_files.txt`を作ります。  
```
audioname.wav|0|おはようございます
音声ファイル(wavである必要はない)|話者id(単一話者向け学習なら0)|内容
```
4. `install.sh`の通りにdependenciesをインストールします。  
5. `python fast_preprocess.py`を実行して、データを前処理します。  
6. `python fast_train.py`を実行して、学習を開始します。  
    tensorboardで学習の進捗を確認できます。(`tensorboard --logdir logs --bind_all`)

# 気を付けたほうがいいこと
- 最初から大量のデータで学習すると、DurationPredictorがうまく学習できないことがあります。そのため、最初は少ないデータで学習し、徐々にデータを増やしていくことをおすすめします。
- hyperparamsはfast_hparam.pyにあります。適宜変更してください。
- 何かご不明な点があれば気軽に質問してください。

# ライセンス
このリポジトリはMITライセンスのもとで公開されています。
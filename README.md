漫画セリフ発話者推定プログラム
====

漫画に登場するセリフの発話者を推定するプログラム  
2019年度明治大学大学院中村研究室修士研究 阿部和樹

# 文献

セリフ発話者データセットおよび本リポジトリのコードを使用した論文を執筆される際には、下記文献を引用してください。

> - 阿部 和樹, 中村 聡史. コミックのセリフと発話者対応付けデータセットの構築とその困難性, 第3回 コミック工学研究会, pp.7-12, 2020.

# 準備

研究を再現するにあたり、本コードの他に以下の2つが必要となります。
- Manga109データセット本体 （[リンク](http://www.manga109.org/ja/download.html)）
- mecab + mecab-ipadic-Neologd （[リンク](https://github.com/neologd/mecab-ipadic-neologd/blob/master/README.ja.md)）

mecab-ipadic-Neologdは、形態素解析ライブラリであるMecabで使用する辞書です。  
macの場合、インストールは[こちらの記事](https://qiita.com/taroc/items/b9afd914432da08dafc8)を参考にするといいと思われます。
一人称および語尾による推定を使用しない場合は必要ありません。  
インストールするディレクトリは問いませんが、インストール先のパスがわかる場所にインストールするようにお願いします。

また、Manga109およびipadic-Neologdのバージョンによっては論文の結果と異なる値が出力される可能性があります。

# インストール

インストール前に`python -m venv venv && source venv/bin/activate`などを実行して仮想環境を作ることをお勧めします。

```Shell
$ pip install -r requirements.txt

$ cp .env.example .env

```

上記コマンドを実行後に`.env`の中身を編集してください。  
`MANGA109_ROOT_DIR`にはダウンロードしたManga109のディレクトリまでの絶対パス（`books.txt`が置いてあるディレクトリです）、
`MECAB_IPADIC_PATH`は辞書が入ったディレクトリ（標準だと`/usr/local/lib/mecab/dic/mecab-ipadic-neologd`だと思います）を指定します。

また、研究のために使用した発話者のデータセットも以下に公開します。  
- [dataset_all.csv（全てのキャラクタを対象としたセリフ）](https://nkmr.io/comic/speaker-dataset/public/datasets_all.csv)
- [dataset_top5.csv（出現率上位5名のキャラクタに絞ったセリフ）](https://nkmr.io/comic/speaker-dataset/public/datasets_top5.csv)

こちらは、[セリフ発話者データセット](https://nkmr.io/comic/speaker-dataset/)を本研究のために整形したものになります。
上記のデータをダウンロード後、本プロジェクトの`data/raw`以下に置いてください。

(※)コマンドで実行する例
```Shell
$ curl -o data/raw/datasets_all.csv https://nkmr.io/comic/speaker-dataset/public/datasets_all.csv
$ curl -o data/raw/datasets_top5.csv https://nkmr.io/comic/speaker-dataset/public/datasets_top5.csv
```

# 実行方法

## サンプル
修士論文の表6.2における「組み合わせの正解率」を得るための実行例です。  
**（2020/08/08 追記）計算方法のミスを修正したため、修士論文とは異なる結果が表示されます。**
```Shell
$ make calc data=all target=neighbor_nonface,inframe,taildirection3,firstperson,endingword

$ make predict data=all target=neighbor_nonface,inframe,taildirection3,firstperson,endingword

# all ['neighbor_nonface', 'inframe', 'taildirection3', 'firstperson', 'endingword']
# 0.7851257374509739


# 単一の手法による推定結果は以下によって得られる
$ make predict data=all target=neighbor_nonface

# all ['neighbor_nonface']
# .7433753007481626

```

各コマンドの詳細は以下を参照してください。

## スコア計算の実行

```Shell
# 全てのキャラクタを対象に、全ての手法の計算を行う
$ make calc data=all target=neighbor_nonface,inframe,taildirection3,firstperson,endingword

# neighbor_nonface: セリフとキャラクタの距離による計算
# inframe:           同じコマ内にいるキャラクタの情報による計算
# taildirection3:    吹き出しのしっぽ方向による計算
# firstperson:       一人称による計算
# endingword:        語尾による計算


# （任意）上位5名のキャラクタで計算する場合は、data=top5 に変更する
$ make calc data=top5 target=neighbor_nonface,inframe,taildirection3,firstperson,endingword


# （任意）targetの引数をカンマ区切りで指定することで手法を指定する
$ make calc data=all target=neighbor_nonface,inframe

```

`make cale <オプション>` によって、発話者を推定するためのスコアの計算を行います。  
実行時のオプション
- data: 使用するデータセットを指定します(`all`か`top5`を指定します)
- target: 計算の手法をカンマ区切りで指定します。指定された計算が順次実行されます。

windowsで実行する場合は、makeコマンドが正常に作動しない可能性が高いです。
その場合、以下を順に実行してください。
```Shell
$ python formatting.py all
$ python calc_neighbor_nonface.py all
$ python calc_inframe.py all
$ python calc_taildirection3.py all
$ python calc_firstperson.py all
$ python calc_endingword.py all

```

上記を実行することで、`data/scores`以下に値を記録したcsvファイルが作成されます。  
発話者の推定はこのスコアを記録したcsvファイルをもとに行われるため、先にこの計算を実行する必要があります。  
**（注意） 実行例の5種類の手法を計算する場合、マシンのスペックにもよりますが1時間以上の時間がかかります。**  
ただし、同じオプションで実行したものについては一度計算すればcsvファイルが残るため、再度実行する必要はありません。

## 推定の実行

```Shell
# 全てのキャラクタを対象に、全ての手法の計算を行う
$ make predict data=all target=neighbor_nonface,inframe,taildirection3,firstperson,endingword

# all ['neighbor_nonface', 'inframe', 'taildirection3', 'firstperson', 'endingword']
# 0.7851257374509739

# （任意）手法を限定する場合
$ make predict data=all target=inframe  # 同じコマによる手法
$ make predict data=all target=firstperson,endingword  # 一人称・語尾による手法

```

`make predict <オプション>`によって、発話者推定の正解率を標準出力に表示します。  
実行時のオプションは計算の実行と同様です。

同一の`data`および`target`のオプションによって`make calc`を実行しておくことで推定が可能となります。  
また、スコア計算の実行で全ての手法による計算が成功している場合、推定の方では単一の手法（例：`target=inframe`）でも実行が可能です。

以上により、論文と同様の結果が得られます。

# 検証環境

参考までに、動作検証を行った環境の情報を載せておきます。

- Python Version: 3.8.5
- macOS Catalina バージョン10.15.6

# リンク

- [コミックのセリフと発話者対応付けデータセットの構築とその困難性 - 明治大学 中村聡史研究室 論文レポジトリ](https://dl.nkmr-lab.org/papers/240)
- [漫画におけるセリフと発話者の対応付け手法の研究 - 明治大学 中村聡史研究室 論文レポジトリ](https://dl.nkmr-lab.org/papers/227)
- [セリフ発話者データセット配布サイト](https://nkmr.io/comic/speaker-dataset/)

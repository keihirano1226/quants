# ベースイメージ名:タグ名
FROM continuumio/anaconda3:2019.03

# pipをアップグレードし必要なパッケージをインストール.複数画面あったりすると後々便利だからtmuxも一応インストール。
RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install build-essential -y
RUN pip install --upgrade pip && \
    pip install pandas && \
    pip install mplfinance && \
    pip install numpy && \
    pip install pyti && \
    pip install opencv-python && \
    pip install scipy && \
    pip install matplotlib && \
    pip install seaborn && \
    pip install sklearn && \
    pip install jupyterlab && \
    pip install lightgbm && \
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xvzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install
RUN pip install TA-Lib
RUN rm -R ta-lib ta-lib-0.4.0-src.tar.gz
EXPOSE 8888

# ENTRYPOINT命令はコンテナ起動時に実行するコマンドを指定（基本docker runの時に上書きしないもの）
# "jupyter-lab" => jupyter-lab立ち上げコマンド
# "--ip=0.0.0.0" => ip制限なし
# "--port=8888" => EXPOSE命令で書いたポート番号と合わせる
# ”--no-browser” => ブラウザを立ち上げない。コンテナ側にはブラウザがないので 。
# "--allow-root" => rootユーザーの許可。セキュリティ的には良くないので、自分で使うときだけ。
# "--NotebookApp.token=''" => トークンなしで起動許可。これもセキュリティ的には良くない。
ENTRYPOINT ["jupyter-lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''","--no-mathjax","--NotebookApp.password=''"]
#ENTRYPOINT ["tmux"]

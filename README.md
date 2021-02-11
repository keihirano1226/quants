## docker環境構築方法 && quatsコンペ用ファイル構成
- ファイル説明
    - Dockerfile: 元となるkernel image に必要なライブラリ  
    (python関係及び必要であればtmuxなどのapt-get 関係)をインストールするためのもの。
    - docker-compose.yml: コンテナの起動時における引数をまとめたもの。  
    コンテナ外のフォルダをマウントしてファイルとして保存できるなど結構便利。
- ファイル構成
    work_dir/  
┠  quants_data_dir(コンペのデータセットを入れるフォルダ。git のカレントディレクトリの範囲外)  
┠  quants(gitでcloneしてくる中身)  
というようにしておくこと.  
上記を守っておくと、docker-compose.ymlの中身を書き換えなくて済む。
- docker コンテナ起動手順
    - docker build -t quants_forecast .
    - docker-compose up
- vscodeでのコンテナアクセス手順
    - vscodeの Remote Containersをインストールしておく。
    - F1キーから Remote Containers Attach to remote containersで起動中のコンテナを指定。  
    - フォルダとして/notebookディレクトリを開く。
    - コンテナ内にvscodeのpython拡張機能をインストール。
    - create new　notebookで新しいノートブックを作ると開発開始。


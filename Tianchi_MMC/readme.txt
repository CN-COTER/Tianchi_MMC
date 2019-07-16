model_relation
模型主要采用的是Bilstm+Attention
数据集格式为 关系索引 关系名 关系对起始位置 关系对结束位置
Demo中存放了测试数据集和训练数据集
model文件下存放了5个主模型文件:base_net.py,model_2c_base.py,model_2c_board.py
model_2c_ensemble.py,model_net.py
submit文件下存放最终测试的输入文件
run.sh为训练和测试脚本




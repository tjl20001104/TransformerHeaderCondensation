# Tranformer Header Condensation Research Based on Anchor function

主要代码结构如下：
    
    ```
    ├── data_generator 
    ├── model
    ├── utils
    ├── result
    ├── main.py
    ├── data.py
    ├── train.py
    ├── script.py
    ```

运行python script.py即可得到结果，其中script.py可以根据需要进行修改。


文件夹说明：

- data_generator: 用于生成数据集的代码
- model: 模型定义代码
- utils: 作图、设置随机数等工具代码
- result: 保存结果的文件夹

We found that during the early stage of training for Transformer models, Condensation phenomena exhibit the following characteristics:

    The maximum number of condensation directions for Q(K, V) parameter matrices belonging to the same multi-head self-attention module is two to the power of their respective embedding space dimension;
    
    Attention matrices for all self-attention heads in all layers are identical and serve to take average inline;
    
    Overall output matrices for self-attention heads belonging to the same multi-head selfattention module follow a similar pattern as those from their V modules in terms of condensation.

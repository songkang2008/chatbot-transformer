import os
import tensorflow as tf
file_train_input = './douban-multiturn-100w/train.txt'
file_train_target = './deal_data_train.txt'
file_test_input = './douban-multiturn-100w/test.txt'
file_test_target = './deal_data_test.txt'

def data_deal(input_file,target_file):
    seq_one = []
    with open(input_file, encoding='utf-8') as f:
        # 对数据进行处理，通过行数奇偶分为问和答
        for i, lines in enumerate(f):
            # print(i)
            # print(lines)
            if i%2==0:
                # 按照制表符分句
                lines = lines.split('\t')
                # 去掉前面的不要的
                lines = lines[1:]
                # print(len(lines))
                # 将问答固定成偶数对
                if len(lines)%2==1:
                    lines = lines[:-1]
                for j in range(len(lines)):
                    if j%2==0:
                        lines1 = ''.join(lines[j].split())
                        lines2 = ''.join(lines[j+1].split())
                        seq_one.append(lines1+ '\t'+lines2)
    # print(seq_one[1])
    with open(target_file, 'w', encoding='utf-8') as f:
        # 将处理好的数据集写到文件
        for seq in seq_one:
            f.write(seq+'\n')

data_deal(file_test_input, file_test_target)


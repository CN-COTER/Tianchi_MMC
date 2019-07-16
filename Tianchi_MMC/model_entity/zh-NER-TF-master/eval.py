import os


def conlleval(label_predict, label_path, metric_path):
    """

    :param label_predict:
    :param label_path:
    :param metric_path:
    :return:
    """
    #打开./conlleval_rev.pl，读取label
    eval_perl = "./conlleval_rev.pl"
    with open(label_path, "w") as fw:
        line = []
        #label_predict是打包好的标签label，tag等
        for sent_result in label_predict:
            for char, tag, tag_ in sent_result:
                print(char, tag, tag_)
                tag = '0' if tag == 'O' else tag
                char = char.encode("utf-8")
                line.append("{} {} {}\n".format(char, tag, tag_))
            line.append("\n")
        fw.writelines(line)
    os.system("perl {} < {} > {}".format(eval_perl, label_path, metric_path))
    with open(metric_path) as fr:
        metrics = [line.strip() for line in fr]
    #返回转移矩阵
    return metrics
    
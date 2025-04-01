import argparse

from flask import Flask, request, jsonify
import torch
from data_loader import RelationDataset
from multi_atten_model import Model
from config import Config
from data_loader import process_bert, collate_fn
from transformers import AutoTokenizer
import utils
from torch.utils.data import DataLoader

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./config/resume-zh.json')
parser.add_argument('--save_path', type=str, default='./Diakg_multi_atten_model2.pt')
# parser.add_argument('--save_path', type=str, default='./model.pt')
parser.add_argument('--predict_path', type=str, default='./output.json')
parser.add_argument('--device', type=int, default=0)

parser.add_argument('--dist_emb_size', type=int)
parser.add_argument('--type_emb_size', type=int)
parser.add_argument('--lstm_hid_size', type=int)
parser.add_argument('--conv_hid_size', type=int)
parser.add_argument('--bert_hid_size', type=int)
parser.add_argument('--ffnn_hid_size', type=int)
parser.add_argument('--biaffine_size', type=int)

parser.add_argument('--dilation', type=str, help="e.g. 1,2,3")

parser.add_argument('--emb_dropout', type=float)
parser.add_argument('--conv_dropout', type=float)
parser.add_argument('--out_dropout', type=float)

parser.add_argument('--epochs', type=int)
parser.add_argument('--batch_size', type=int)

parser.add_argument('--clip_grad_norm', type=float)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--weight_decay', type=float)

parser.add_argument('--bert_name', type=str)
parser.add_argument('--bert_learning_rate', type=float)
parser.add_argument('--warm_factor', type=float)

parser.add_argument('--use_bert_last_4_layers', type=int, help="1: true, 0: false")

parser.add_argument('--seed', type=int)
parser.add_argument('--unet_in_dim', type=int, default=3)
parser.add_argument('--unet_out_dim', type=int, default=256)
parser.add_argument('--down_dim', type=int, default=256)
args = parser.parse_args()

# 加载配置
config = Config(args)

# 初始化词汇表并设置 label_num
class Vocabulary(object):
    PAD = '<pad>'
    UNK = '<unk>'
    SUC = '<suc>'

    def __init__(self):
        self.label2id = {self.PAD: 0, self.SUC: 1}
        self.id2label = {0: self.PAD, 1: self.SUC}

    def add_label(self, label):
        label = label.lower()
        if label not in self.label2id:
            self.label2id[label] = len(self.label2id)
            self.id2label[self.label2id[label]] = label

        assert label == self.id2label[self.label2id[label]]

    def __len__(self):
        return len(self.token2id)

    def label_to_id(self, label):
        label = label.lower()
        return self.label2id[label]

    def id_to_label(self, i):
        return self.id2label[i]


# 从配置文件或预存文件加载标签列表（假设标签列表已知）
label_list = [
    "Disease", "Class", "Reason", "pathogenesis", "Symptom",
    "Test", "Test_Items", "Test_Value", "Drug", "Frequency",
    "Amount", "Method", "Treatment", "Operation", "ADE",
    "Anatomy", "Level", "Duration"
]  # 替换为你的实际标签

# 初始化词汇表
vocab = Vocabulary()
for label in label_list:
    vocab.add_label(label)

# 动态设置 config 的 label_num 和 vocab
config.label_num = len(vocab.label2id)
config.vocab = vocab

# 加载分词器
model_path = r".\bert_base_chinese"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 在 api.py 的模型加载部分添加以下代码
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
# 加载模型并移动到 GPU
model = Model(config).to(device)
# try:
#     state_dict = torch.load(config.save_path, map_location=device)
#     # 过滤不匹配的键
#     model_state_dict = model.state_dict()
#     matched_state_dict = {k: v for k, v in state_dict.items()
#                          if k in model_state_dict and v.size() == model_state_dict[k].size()}
#     model_state_dict.update(matched_state_dict)
#     model.load_state_dict(model_state_dict, strict=False)  # strict=False 忽略不匹配的键
#     model.eval()
# except Exception as e:
#     print(f"Error loading model: {e}")
#     # 可以在这里初始化新权重或退出


@app.route('/ner', methods=['POST'])
def ner():
    try:
        # 获取输入数据
        data = request.json
        sentence = data.get('sentence', '')

        if not sentence:
            return jsonify({'error': 'No sentence provided'}), 400

        # 处理输入数据
        processed_data = [{
            # 'sentence': list(sentence),
            'sentence': sentence,
            'ner': []
        }]

        # 数据预处理
        bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text = process_bert(
            processed_data, tokenizer, config.vocab
        )

        # 创建数据集和数据加载器，与my_main.py保持一致
        dataset = RelationDataset(bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            pin_memory=True
        )

        result = []
        with torch.no_grad():
            for data_batch in data_loader:
                entity_text = data_batch[-1]
                data_batch = [data.to(device) for data in data_batch[:-1]]
                bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

                outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
                length = sent_length

                outputs = torch.argmax(outputs, -1)
                ent_c, ent_p, ent_r, decode_entities = utils.decode(outputs.cpu().numpy(), entity_text, length.cpu().numpy())

                for ent_list, sentence_item in zip(decode_entities, processed_data):
                    sentence_text = ''.join(sentence_item["sentence"])
                    entities = []
                    for ent in ent_list:
                        start, end = ent[0][0], ent[0][-1]
                        entities.append({
                            'text': ''.join(sentence_item["sentence"][start:end+1]),
                            'type': vocab.id2label[ent[1]],
                            'start': start,
                            'end': end
                        })
                    result.append({
                        'sentence': sentence_text,
                        'entities': entities
                    })

        return jsonify(result[0] if result else {'sentence': sentence, 'entities': []})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
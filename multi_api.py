import argparse
import json

import torch
import transformers

from flask import Flask, request, jsonify
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import data_loader
import multi_atten_main
from multi_atten_model import Model
# from multi_atten_main import Trainer
from data_loader import process_bert, collate_fn, Vocabulary, RelationDataset
from config import Config
import utils

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
if torch.cuda.is_available():
    torch.cuda.set_device(args.device)
config = Config(args)
label_list = [
    "Disease", "Class", "Reason", "pathogenesis", "Symptom",
    "Test", "Test_Items", "Test_Value", "Drug", "Frequency",
    "Amount", "Method", "Treatment", "Operation", "ADE",
    "Anatomy", "Level", "Duration"
]
vocab = Vocabulary()
for label in label_list:
    vocab.add_label(label)
config.label_num=len(vocab.label2id)
config.vocab=vocab
model_path = r".\bert_base_chinese"
tokenizers = AutoTokenizer.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(config).to(device)
# model = model.cuda()
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 读入数据
        data = request.json

        process_data = [{
            'sentence': data['sentence'],
            'ner': []
        }]

        logger = utils.get_logger(config.dataset)
        logger.info(config)
        config.logger = logger
        datasets, ori_data = data_loader.load_data_bert(config)

        dataset = RelationDataset(*process_bert(process_data, tokenizers, vocab))
        dataloader = DataLoader(dataset=dataset,
                                batch_size=config.batch_size,
                                collate_fn=collate_fn,
                                shuffle=False,
                                num_workers=4,
                                drop_last=False,
                                pin_memory=True
                                )


        trainer = predict_trainer(model)
        # trainer = multi_atten_main.Trainer(model)
        trainer.load(config.save_path)
        result = trainer.predict("Final", dataloader, ori_data[-1])
        print(result)
        return jsonify(result)



    except Exception as e:
        return jsonify({'error': str(e)}), 500


class predict_trainer(object):
    def __init__(self, model):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        bert_params = set(self.model.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': config.learning_rate,
             'weight_decay': config.weight_decay},
        ]
        self.optimizer = torch.optim.Adam(params, lr=config.learning_rate,weight_decay=config.weight_decay)
        # self.optimizer = transformers.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        # self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                      # num_warmup_steps=config.warm_factor * updates_total,
                                                                      # num_training_steps=updates_total)
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
    def predict(self, epoch, data_loader, data):
        self.model.eval()

        # pred_result = []
        # label_result = []

        result = []

        # total_ent_r = 0
        # total_ent_p = 0
        # total_ent_c = 0

        i = 0
        with torch.no_grad():
            for data_batch in data_loader:
                sentence_batch = data[i:i + config.batch_size]
                entity_text = data_batch[-1]
                # data_batch = [
                #     data.cuda() if torch.cu da.is_available() else data
                #     for data in data_batch["-1"]
                # ]
                data_batch = [data.cuda() for data in data_batch[:-1]]
                # data_batch = [data for data in data_batch[:-1]]
                bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

                outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
                length = sent_length

                # grid_mask2d = grid_mask2d.clone()

                outputs = torch.argmax(outputs, -1)
                ent_c, ent_p, ent_r, decode_entities = utils.decode(outputs.cpu().numpy(), entity_text,
                                                                    length.cpu().numpy())

                for ent_list, sentence in zip(decode_entities, sentence_batch):
                    sentence = sentence["sentence"]
                    instance = {"sentence": sentence, "entity": []}
                    for ent in ent_list:
                        instance["entity"].append({"text": [sentence[x] for x in ent[0]],
                                                   "type": config.vocab.id_to_label(ent[1])})
                    result.append(instance)

                # total_ent_r += ent_r
                # total_ent_p += ent_p
                # total_ent_c += ent_c
                #
                # grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                # outputs = outputs[grid_mask2d].contiguous().view(-1)
                #
                # label_result.append(grid_labels.cpu())
                # pred_result.append(outputs.cpu())
                i += config.batch_size

        # label_result = torch.cat(label_result)
        # pred_result = torch.cat(pred_result)

        # p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
        #                                               pred_result.numpy(),
        #                                               average="macro")
        # e_f1, e_p, e_r = utils.cal_f1(total_ent_c, total_ent_p, total_ent_r)

        # title = "TEST"
        # logger.info('{} Label F1 {}'.format("TEST", f1_score(label_result.numpy(),
        #                                                     pred_result.numpy(),
        #                                                     average=None)))
        #
        # table = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])
        # table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        # table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [e_f1, e_p, e_r]])
        #
        # logger.info("\n{}".format(table))

        # with open(config.predict_path, "w", encoding="utf-8") as f:
        #     json.dump(result, f, ensure_ascii=False)

        return result


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# visualization libraries
import matplotlib.pyplot as plt
import numpy as np

# pytorch libraries
import torch # the main pytorch library
import torch.nn as nn # the sub-library containing Softmax, Module and other useful functions
import torch.optim as optim # the sub-library containing the common optimizers (SGD, Adam, etc.)
from torch.utils.data import DataLoader, TensorDataset

# huggingface's transformers library
from transformers import RobertaForTokenClassification, RobertaTokenizer

# huggingface's datasets library
from datasets import load_dataset

# the tqdm library used to show the iteration progress
import tqdm
tqdmn = tqdm.notebook.tqdm

import torch.optim as optim

roberta_version = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(roberta_version)

def read_ner_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences, labels = [], []
        current_sentence, current_labels = [], []
        for line in file:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    labels.append(current_labels)
                    current_sentence, current_labels = [], []
            else:
                parts = line.split()
                if len(parts) == 4:  # Đảm bảo rằng dòng có 4 phần
                    word, pos_tag, chunk_tag, ner_tag = parts
                    current_sentence.append(word)
                    current_labels.append(ner_tag)
        if current_sentence:
            sentences.append(current_sentence)
            labels.append(current_labels)
    return {'sentences': sentences, 'ner_labels': labels}

train_data = read_ner_data('/vn-ner-2023/train.txt')
test_data = read_ner_data('/vn-ner-2023/test.txt')
dev_data = read_ner_data('/vn-ner-2023/dev.txt')

# Xác định số lượng nhãn duy nhất và tạo từ điển ánh xạ
unique_labels = set(label for sublist in train_data['ner_labels'] for label in sublist)
num_labels = len(unique_labels)
label2id = {label: id for id, label in enumerate(unique_labels)}
id2label = {id: label for label, id in label2id.items()}

def add_encodings(example):
    # Mã hóa các token
    encodings = tokenizer(example['sentences'], truncation=True, padding='max_length', is_split_into_words=True)
    # Chuyển đổi nhãn NER thành chỉ số
    labels = [label2id[label] for label in example['ner_labels']]
    # Thêm nhãn phụ để đảm bảo độ dài nhất quán
    labels += [-100] * (tokenizer.model_max_length - len(labels))  # Sử dụng -100 để bỏ qua trong tính toán loss
    encodings['labels'] = labels  # Add labels to the encodings dictionary
    return encodings

train_encodings = [add_encodings({"sentences": s, "ner_labels": l}) for s, l in zip(train_data['sentences'], train_data['ner_labels'])]
test_encodings = [add_encodings({"sentences": s, "ner_labels": l}) for s, l in zip(test_data['sentences'], test_data['ner_labels'])]
dev_encodings = [add_encodings({"sentences": s, "ner_labels": l}) for s, l in zip(dev_data['sentences'], dev_data['ner_labels'])]

# Khởi tạo mô hình với số lượng nhãn
model = RobertaForTokenClassification.from_pretrained(roberta_version, num_labels=num_labels)
model.config.id2label = id2label
model.config.label2id = label2id

def create_dataloader(encodings):
    input_ids = [enc['input_ids'] for enc in encodings]
    attention_masks = [enc['attention_mask'] for enc in encodings]
    labels = [enc['labels'] for enc in encodings]
    
    dataset = TensorDataset(torch.tensor(input_ids),
                            torch.tensor(attention_masks),
                            torch.tensor(labels))
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Convert each batch to a dictionary
    dataloader = [{'input_ids': batch[0],
                   'attention_mask': batch[1],
                   'labels': batch[2]} for batch in dataloader]
    
    return dataloader

# Tạo DataLoader cho tập huấn luyện, kiểm thử và phát triển
train_loader = create_dataloader(train_encodings)
test_loader = create_dataloader(test_encodings)
dev_loader = create_dataloader(dev_encodings)



# Xác định thiết bị cho huấn luyện
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Đặt mô hình ở chế độ huấn luyện và chuyển nó đến thiết bị
model.train().to(device)

# Khởi tạo bộ tối ưu hóa
optimizer = optim.AdamW(params=model.parameters(), lr=1e-5)

n_epochs = 60  # Số lượng epoch
train_loss = []

for epoch in tqdmn(range(n_epochs)):
    current_loss = 0
    for i, batch in enumerate(tqdmn(train_loader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs[0]
        loss.backward()

        current_loss += loss.item()
        if i % 8 == 0 and i > 0:
            optimizer.step()
            optimizer.zero_grad()
            train_loss.append(current_loss / 32)
            current_loss = 0
    
    optimizer.step()
    optimizer.zero_grad()

# Trực quan hóa loss huấn luyện
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(train_loss)
ax.set_ylabel('Loss')
ax.set_xlabel('Iterations (32 examples)')
fig.tight_layout()
fig.savefig('loss.png', dpi=300)


# Lưu mô hình sau khi quá trình huấn luyện hoàn tất
output_model_file = "checkpoints/vietnamese_ner_model.bin"  # Đường dẫn lưu mô hình
output_config_file = "checkpoints/bert_config.json"  # Đường dẫn lưu cấu hình

model_to_save = model.module if hasattr(model, 'module') else model  # Xử lý cho DataParallel
torch.save(model_to_save.state_dict(), output_model_file)
model_to_save.config.to_json_file(output_config_file)
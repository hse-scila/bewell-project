import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertTokenizer
from transformers import get_linear_schedule_with_warmup
import time
import datetime
import random
import re
from scripts.metrics import get_metrics, plot_two_distributions
from scripts.bert import BertMSE


def clear_data(
    text
):
    return " ".join(re.findall("[A-Za-zА-Яа-я]+", text))
    
    
def prepare_data(
    sentences, 
    max_len=512, 
    tokenizer_path_or_name="bert-base-multilingual-cased",
    clear_texts=False
):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path_or_name)
    input_ids = []
    for sent in tqdm(sentences):
        if clear_texts:
            sent = clear_data(sent)
            
        encoded_sent = tokenizer.encode(sent, add_special_tokens = True)
        if encoded_sent and isinstance(encoded_sent, list):
            encoded_sent_padded = [0] * max_len
            for i, value in enumerate(encoded_sent[:max_len]):
                if isinstance(value, int):
                    encoded_sent_padded[i] = value
            input_ids.append(encoded_sent_padded)
            
    attention_masks = []
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    return input_ids, attention_masks


def format_time(
    elapsed
):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train_bert(
    train, 
    test, 
    target, 
    source,
    model=BertMSE,
    model_name_or_path="bert-base-multilingual-cased",
    batch_size=16,
    epochs=10,
    seed_val=42
):
    device = 'cuda:0'
    
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


    # prepare input: cut and pad
    train_input_ids, train_attention_masks = prepare_data(
        train[source].tolist()
    )
    test_input_ids, test_attention_masks = prepare_data(
        test[source].tolist()
    )

    # convert to tensor
    # train
    train_inputs = torch.tensor(train_input_ids)
    train_masks = torch.tensor(train_attention_masks)
    train_labels = torch.tensor(train[target].values.tolist(), dtype = torch.float)
    
    # valid
    validation_inputs = torch.tensor(test_input_ids)
    validation_masks = torch.tensor(test_attention_masks)
    validation_labels = torch.tensor(test[target].values.tolist(), dtype = torch.float)

    train_data = TensorDataset(
        train_inputs,
        train_masks,
        train_labels
    )
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data,
        sampler=train_sampler,
        batch_size=batch_size
    )

    validation_data = TensorDataset(
        validation_inputs,
        validation_masks,
        validation_labels
    )
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(
        validation_data,
        sampler=validation_sampler,
        batch_size=batch_size
    )

    model = model.from_pretrained(
        model_name_or_path, 
        output_attentions=False,
        output_hidden_states=True
    )
    
    model = model.to(device)
    
    optimizer = AdamW(
        model.parameters(),
        lr=2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
        eps=1e-8
    )

    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=50, # Default value in run_glue.py
        num_training_steps=total_steps
    )

    loss_train, loss_valid = [], []

    for epoch_i in range(0, epochs):

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i, epochs))
        print('Training...')
        total_loss = 0
        model.train()
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch_i}")
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                model.zero_grad()

                outputs = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels
                )

                loss = outputs[0]
                total_loss += loss.item()
                loss.backward()

                loss_train.append(loss.item())

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                tepoch.set_postfix(loss=loss.item())


        t0 = time.time()
        test_targets, test_pred_class = [], []

        model.eval()

        nb_eval_steps, nb_eval_examples = 0, 0
        valid_loss = 0

        for batch in validation_dataloader:

            batch = tuple(t.to(device) for t in batch)

            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():        
                outputs = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels
                )
            loss, logits = outputs[:2]
            
            valid_loss +=loss
            loss_valid.append(loss)
            

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.detach().cpu().numpy()
            test_targets.append(label_ids)
            test_pred_class.append(logits)

            # Calculate the accuracy for this batch of test sentences.          
            # Accumulate the total accuracy.

            nb_eval_steps += 1
            
        avg_valid_loss = valid_loss / len(validation_dataloader)            

        print("")
        print("  Average valid loss: {0:.2f}".format(avg_valid_loss))

        test_targets = np.concatenate(test_targets).squeeze()
        test_pred_class = np.concatenate(test_pred_class).squeeze()

        get_metrics(test_targets, test_pred_class)
        plot_two_distributions(test_targets, test_pred_class)


        print("  Validation took: {:}".format(format_time(time.time() - t0)))

    print("")
    print("Training complete!")
    
    plt.plot(loss_train)
    plt.title('Train loss')
    plt.show()
    
    plt.plot(loss_valid)
    plt.title('Valid loss')
    plt.show()
    return model

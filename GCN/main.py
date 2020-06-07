# coding=utf-8

"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
from collections import OrderedDict
import csv
import logging
import os
import random
import sys
import pandas as pd

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss

from file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from modeling_gloss import BertForSequenceClassification, BertConfig
# from tokenization import BertTokenizer
from optimization import BertAdam, warmup_linear


logger = logging.getLogger(__name__)


class BertOutputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, bert_pooled_output, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.bert_pooled_output = bert_pooled_output
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class WSD_sent_Processor(DataProcessor):
    """Processor for the WSD data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        #train_data = pd.read_csv(data_dir, sep="\t", na_filter=False).values
        train_data = np.load(data_dir, allow_pickle=True)
        train_data = train_data[()]
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        #dev_data = pd.read_csv(data_dir, sep="\t", na_filter=False).values
        dev_data = np.load(data_dir, allow_pickle=True)
        dev_data = dev_data[()]
        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, data, set_type): # TODO: fix
        """Creates examples for the training and dev sets."""
        examples = []
        length = data['embeddings'].shape[0]
        for i in range(length):
            guid = "%s-%s" % (set_type, i)

            # TODO: float32 or float16 or float64?
            pooled_output = data['embeddings'][i]
            label = int(data['labels'][i])
            # pooled_output = np.array(line[:-1], dtype=np.float32)
            # label = int(line[-1])
            if i%1000==0:
                print(i)
                # print("guid=",guid)
                # print("pooled_output=",pooled_output)
                # print("label=",label)
            examples.append(
                BertOutputExample(guid=guid, bert_pooled_output=pooled_output, label=label))
        return examples


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        choices=["WSD"],
                        help="The name of the task to train.")
    parser.add_argument("--train_data_dir",
                        default=None,
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--eval_data_dir",
                        default=None,
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")
    # parser.add_argument("--bert_model", default=None, type=str, required=True,
    #                     help='''a path or url to a pretrained model archive containing:
    #                     'bert_config.json' a configuration file for the model
    #                     'pytorch_model.bin' a PyTorch dump of a BertForPreTraining instance''')
    parser.add_argument("--config", default=None, type=str, required=True,
                        help="a path or url to a configuration file for the model.")
    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run test on the test set.")
    # parser.add_argument("--do_lower_case",
    #                     default=False,
    #                     action='store_true',
    #                     help="Whether to lower case the input text. True for uncased models, False for cased models.")
    # parser.add_argument("--max_seq_length",
    #                     default=128,
    #                     type=int,
    #                     help="The maximum total input sequence length after WordPiece tokenization. \n"
    #                          "Sequences longer than this will be truncated, and sequences shorter \n"
    #                          "than this will be padded.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    # parser.add_argument('--fp16',
    #                     action='store_true',
    #                     help="Whether to use 16-bit float precision instead of 32-bit")
    # parser.add_argument('--loss_scale',
    #                     type=float, default=0,
    #                     help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
    #                          "0 (default value): dynamic loss scaling.\n"
    #                          "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')


    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    # logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
    #     device, n_gpu, bool(args.local_rank != -1), args.fp16))
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: False".format(
        device, n_gpu, bool(args.local_rank != -1)))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_test` must be True.")
    if args.do_train:
        assert args.train_data_dir != None, "train_data_dir can not be None"
    if args.do_eval:
        assert args.eval_data_dir != None, "eval_data_dir can not be None"

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    # prepare dataloaders
    processors = {
        "WSD":WSD_sent_Processor
    }

    # output_modes = {
    #     "WSD": "classification"
    # }

    processor = processors[args.task_name]()
    # output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # training set
    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.train_data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()


    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))

    config = BertConfig.from_json_file(args.config)
    model = BertForSequenceClassification(config, 2)

    # if args.fp16:
    #     model.half()
    model.to(device)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)


    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    # if args.fp16:
    #     try:
    #         from apex.optimizers import FP16_Optimizer
    #         from apex.optimizers import FusedAdam
    #     except ImportError:
    #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    #     optimizer = FusedAdam(optimizer_grouped_parameters,
    #                           lr=args.learning_rate,
    #                           bias_correction=False,
    #                           max_grad_norm=1.0)
    #     if args.loss_scale == 0:
    #         optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
    #     else:
    #         optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    # else:
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)



    # load data
    if args.do_train:
        # train_features = convert_examples_to_features(
        #     train_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_inputs = torch.tensor([f.bert_pooled_output for f in train_examples], dtype=torch.float)
        labels = torch.tensor([f.label for f in train_examples], dtype=torch.long)
        # all_segment_ids = torch.tensor([f.segment_ids for f in train_examples], dtype=torch.long)

        # if output_mode == "classification":
        #     labels = torch.tensor([f.label for f in train_examples], dtype=torch.long)
        # elif output_mode == "regression":
        #     labels = torch.tensor([f.label for f in train_examples], dtype=torch.float)

        train_data = TensorDataset(all_inputs, labels)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)


    if args.do_eval:
        eval_examples = processor.get_dev_examples(args.eval_data_dir)
        # eval_features = convert_examples_to_features(
        #     eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_inputs = torch.tensor([f.bert_pooled_output for f in train_examples], dtype=torch.float)
        labels = torch.tensor([f.label for f in train_examples], dtype=torch.long)

        # if output_mode == "classification":
        #     all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        # elif output_mode == "regression":
        #     all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

        # eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        eval_data= TensorDataset(all_inputs, labels)
        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size, shuffle=False)




    # train
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    loss_fct = CrossEntropyLoss()

    if args.do_train:
        model.train()
        epoch = 0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            epoch += 1
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                inputs, labels = batch

                logits = model(inputs, labels=None)

                # if output_mode == "classification":
                loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
                # elif output_mode == "regression":
                #     loss_fct = MSELoss()
                #     loss = loss_fct(logits.view(-1), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                # if args.fp16:
                #     optimizer.backward(loss)
                # else:
                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += inputs.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # if args.fp16:
                    #     # modify learning rate with special warm up BERT uses
                    #     # if args.fp16 is False, BertAdam is used that handles this automatically
                    #     lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                    #     for param_group in optimizer.param_groups:
                    #         param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1


            # Save a trained model, configuration and tokenizer
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

            # If we save using the predefined names, we can load using `from_pretrained`
            model_output_dir = os.path.join(args.output_dir, str(epoch))
            if not os.path.exists(model_output_dir):
                os.makedirs(model_output_dir)
            output_model_file = os.path.join(model_output_dir, WEIGHTS_NAME)
            output_config_file = os.path.join(model_output_dir, CONFIG_NAME)

            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            # tokenizer.save_vocabulary(model_output_dir)



            if args.do_eval:
                model.eval()
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0
                loss_fct = CrossEntropyLoss()

                with open(os.path.join(args.output_dir, "results_"+str(epoch)+".txt"),"w") as f:
                    for inputs, labels in tqdm(eval_dataloader, desc="Evaluating"):
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        with torch.no_grad():
                            logits = model(inputs, labels=None)

                        logits_ = F.softmax(logits, dim=-1)
                        logits_ = logits_.detach().cpu().numpy()
                        labels_ = labels.to('cpu').numpy()
                        outputs = np.argmax(logits_, axis=1)
                        for output_i in range(len(outputs)):
                            f.write(str(outputs[output_i]))
                            for ou in logits_[output_i]:
                                f.write(" " + str(ou))
                            f.write("\n")
                        tmp_eval_accuracy = np.sum(outputs == labels_)

                        # create eval loss and other metric required by the task
                        # if output_mode == "classification":
                        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
                        # elif output_mode == "regression":
                        #     loss_fct = MSELoss()
                        #     tmp_eval_loss = loss_fct(logits.view(-1), labels.view(-1))

                        eval_loss += tmp_eval_loss.mean().item()
                        eval_accuracy += tmp_eval_accuracy
                        nb_eval_examples += inputs.size(0)
                        nb_eval_steps += 1

                eval_loss = eval_loss / nb_eval_steps
                eval_accuracy = eval_accuracy / nb_eval_examples
                loss = tr_loss/nb_tr_steps if args.do_train else None

                result = OrderedDict()
                result['eval_loss'] = eval_loss
                result['eval_accuracy'] = eval_accuracy
                result['global_step'] = global_step
                result['loss'] = loss

                output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                with open(output_eval_file, "a+") as writer:
                    writer.write("epoch=%s\n"%str(epoch))
                    logger.info("***** Eval results *****")
                    for key in result.keys():
                        logger.info("  %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))



    if args.do_test and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples(args.eval_data_dir)
        # eval_features = convert_examples_to_features(
        #     eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_inputs = torch.tensor([f.bert_pooled_output for f in eval_examples], dtype=torch.float)

        # if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in eval_examples], dtype=torch.long)
        # elif output_mode == "regression":
        #     all_labels = torch.tensor([f.label for f in eval_examples], dtype=torch.float)

        eval_data = TensorDataset(all_inputs, all_labels)
        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size, shuffle=False)



        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        loss_fct = CrossEntropyLoss()

        with open(os.path.join(args.output_dir, "results.txt"),"w") as f:
            for inputs, labels in tqdm(eval_dataloader, desc="Evaluating"):
                # segments = segments.to(device)
                labels = labels.to(device)
                inputs = inputs.to(device)

                with torch.no_grad():
                    logits = model(inputs, labels=None)

                logits_ = F.softmax(logits, dim=-1)
                logits_ = logits_.detach().cpu().numpy()
                labels_ = labels.to('cpu').numpy()
                outputs = np.argmax(logits_, axis=1)
                for output_i in range(len(outputs)):
                    f.write(str(outputs[output_i]))
                    for ou in logits_[output_i]:
                        f.write(" " + str(ou))
                    f.write("\n")
                tmp_eval_accuracy = np.sum(outputs == labels_)

                # create eval loss and other metric required by the task
                # if output_mode == "classification":
                tmp_eval_loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
                # elif output_mode == "regression":
                #     loss_fct = MSELoss()
                #     tmp_eval_loss = loss_fct(logits.view(-1), labels.view(-1))

                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy
                nb_eval_examples += inputs.size(0)
                nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        loss = tr_loss/nb_tr_steps if args.do_train else None

        result = OrderedDict()
        result['eval_loss'] = eval_loss
        result['eval_accuracy'] = eval_accuracy
        result['global_step'] = global_step
        result['loss'] = loss

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "a+") as writer:
            logger.info("***** Eval results *****")
            for key in result.keys():
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == "__main__":
    main()
